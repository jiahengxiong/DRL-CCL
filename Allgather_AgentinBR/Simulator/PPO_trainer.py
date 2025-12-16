# ppo_trainer.py
import math
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.distribution import Distribution
Distribution.set_default_validate_args(False)

import random
from copy import deepcopy


# -----------------------
# 工具函数
# -----------------------
def _normalize_probs(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.maximum(p, 0)
    s = p.sum()
    if s <= 0:
        # 退化：均匀
        p = np.ones_like(p) / len(p)
    else:
        p = p / s
    return p

def _mc_returns(rewards, gamma: float):
    """蒙特卡洛回报：支持'只有最后一步reward'或逐步奖励。"""
    R = 0.0
    out = []
    for r in reversed(rewards):
        R = float(r) + gamma * R
        out.append(R)
    out.reverse()
    return np.asarray(out, dtype=np.float32)

def _prep_critic_input(x: torch.Tensor) -> torch.Tensor:
    """
    将 x 变为 critic 接受的形状：[L, D] 或 [B, L, D]
    若 x 是 [B, L, K, D]（例如 [T, L, 2, 6]），对 K 维求均值 -> [B, L, D]。
    若维度更多（[B, L, *, D]），先把中间维合并成 K，再对 K 求均值。
    """
    if x.dim() == 2:        # [L, D]
        return x
    if x.dim() == 3:        # [B, L, D]
        return x
    if x.dim() == 4:        # [B, L, K, D]
        return x.mean(dim=2)
    # 一般情形：[B, L, *, D]
    B, L = x.shape[0], x.shape[1]
    D = x.shape[-1]
    x = x.view(B, L, -1, D)
    return x.mean(dim=2)

def _reduce_critic_output(v: torch.Tensor) -> torch.Tensor:
    """
    将 critic 输出压成 [B]（或 [T]）的一维向量：
    - 若 v 是 [B, L] 或 [B, ...]，对后续维取均值。
    - 若 v 是标量，unsqueeze 成 [1]。
    """
    if v.dim() == 0:
        return v.unsqueeze(0)
    if v.dim() == 1:
        return v
    reduce_dims = tuple(range(1, v.dim()))
    return v.mean(dim=reduce_dims)


# -----------------------
# Gossip helpers
# -----------------------
def _extract_edges_from_action_set(action_set):
    """
    action_set: {br_id: {idx: (src_br, dst_br), ...}, ...}
    返回去重后的无向边列表，如 [(0,2), (0,3), ...]
    仅保留 agent_set 覆盖到的 BR 节点的边（在构建邻居时再过滤）。
    """
    edges = set()
    for _, mapping in action_set.items():
        for _, pair in mapping.items():
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            u, v = pair
            if u == v:
                continue
            a, b = (u, v) if u <= v else (v, u)
            edges.add((a, b))
    return sorted(list(edges))

def _build_neighbors_from_edges(edges, nodes):
    """
    根据无向边和节点集合构建邻接表（仅包含 nodes 内的节点）。
    返回: {node: [邻居...]}
    """
    node_set = set(nodes)
    neighbors = {n: set() for n in node_set}
    for u, v in edges:
        if u in node_set and v in node_set:
            neighbors[u].add(v)
            neighbors[v].add(u)
    return {n: sorted(list(nei)) for n, nei in neighbors.items()}

def _compute_degrees(neighbors_of):
    return {n: len(nei) for n, nei in neighbors_of.items()}

@torch.no_grad()
def one_gossip_round(agent_set, neighbors_of, mode: str = "metropolis"):
    """
    同步 gossip 一轮：对每个节点 i，θ_i ← Σ_j W_ij θ_j。
    mode: "uniform" | "metropolis"
    """
    br_ids = list(agent_set.keys())
    # 取快照，防止读写覆盖
    snaps = {i: deepcopy(agent_set[i].state_dict()) for i in br_ids}
    deg = _compute_degrees(neighbors_of)

    for i in br_ids:
        nb = neighbors_of.get(i, [])
        if mode == "uniform":
            Zi = max(1, len(nb) + 1)
            weights = {j: 1.0 / Zi for j in nb}
            weights[i] = 1.0 / Zi
        elif mode == "metropolis":
            weights = {}
            for j in nb:
                weights[j] = 1.0 / (1.0 + max(deg.get(i, 0), deg.get(j, 0)))
            weights[i] = 1.0 - sum(weights.values())
        else:
            raise ValueError(f"Unknown gossip mode: {mode}")

        # 加权融合参数
        mixed = {}
        for k, v in snaps[i].items():
            if torch.is_tensor(v):
                acc = torch.zeros_like(v)
                for j, w in weights.items():
                    acc = acc + w * snaps[j][k]
                mixed[k] = acc
            else:
                mixed[k] = v
        agent_set[i].load_state_dict(mixed, strict=True)

def run_gossip(agent_set, neighbors_of, rounds: int = 1, mode: str = "metropolis"):
    for _ in range(max(1, rounds)):
        one_gossip_round(agent_set, neighbors_of, mode=mode)

def _reset_actor_optim_states(optim_actor_dict):
    for opt in optim_actor_dict.values():
        opt.state.clear()


@dataclass
class PPOConfig:
    gamma: float = 0.995
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    target_kl: float = 0.02
    update_epochs: int = 8
    minibatch_size: int = 2048
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    device: str = "auto"  # "cpu" | "cuda" | "mps" | "auto"


class PPOTrainer:
    """
    使用方法（示例）：
        trainer = PPOTrainer(config=PPOConfig())
        trainer.train(
            simulation=simulation,                 # 你的函数
            num_iterations=200,
            agent_set=agent_set,                  # {br_id: actor_module}
            critics_value=critics_value,          # 共享 critic 模块
            collective_time=collective_time,      # 你的仿真参数
            policy=[],                            # 按需传
            action_set=action_set,
            optim_actor_dict=None,                # 也可不传，内部自动建
            optim_critic=None,                    # 也可不传，内部自动建
            save_every=10,                        # 可选，周期保存
            save_dir="/path/to/ckpt"              # 可选
        )
    """

    def __init__(self, config: Optional[PPOConfig] = None):
        self.cfg = config or PPOConfig()

    # --------- 对外主入口 ----------
    def train(
        self,
        simulation,                      # 必传：你的仿真函数
        num_iterations: int,
        agent_set: Dict[Any, torch.nn.Module],
        critics_value: torch.nn.Module,
        collective_time: Any,
        policy: Any,
        action_set: Any,
        optim_actor_dict: Optional[Dict[Any, torch.optim.Optimizer]] = None,
        optim_critic: Optional[torch.optim.Optimizer] = None,
        save_every: int = 0,
        save_dir: Optional[str] = None,
        sim_kwargs: Optional[dict] = None,
        verbose: bool = True,
        # ==== gossip 开关 ====
        enable_gossip: bool = False,
        gossip_every: int = 5,
        gossip_rounds: int = 1,
        gossip_scheme: str = "metropolis",  # or "uniform"
        reset_optim_after_gossip: bool = False,
    ):
        device = self._resolve_device()
        critics_value.to(device)
        for a in agent_set.values():
            a.to(device)
        # torch.autograd.set_detect_anomaly(True)  # disabled for speed/stability

        # 优化器（可外部传入，也可内部创建）
        if optim_actor_dict is None:
            optim_actor_dict = {
                br: torch.optim.Adam(actor.parameters(), lr=self.cfg.actor_lr)
                for br, actor in agent_set.items()
            }
        if optim_critic is None:
            optim_critic = torch.optim.Adam(critics_value.parameters(), lr=self.cfg.critic_lr)

        sim_kwargs = sim_kwargs or {}

        # 依据 action_set 预抽边并构建邻接（仅包含已注册的 agent 节点）
        edges = _extract_edges_from_action_set(action_set)
        neighbors_of = _build_neighbors_from_edges(edges, agent_set.keys())
        if verbose and enable_gossip:
            num_edges = sum(len(v) for v in neighbors_of.values()) // 2
            print(f"[Gossip] enabled: |nodes|={len(agent_set)} |edges|={num_edges} scheme={gossip_scheme} every={gossip_every} rounds={gossip_rounds}")

        for it in range(num_iterations):
            # 1) 跑一集（一次 All-Gather），收集 BR_ac
            BR_ac = simulation(collective_time, policy, agent_set, action_set, **sim_kwargs)

            # 2) 整理 & 计算优势
            per_br = self._collate(BR_ac, device)
            if not per_br:
                if verbose:
                    print(f"[Iter {it}] empty episode, skip.")
                continue
            self._compute_advantages(per_br, critics_value)

            # 3) PPO 更新
            self._ppo_update(per_br, agent_set, critics_value, optim_actor_dict, optim_critic)

            # === Gossip: 每隔若干次 PPO 更新执行若干轮局部平均 ===
            if enable_gossip and ((it + 1) % gossip_every == 0):
                run_gossip(agent_set, neighbors_of, rounds=gossip_rounds, mode=gossip_scheme)
                if reset_optim_after_gossip:
                    _reset_actor_optim_states(optim_actor_dict)
                if verbose:
                    print(f"[Iter {it}] gossip done (rounds={gossip_rounds}, scheme={gossip_scheme})")

            # 4) 简单日志
            if verbose:
                total_steps = int(sum(d["actions"].numel() for d in per_br.values()))
                rough_return = float(sum(d["returns"][0].item() for d in per_br.values()))
                print(f"[Iter {it}] steps={total_steps}  rough_ep_ret_sum={rough_return:.3f}")

            # 5) 可选保存
            if save_every and save_dir and (it + 1) % save_every == 0:
                try:
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    for br, actor in agent_set.items():
                        torch.save(actor.state_dict(), f"{save_dir}/BR_{br}_actor.pt")
                    torch.save(critics_value.state_dict(), f"{save_dir}/Critic.pt")
                    if verbose:
                        print(f"[Iter {it}] checkpoints saved to {save_dir}")
                except Exception as e:
                    if verbose:
                        print(f"[Iter {it}] save failed: {e}")

    # --------- 内部：展平 batch ----------
    def _collate(self, BR_ac, device):
        per_br = {}
        for br_id, traj in BR_ac.items():
            if not traj:
                continue
            link_obs, glob_obs, acts, old_logp, rewards = [], [], [], [], []
            for (link_feats, global_feats, act, probs, reward) in traj:
                probs = np.asarray(probs, dtype=np.float64)  # 直接使用 BR_ac 里的概率
                # 保障 old_logp 数值稳定：先做一次概率归一化并裁剪下界
                probs = _normalize_probs(probs)
                probs = np.clip(probs, 1e-12, 1.0)
                a = int(act) if not isinstance(act, int) else act
                link_obs.append(np.asarray(link_feats, dtype=np.float32))
                glob_obs.append(np.asarray(global_feats, dtype=np.float32))
                acts.append(a)
                old_logp.append(float(math.log(probs[a])))
                rewards.append(float(reward))

            returns = _mc_returns(rewards, gamma=self.cfg.gamma)

            per_br[br_id] = {
                "link_obs": torch.tensor(np.stack(link_obs), dtype=torch.float32, device=device),  # [T, L]
                "glob_obs": torch.tensor(np.stack(glob_obs), dtype=torch.float32, device=device),  # [T, G]
                "actions": torch.tensor(acts, dtype=torch.long, device=device),                    # [T]
                "old_logp": torch.tensor(old_logp, dtype=torch.float32, device=device),            # [T]
                "returns": torch.tensor(returns, dtype=torch.float32, device=device),              # [T]
            }
        return per_br

    # --------- 内部：A = R - V(s)（全局标准化） ----------
    @torch.no_grad()
    def _compute_advantages(self, per_br, critic):
        all_adv = []
        for d in per_br.values():
            x = _prep_critic_input(d["glob_obs"])  # 适配形状，如 [T,L,2,6] -> [T,L,12]
            v_raw = critic(x)
            v = _reduce_critic_output(v_raw)  # 压成 [T]      # [T]
            d["value"] = v
            d["adv"] = d["returns"] - v
            all_adv.append(d["adv"])

        if not all_adv:
            return

        all_adv = torch.cat(all_adv, dim=0)
        mean, std = all_adv.mean(), all_adv.std().clamp_min(1e-6)
        for d in per_br.values():
            d["adv"] = (d["adv"] - mean) / std

    # --------- 内部：一次 PPO 更新 ----------
    def _ppo_update(self, per_br, agent_set, critic, optim_actor_dict, optim_critic):
        device = next(critic.parameters()).device

        # ---- 共享 Critic 先训练 ----
        all_glob = torch.cat([d["glob_obs"] for d in per_br.values()], dim=0)
        all_glob = _prep_critic_input(all_glob).contiguous()
        all_ret  = torch.cat([d["returns"] for d in per_br.values()],  dim=0)

        for _ in range(self.cfg.update_epochs):
            perm = torch.randperm(all_glob.size(0), device=device)
            for i in range(0, len(perm), self.cfg.minibatch_size):
                idx = perm[i:i+self.cfg.minibatch_size]
                x_mb = all_glob[idx].detach().contiguous()
                v_raw = critic(x_mb)
                v = _reduce_critic_output(v_raw)
                v_loss = F.mse_loss(v, all_ret[idx])
                optim_critic.zero_grad(set_to_none=True)
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optim_critic.step()

        # ---- 各 BR 的 Actor 分别训练 ----
        for br_id, actor in agent_set.items():
            if br_id not in per_br:
                continue
            d = per_br[br_id]
            T = d["actions"].size(0)
            if T == 0:
                continue

            opt = optim_actor_dict[br_id]
            stop_early = False
            for _ in range(self.cfg.update_epochs):
                if stop_early:
                    break
                perm = torch.randperm(T, device=device)
                for i in range(0, T, self.cfg.minibatch_size):
                    idx = perm[i:i+self.cfg.minibatch_size]
                    logits = actor(d["link_obs"][idx].detach().contiguous())     # 假定 actor 输出 logits 向量
                    # 数值稳健：清理 NaN/Inf，并限制量级，避免分布验证/梯度异常
                    logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
                    logits = torch.clamp(logits, -50.0, 50.0)
                    # 可选：去掉常数偏移（softmax 平移不变）。保留该行或直接用原始 logits 均可。
                    logits = logits - logits.logsumexp(dim=-1, keepdim=True)
                    dist = Categorical(logits=logits)
                    logp = dist.log_prob(d["actions"][idx])
                    with torch.no_grad():
                        approx_kl = (d["old_logp"][idx] - logp).mean()
                    if approx_kl > self.cfg.target_kl:
                        stop_early = True
                        break
                    ratio = torch.exp(logp - d["old_logp"][idx])

                    surr1 = ratio * d["adv"][idx]
                    surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * d["adv"][idx]
                    pg_loss = -torch.min(surr1, surr2).mean()

                    entropy = dist.entropy().mean()
                    loss = pg_loss - self.cfg.ent_coef * entropy
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    opt.step()

    # -----------------------
    def _resolve_device(self):
        if self.cfg.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.cfg.device)