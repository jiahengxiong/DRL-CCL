# ======= utils/ppo_trainer.py =======
from dataclasses import dataclass
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ---------------------------
# Config & State
# ---------------------------
@dataclass
class PPOConfig:
    lr: float = 1e-5
    clip_eps: float = 0.2
    value_coef: float = 0.1
    entropy_coef: float = 0.01
    maximize_reward: bool = True
    # Ban 掉的 reward 阈值（用于 worst buffer）
    worst_threshold: float = 0.0
    reward_scale: float = 1000  # Controls reward scaling (divide rewards by this value for optimization)

    # 探索噪声（围绕 actor 输出 se 的高斯）
    init_action_std: float = 0.25
    min_action_std: float = 5e-2
    std_anneal_steps: int = 15000  # 逐步退火到最小
    # 探索采样相关参数
    restart_prob: float = 1.0   # 比例: 每轮样本里 random restart 的概率
    strong_perturb_prob: float = 0.0   # 比例: 每轮样本里 strong perturb 的概率
    strong_perturb_steps: int = 6        # 远眺扰动的步数
    # 训练步数（每步一次 roll-out + 一次更新）
    train_steps: int = 1000
    num_action_samples: int = 64

    # 结构化扰动（围绕 mu 的非高斯、保持稀疏/重排）
    use_structured_perturb: bool = True
    per_row_ops: int = 1
    per_col_ops: int = 1
    row_swap_prob: float = 1          # 行内交换 一个0 和 一个>0 的位置
    col_swap_prob: float = 1          # 某列内交换 两个>0 的位置（跨不同行）
    row_zero_to_small_prob: float = 0.0 # 将行内若干0 置为小随机数
    row_pos_to_zero_prob: float = 0.1   # 将行内若干>0 置为0
    granularity: str = "soft"         # "hard" or "soft" granularity for structured_perturb

    sample_topk_per_col: int = 0      # keep top-k per column for sampled actions before renorm (0=disabled)

    # K-样本优势聚合：0=softmax权重；>0 用 top-k 均匀权重（如1=best-of-K）
    topk_surrogate: int = 8

    cvar_q_start = 0.4  # 前期平均一些
    cvar_q_mid_step = 400
    cvar_q_mid = 0.6
    cvar_q_end_step = 800
    cvar_q_end = 0.8

    # Elite buffer (keep a small set of high-return actions)
    elite_buf_size: int = 128
    elite_mix_prob: float = 0.25       # probability to mix elite samples each iteration
    elite_pick_per_iter: int = 3       # how many elite actions to inject/select per iteration

    # KL 防撞墙（可选）
    target_kl: float = 0.02
    kl_adapt: bool = False
    kl_warmup_steps: int = 50
    lr_min: float = 1e-5
    lr_max: float = 1e-3

    # 优势标准化用的滑窗
    adv_norm_window: int = 256

    sample_temp: float = 0.1
    # 昨天那版里用到的“远跳样本”配方
    restart_ratio: float = 0.25       # 每轮 K 个样本里有 25% 使用远跳（random/强扰动）
    vmix_alpha: float = 0.5           # critic 目标: R_target = α*R_best + (1-α)*R_wmean

    # # 更激进的探索默认值（先不退火，熵小幅度约束）
    # init_action_std: float = 0.30
    # min_action_std: float = 0.25
    # std_anneal_steps: int = 0
    # entropy_coef: float = 1e-3
    # kl_adapt: bool = False
    n_policy_epochs: int = 4
    device: str = ("cuda" if torch.cuda.is_available()
                   else ("mps" if torch.backends.mps.is_available() else "cpu"))


@dataclass
class PPOState:
    old_mu: Optional[torch.Tensor] = None
    step: int = 0
    # 滑窗优势，做标准化
    adv_buf: deque = deque(maxlen=256)
    # 学习率自适应
    lr_scale: float = 1.0
    # elite action buffer: list of (reward, action)
    elite_buf: list = None
    # worst action buffer: list of (reward, action)
    worst_buf: list = None


def build_optimizer(model: nn.Module, cfg: PPOConfig):
    return torch.optim.Adam(model.parameters(), lr=cfg.lr)


# ---------------------------
# 工具：log prob / entropy / KL
# ---------------------------
def normal_logp_entropy(a: torch.Tensor, mu: torch.Tensor, std: torch.Tensor):
    dist = Normal(mu, std)
    # 先对列(边)求和，再对行(subchunk)求平均，得到标量
    logp = dist.log_prob(a).sum(dim=1).mean()
    ent  = dist.entropy().sum(dim=1).mean()
    return logp, ent

def normal_kl(mu_old: torch.Tensor, mu_new: torch.Tensor, std: torch.Tensor):
    # 同方差的 Normal KL(old||new)：(Δμ^2) / (2σ^2)
    kl = ((mu_old - mu_new)**2 / (2.0 * (std**2 + 1e-12))).sum(dim=1).mean()
    return kl

# Batched logp/entropy for [K,S,E]
# ---------------------------
# Batched logp/entropy for [K,S,E]
# ---------------------------
def normal_logp_entropy_batched(a: torch.Tensor, mu: torch.Tensor, std: torch.Tensor):
    """
    a:  [K,S,E] actions
    mu: [S,E] (will be broadcast to [K,S,E]) or [K,S,E]
    std: scalar tensor or broadcastable to [K,S,E]
    returns: logp_per_sample [K], ent_per_sample [K]
    """
    if mu.dim() == 2:
        mu = mu.unsqueeze(0).expand_as(a)
    if std.dim() == 0:
        std = std.view(1, 1, 1)
    std = std.expand_as(a)
    dist = Normal(mu, std)
    # sum over E then mean over S -> per-sample scalar
    logp = dist.log_prob(a).sum(dim=2).mean(dim=1)
    ent  = dist.entropy().sum(dim=2).mean(dim=1)
    return logp, ent

# ---------------------------
# 结构化扰动工具
# ---------------------------
@torch.no_grad()
def renorm_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """可选：行归一化到[0,1]且不改变0位（如果行全0则保持）。实际默认不在采样里调用，仅供外部调试。"""
    s = x.sum(dim=1, keepdim=True)
    mask = (s > eps).float()
    x_out = x / (s + (1.0 - mask) * 1.0)  # 行和>0 则正规化；否则除以1等于不变
    return x_out.clamp(0.0, 1.0)

# === 远跳样本相关工具 ===
@torch.no_grad()
def random_restart_like(mu: torch.Tensor) -> torch.Tensor:
    """Uniform random in [0,1], then列归一，作为远跳起点。"""
    x = torch.rand_like(mu)
    return renorm_cols(x)

@torch.no_grad()
def strong_perturb_multi(mu: torch.Tensor, cfg: PPOConfig, steps=None) -> torch.Tensor:
    """从 mu 出发，连续多次结构化强扰动，再做列归一。"""
    if steps is None:
        steps = getattr(cfg, "strong_perturb_steps", 6)
    x = mu.clone()
    for _ in range(steps):
        x = structured_perturb(x, cfg)
    return renorm_cols(x)
# ---------------------------
# Column-wise renorm utility
# ---------------------------
@torch.no_grad()
def topk_per_col(x: torch.Tensor, k: int) -> torch.Tensor:
    S, E = x.shape
    if k <= 0 or k >= S:
        return x
    idx = torch.topk(x, k=k, dim=0).indices  # [k,E]
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(0, idx, True)
    return x * mask.float()

# ---------------------------
# Column-wise renorm utility
# ---------------------------
@torch.no_grad()
def renorm_cols(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Column-wise normalize to keep per-edge scale comparable while preserving zeros.
    For each column e, if sum over rows > 0, divide that column by its sum.
    This keeps the relative size of subchunks within the same column meaningful.
    """
    s = x.sum(dim=0, keepdim=True)             # [1,E]
    mask = (s > eps).float()
    x_out = x / (s + (1.0 - mask))             # if column-sum==0 => divide by 1 (no-op)
    return x_out.clamp(0.0, 1.0)


# === CVaR weighting utilities ===
@torch.no_grad()
def get_cvar_q(cfg: PPOConfig, step: int) -> float:
    """Piecewise schedule for CVaR quantile q based on training step."""
    if step >= getattr(cfg, 'cvar_q_end_step', 300):
        return float(getattr(cfg, 'cvar_q_end', 0.8))
    if step >= getattr(cfg, 'cvar_q_mid_step', 150):
        return float(getattr(cfg, 'cvar_q_mid', 0.7))
    return float(getattr(cfg, 'cvar_q_start', 0.6))

@torch.no_grad()
def cvar_weights_from_rewards(R: torch.Tensor, q: float, eps: float = 1e-8):
    """Compute CVaR-style weights from rewards: w_i ∝ max(0, R_i - τ_q). Returns (w,[threshold, cvar])."""
    assert R.dim() == 1, "R must be a vector [K]"
    # quantile threshold τ
    tau = torch.quantile(R, q)
    # head surplus
    surplus = (R - tau).clamp(min=0.0)
    ssum = surplus.sum()
    if ssum <= eps:
        # fallback: put all mass on best (best-of-K)
        w = torch.zeros_like(R)
        w[torch.argmax(R)] = 1.0
        cvar = R.max()
        return w, float(tau.item()), float(cvar.item())
    w = surplus / ssum
    # CVaR: average over head samples (where surplus>0)
    head = R[surplus > 0]
    cvar = head.mean() if head.numel() > 0 else R.max()
    return w, float(tau.item()), float(cvar.item())

@torch.no_grad()
def _row_swap_zero_pos(mat: torch.Tensor, row_idx: int):
    row = mat[row_idx]
    pos_idx = (row > 0).nonzero(as_tuple=False).flatten()
    zero_idx = (row == 0).nonzero(as_tuple=False).flatten()
    if len(pos_idx) == 0 or len(zero_idx) == 0:
        return
    i = zero_idx[torch.randint(0, len(zero_idx), (1,)).item()]
    j = pos_idx[torch.randint(0, len(pos_idx), (1,)).item()]
    tmp = row[i].item()
    row[i] = row[j]
    row[j] = tmp

@torch.no_grad()
def _col_swap_two_pos(mat: torch.Tensor, col_idx: int):
    col = mat[:, col_idx]
    pos_rows = (col > 0).nonzero(as_tuple=False).flatten()
    if len(pos_rows) < 2:
        return
    i, j = torch.randperm(len(pos_rows))[:2]
    ri = pos_rows[i].item()
    rj = pos_rows[j].item()
    tmp = mat[ri, col_idx].item()
    mat[ri, col_idx] = mat[rj, col_idx]
    mat[rj, col_idx] = tmp

@torch.no_grad()
def _row_zero_to_small(mat: torch.Tensor, row_idx: int, soft: bool = False):
    row = mat[row_idx]
    zero_idx = (row == 0).nonzero(as_tuple=False).flatten()
    if len(zero_idx) == 0:
        return
    # Flip all zeros in the row
    chosen = zero_idx
    if soft:
        row[chosen] = 0.01 * torch.rand_like(row[chosen])  # (0,0.01)
    else:
        row[chosen] = torch.rand_like(row[chosen])  # (0,1)

@torch.no_grad()
def _row_pos_to_zero(mat: torch.Tensor, row_idx: int, prob: float = 0.1, soft: bool = False):
    """
    For each >0 element in the row, with probability `prob`, perturb:
      - in soft mode: halve the value
      - in hard mode: set to zero
    """
    row = mat[row_idx]
    pos_idx = (row > 0).nonzero(as_tuple=False).flatten()
    if len(pos_idx) == 0:
        return
    # For each pos_idx, draw a Bernoulli(prob)
    mask = torch.rand(len(pos_idx), device=row.device) < prob
    if mask.any():
        chosen = pos_idx[mask]
        if soft:
            row[chosen] = row[chosen] * 0.5
        else:
            row[chosen] = 0.0

@torch.no_grad()
def structured_perturb(mu: torch.Tensor, cfg: PPOConfig) -> torch.Tensor:
    """在 mu 附近做结构化扰动：行/列的交换与稀疏化/致密化。返回同形状张量，范围[0,1]。支持 granularity: "hard" 或 "soft"。"""
    x = mu.clone()
    S, E = x.shape

    granularity = getattr(cfg, "granularity", "hard")
    if granularity == "soft":
        row_swap_prob = 0.2
        col_swap_prob = 0.2
        row_zero_to_small_prob = 0.05
        row_pos_to_zero_prob = 0.1
        soft = True
    else:
        row_swap_prob = getattr(cfg, "row_swap_prob", 1.0)
        col_swap_prob = getattr(cfg, "col_swap_prob", 1.0)
        row_zero_to_small_prob = getattr(cfg, "row_zero_to_small_prob", 0.1)
        row_pos_to_zero_prob = getattr(cfg, "row_pos_to_zero_prob", 0.25)
        soft = False

    # 行操作
    n_row_ops = max(0, int(getattr(cfg, "per_row_ops", 1)))
    for _ in range(n_row_ops):
        r = torch.randint(0, S, (1,)).item()
        u = torch.rand(())
        if u < row_swap_prob:
            _row_swap_zero_pos(x, r)
        elif u < row_swap_prob + row_zero_to_small_prob:
            _row_zero_to_small(x, r, soft=soft)
        elif u < row_swap_prob + row_zero_to_small_prob + row_pos_to_zero_prob:
            _row_pos_to_zero(
                x,
                r,
                prob=row_pos_to_zero_prob,
                soft=soft,
            )
        # else: no-op
    # 列操作
    n_col_ops = max(0, int(getattr(cfg, "per_col_ops", 1)))
    for _ in range(n_col_ops):
        c = torch.randint(0, E, (1,)).item()
        if torch.rand(()) < col_swap_prob:
            _col_swap_two_pos(x, c)
    return x.clamp(0.0, 1.0)
# ---------------------------
# 单步 PPO 更新
# ---------------------------
# ---------------------------
# 单步 PPO 更新（在此函数内重新计算新策略）
# ---------------------------
def ppo_update_multi(
    agent: nn.Module,
    v_old: torch.Tensor,              # scalar V_old(s)
    actions: torch.Tensor,            # [K,S,E]
    rewards: Union[torch.Tensor, List[float]],  # [K]
    optimizer: torch.optim.Optimizer,
    state: PPOState,
    cfg: PPOConfig,
    std_now: torch.Tensor,
):
    device = cfg.device
    assert state.old_mu is not None, "old_mu 尚未初始化"

    # Convert rewards to tensor [K]
    if not torch.is_tensor(rewards):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    rewards = rewards.to(device).view(-1)  # [K]

    # Keep a copy of unscaled rewards for logging
    rewards_raw = rewards.clone().detach()

    # Static reward scaling (divide by cfg.reward_scale for optimization, keep raw for logging)
    rewards = rewards / cfg.reward_scale

    # Logging helper: mean raw reward over K samples (不参与优化)
    R_raw_mean = rewards_raw.mean()

    # Sign for optimization
    R = rewards if cfg.maximize_reward else -rewards  # [K]

    # Old value scalar
    v_old = v_old.view(-1)[0].to(device)

    # Advantages per sample [K]
    advantages = (R - v_old).detach()

    # Push into sliding window for normalization
    for a in advantages.tolist():
        state.adv_buf.append(a)
    if len(state.adv_buf) > 8:
        m = sum(state.adv_buf) / len(state.adv_buf)
        var = sum((x - m) ** 2 for x in state.adv_buf) / max(1, len(state.adv_buf) - 1)
        adv_mean = torch.tensor(m, device=device)
        adv_std  = torch.tensor(max(1e-6, var ** 0.5), device=device)
    else:
        adv_mean = advantages.mean()
        adv_std  = advantages.std(unbiased=False) + 1e-6
    if adv_std > 1e-6:
        advantages_n = ((advantages - adv_mean) / adv_std).clamp(-5.0, 5.0)
    else:
        advantages_n = advantages

    # --- Weighting across K samples ---
    if getattr(cfg, 'use_cvar_weight', True):
        q = get_cvar_q(cfg, state.step)
        w, R_q, R_cvar = cvar_weights_from_rewards(R, q)
    else:
        # fallback to previous soft weighting or top-k
        tau = max(1e-3, float(getattr(cfg, 'sample_temp', 0.5)))
        with torch.no_grad():
            if getattr(cfg, 'topk_surrogate', 0) and cfg.topk_surrogate > 0:
                k = min(cfg.topk_surrogate, advantages_n.numel())
                topk_vals, topk_idx = torch.topk(advantages_n, k=k, largest=True, sorted=False)
                w = torch.zeros_like(advantages_n)
                w[topk_idx] = 1.0 / k
            else:
                w = torch.softmax(advantages_n / tau, dim=0)
        R_q = float(torch.quantile(R, 0.5).item())
        R_cvar = float((w * R).sum().item())

    std_now = std_now.to(device)
    actions = actions.to(device)  # [K,S,E]

    # --- Deterministic forward (freeze dropout etc.) to compute current policy BEFORE update ---
    was_training = agent.training
    agent.train(False)
    mu_before, v_before = agent.forward()
    agent.train(was_training)
    if isinstance(mu_before, tuple):
        mu_before = mu_before[0]
    mu_before = mu_before.to(device)         # [S,E]
    v_before = v_before.view(-1)[0].to(device)

    # Log-probs per sample against old policy and current policy (before update)
    logp_old, _      = normal_logp_entropy_batched(actions, state.old_mu.to(device), std_now)
    logp_new, ent_k  = normal_logp_entropy_batched(actions, mu_before,               std_now)
    ratio = torch.exp(torch.clamp(logp_new - logp_old, -10.0, 10.0))  # [K]

    # PPO clipped objective (weighted over K)
    surr1 = ratio * advantages_n
    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages_n
    policy_loss = -(w * torch.min(surr1, surr2)).sum()

    # Critic target: mix of best and CVaR (risk-seeking)
    if getattr(cfg, 'use_cvar_weight', True):
        R_wmean = (w * R).sum()
        R_best  = R.max()
        alpha = float(getattr(cfg, 'vmix_alpha', 0.5))
        R_target = alpha * R_best + (1.0 - alpha) * R_wmean
    else:
        R_wmean = (w * R).sum()
        R_best  = R.max()
        alpha = float(getattr(cfg, 'vmix_alpha', 0.5))
        R_target = alpha * R_best + (1.0 - alpha) * R_wmean
    value_loss = F.mse_loss(v_before, R_target)

    # Entropy, also weighted
    ent = (w * ent_k).sum().clamp(min=0.0)

    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * ent

    # Optimize
    for g in optimizer.param_groups:
        g['lr'] = max(cfg.lr_min, min(cfg.lr_max, cfg.lr * state.lr_scale))
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    optimizer.step()

    # Recompute policy AFTER update for KL/adaptive LR and logging
    with torch.no_grad():
        was_training = agent.training
        agent.train(False)
        mu_after, v_after = agent.forward()
        agent.train(was_training)
        if isinstance(mu_after, tuple):
            mu_after = mu_after[0]
        mu_after = mu_after.to(device)
        # # KL
        # kl = normal_kl(state.old_mu.to(device), mu_after, std_now if std_now.dim()==0 else std_now[0,0,0])
        # # ratio AFTER update (避免日志恒为1)
        # logp_old_after, _ = normal_logp_entropy_batched(actions, state.old_mu.to(device), std_now)
        # logp_new_after, _ = normal_logp_entropy_batched(actions, mu_after,                std_now)
        # ratio_after = torch.exp(torch.clamp(logp_new_after - logp_old_after, -10.0, 10.0))
        # ratio AFTER update (避免日志恒为1)
        # 1) 先把变量名拼对
        logp_old_after, _ = normal_logp_entropy_batched(actions, state.old_mu.to(device), std_now)
        logp_new_after, _ = normal_logp_entropy_batched(actions, mu_after, std_now)
        ratio_after = torch.exp(torch.clamp(logp_new_after - logp_old_after, -10.0, 10.0))

        # 2) 把 KL 从无权平均 → 带权平均（与损失里相同的 w）
        kl = (w * (logp_old_after - logp_new_after)).sum().clamp(min=0.0)
        if cfg.kl_adapt and state.step > getattr(cfg, "kl_warmup_steps", 0):
            if kl > 1.5 * cfg.target_kl:
                state.lr_scale = max(0.5 * state.lr_scale, cfg.lr_min / cfg.lr)
            elif kl < 0.5 * cfg.target_kl:
                state.lr_scale = min(1.5 * state.lr_scale, cfg.lr_max / cfg.lr)

    return {
        'loss': float(loss.item()),
        'policy_loss': float(policy_loss.item()),
        'value_loss': float(value_loss.item()),
        'entropy': float(ent.item()),
        'ratio': float(ratio_after.mean().item()),
        'ratio_min': float(ratio_after.min().item()),
        'ratio_max': float(ratio_after.max().item()),
        'V': float(v_before.item()),
        'V_old': float(v_old.item()),
        'R_raw_mean': float(R_raw_mean.item()),
        'R_scaled_mean': float(rewards.mean().item()),
        'A_mean': float(advantages.mean().item()),
        'A_norm_mean': float(advantages_n.mean().item()),
        'KL': float(kl.item()),
        'R_q': float(R_q),
        'R_cvar': float(R_cvar),
        'R_raw_min': float(rewards_raw.min().item()),
        'R_raw_max': float(rewards_raw.max().item()),
    }


# ---------------------------
# 单步 PPO 更新
# ---------------------------
# ---------------------------
# 单步 PPO 更新（在此函数内重新计算新策略）
# ---------------------------
def ppo_update(
    agent: nn.Module,
    v_old: torch.Tensor,         # 采样时 critic 的旧值 V_old(s)
    action: torch.Tensor,        # 这次 roll-out 的采样动作
    reward: Union[float, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    state: PPOState,
    cfg: PPOConfig,
    std_now: torch.Tensor,       # 当前使用的 std（标量或同形状广播）
):
    device = cfg.device
    assert state.old_mu is not None, "old_mu 尚未初始化"

    # 目标：reward 越大越好（单步优势）
    R_raw = reward if torch.is_tensor(reward) else torch.tensor(reward, dtype=torch.float32, device=device)
    R_raw = R_raw.to(device)
    # Keep a copy of original raw reward for logging
    R_raw_orig = R_raw.clone().detach()
    # Static reward scaling (divide by cfg.reward_scale for optimization, keep raw for logging)
    R_scaled = R_raw / cfg.reward_scale
    R = R_scaled if cfg.maximize_reward else -R_scaled

    v_old = v_old.view(-1)[0].to(device)
    advantage = (R - v_old).detach()

    # 优势标准化（滑窗）
    state.adv_buf.append(float(advantage.item()))
    adv_mean = torch.tensor(0.0, device=device)
    adv_std  = torch.tensor(1.0, device=device)
    if len(state.adv_buf) > 8:
        m = sum(state.adv_buf) / len(state.adv_buf)
        var = sum((x - m) ** 2 for x in state.adv_buf) / max(1, len(state.adv_buf) - 1)
        adv_mean = torch.tensor(m, device=device)
        adv_std  = torch.tensor(max(1e-6, var ** 0.5), device=device)
    if adv_std > 1e-6:
        advantage_n = ((advantage - adv_mean) / adv_std).clamp(-5.0, 5.0)
    else:
        advantage_n = advantage

    # —— 同上：确定性前向计算新策略 ——
    was_training = agent.training
    agent.train(False)
    mu_new, v_new = agent.forward()
    agent.train(was_training)
    if isinstance(mu_new, tuple):
        mu_new = mu_new[0]
    mu_new = mu_new.to(device)
    v_new = v_new.view(-1)[0].to(device)

    std_now = std_now.to(device)
    action = action.to(device)

    # 概率比 r_t = π_new(a)/π_old(a)
    logp_old, _ = normal_logp_entropy(action, state.old_mu.to(device), std_now)
    logp_new, ent = normal_logp_entropy(action, mu_new, std_now)
    ent = ent.clamp(min=0.0)
    ratio = torch.exp(torch.clamp(logp_new - logp_old, -10.0, 10.0))

    surr1 = ratio * advantage_n
    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantage_n
    policy_loss = -torch.min(surr1, surr2)
    value_loss  = F.mse_loss(v_new, R)
    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * ent

    # 反向 & 更新
    for g in optimizer.param_groups:
        g['lr'] = max(cfg.lr_min, min(cfg.lr_max, cfg.lr * state.lr_scale))
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    optimizer.step()

    # KL 自适应：使用更新后的策略
    with torch.no_grad():
        was_training = agent.training
        agent.train(False)
        mu_after, _ = agent.forward()
        agent.train(was_training)
        if isinstance(mu_after, tuple):
            mu_after = mu_after[0]
        kl = normal_kl(state.old_mu.to(device), mu_after.to(device), std_now if std_now.dim()==0 else std_now[0,0,0])
        if cfg.kl_adapt:
            if kl > 1.5 * cfg.target_kl:
                state.lr_scale = max(0.5 * state.lr_scale, cfg.lr_min / cfg.lr)
            elif kl < 0.5 * cfg.target_kl:
                state.lr_scale = min(1.5 * state.lr_scale, cfg.lr_max / cfg.lr)

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(ent.item()),
        "ratio": float(ratio.item()),
        "V": float(v_new.item()),
        "V_old": float(v_old.item()),
        "R_raw": float(R_raw_orig.item()),
        "R_scaled": float(R_scaled.item()),
        "A": float(advantage.item()),
        "A_norm": float(advantage_n.item()),
        "KL": float(kl.item()),
    }


# ---------------------------
# 训练主循环
# ---------------------------
def ppo_fit(
    agent: nn.Module,
    outputs_to_dict_fn: Callable[[torch.Tensor, Dict, Dict], Dict],
    simulate_fn: Callable[..., float],
    G,
    packet_size_per_subchunk,
    subchunks_node: Dict,
    edges: Dict,
    gpu_nodes: List[int],
    cfg: PPOConfig,
    verbose: bool = True,
):
    device = cfg.device
    agent.to(device)
    agent.train()

    optim = build_optimizer(agent, cfg)
    state = PPOState(adv_buf=deque(maxlen=cfg.adv_norm_window), elite_buf=[], worst_buf=[])

    for it in range(cfg.train_steps):
        state.step += 1

        # 退火后的 std（标量）
        if cfg.std_anneal_steps > 0:
            t = min(1.0, state.step / float(cfg.std_anneal_steps))
        else:
            t = 1.0
        action_std = cfg.init_action_std + (cfg.min_action_std - cfg.init_action_std) * t
        action_std = max(cfg.min_action_std, action_std)  # 保底
        # 让 std 和 mu 形状可广播（统一用标量张量）
        std_now = torch.as_tensor(action_std, device=device)

        # —— 以确定性前向获取 old_mu，避免 Dropout 造成 old/new 不一致 ——
        was_training = agent.training
        agent.train(False)
        mu, V = agent.forward()            # mu: [S,E], V: [1] 或标量
        agent.train(was_training)
        if isinstance(mu, tuple):
            mu = mu[0]
        mu = mu.to(device)
        # 固定 old policy：始终使用当前 mu 作为本轮采样时的 old_mu (raw mu, no topk/renorm, revert to old logic)
        state.old_mu = mu.detach().clone()

        # 探索：同一状态采样 K 个动作（结构化扰动 + 高斯混合 + 远跳样本）
        K = max(1, int(cfg.num_action_samples))
        a_list = []
        for k in range(K):
            u = torch.rand(())
            if u < cfg.restart_prob:
                a_k = random_restart_like(mu)
            elif u < cfg.restart_prob + cfg.strong_perturb_prob:
                a_k = strong_perturb_multi(mu, cfg, steps=cfg.strong_perturb_steps)
            else:
                if cfg.use_structured_perturb:
                    a_k = structured_perturb(mu, cfg)
                else:
                    a_k = (mu + std_now * torch.randn_like(mu)).clamp(0.0, 1.0)
            a_k = renorm_cols(a_k)
            a_list.append(a_k)
        # Optionally mix in elite actions (high-return actions from previous iters)
        if (len(state.elite_buf) > 0) and (torch.rand(()) < cfg.elite_mix_prob):
            n_elite = min(cfg.elite_pick_per_iter, len(state.elite_buf), K)
            idxs = torch.randperm(len(state.elite_buf))[:n_elite]
            for t, i in enumerate(idxs.tolist()):
                # Only use the action part (ignore reward)
                elite_action = state.elite_buf[i][1]
                a_list[-(t+1)] = elite_action.clone().to(mu.device)
        a_batch = torch.stack(a_list, dim=0)  # [K,S,E]

        # 环境 roll-out：逐个样本模拟
        rewards = []
        for k in range(K):
            mats_k = outputs_to_dict_fn(a_batch[k], subchunks_node, edges)
            R_k = simulate_fn(
                G=G,
                packet_size_per_subchunk=packet_size_per_subchunk,
                subchunk_priority_mats=mats_k,
                gpu_nodes=gpu_nodes,
                verbose=False,
            )
            rewards.append(float(R_k))
        best_R = max(rewards) if len(rewards) > 0 else float('nan')
        # Maintain elite buffer as global top-128 actions by reward (across all time)
        # Add all actions and rewards from this batch to the elite buffer
        for idx in range(len(rewards)):
            # Store as (reward, action)
            state.elite_buf.append((float(rewards[idx]), a_batch[idx].detach().cpu()))
        # Keep only the top-N by reward
        elite_buf_size = getattr(cfg, "elite_buf_size", 128)
        # Sort descending by reward, keep top-N
        state.elite_buf.sort(key=lambda x: x[0], reverse=True)
        if len(state.elite_buf) > elite_buf_size:
            state.elite_buf = state.elite_buf[:elite_buf_size]

        # === Maintain worst buffer: keep global lowest-N actions by reward (across all time) ===
        worst_buf_size = getattr(cfg, "worst_buf_size", 128)
        # Add all actions and rewards with reward < cfg.worst_threshold to the worst buffer
        for idx in range(len(rewards)):
            if rewards[idx] < cfg.worst_threshold:
                state.worst_buf.append((float(rewards[idx]), a_batch[idx].detach().cpu()))
        # Keep only the lowest-N by reward
        state.worst_buf.sort(key=lambda x: x[0])  # ascending (lowest first)
        if len(state.worst_buf) > worst_buf_size:
            state.worst_buf = state.worst_buf[:worst_buf_size]

        # 多 epoch：对同一批 K 个样本重复优化，不重采样，不改 old_mu
        last_log = None
        for _ in range(getattr(cfg, 'n_policy_epochs', 1)):
            last_log = ppo_update_multi(agent, V, a_batch, rewards, optim, state, cfg, std_now)
        log = last_log if last_log is not None else {}

        if verbose and (it % max(1, cfg.train_steps // 100) == 0):
            with torch.no_grad():
                # mu statistics from raw mu (actor output), not post-processed
                mu_min, mu_max, mu_mean = float(mu.min().item()), float(mu.max().item()), float(mu.mean().item())
                mu_std = float(mu.std().item())
                # For a_batch, flatten to get stats
                a_min, a_max, a_mean = float(a_batch.min().item()), float(a_batch.max().item()), float(a_batch.mean().item())
                nnz_mu = float((mu > 0).sum().item())
                nnz_a  = float((a_batch > 0).sum().item())
                # --- Compute and log current SE (mu) reward before printing ---
                # For reward computation, use renorm_cols on a copy of mu (no topk masking)
                a_mu = renorm_cols(mu.clone())
                mats_mu = outputs_to_dict_fn(a_mu, subchunks_node, edges)
                R_mu = simulate_fn(
                    G=G,
                    packet_size_per_subchunk=packet_size_per_subchunk,
                    subchunk_priority_mats=mats_mu,
                    gpu_nodes=gpu_nodes,
                    verbose=False,
                )
            total = cfg.train_steps
            print(
                f"[PPO] step {it+1}/{total} | R_mean={log['R_raw_mean']:.4f} R_best={best_R:.4f} R_mu={float(R_mu):.4f} R_q={log.get('R_q', 0.0):.2f} CVaR={log.get('R_cvar', 0.0):.2f} "
                f"V={log['V']:.4f} (V_old={log['V_old']:.4f}) "
                f"A_mean={log['A_mean']:.4f} A_n={log['A_norm_mean']:.4f} "
                f"loss={log['loss']:.4f} (pi={log['policy_loss']:.4f}, V={log['value_loss']:.4f}) "
                f"ratio={log['ratio']:.3f} [{log['ratio_min']:.3f},{log['ratio_max']:.3f}] KL={log['KL']:.5f} H={log['entropy']:.3f} "
                f"| mu[min={mu_min:.3f}, max={mu_max:.3f}, mean={mu_mean:.3f}, std={mu_std:.3f}] "
                f"| a[min={a_min:.3f}, max={a_max:.3f}, mean={a_mean:.3f}] "
                f"| nnz_mu={nnz_mu:.1f}, nnz_a={nnz_a:.1f} | std={float(action_std):.4f} lr_scale={state.lr_scale:.3f}"
                f" | elite_buf={len(state.elite_buf)}"
            )

    return agent