from decimal import Decimal
from random import shuffle

import config
from GNN_DRL.utils.util import load_topology
from utils.es_trainers import ESTrainer
from utils.Agent import AllGatherAgent,RewardNet
from utils.tools import simulate_allgather_pipeline_bfs, build_subchunk_weights_from_policy
import torch.nn.functional as F
import torch

# ==== Module-scope helpers for dataset building ====
# We need a global context so that build_dataset() can call the simulator
# without capturing locals from main().
_SIM_CTX = {"G": None, "edges": None, "subchunks_node": None, "GPU_list": None}

def set_sim_context(G, edges, subchunks_node, GPU_list):
    """Set global simulator context used by sim_reward_from_se()."""
    _SIM_CTX["G"] = G
    _SIM_CTX["edges"] = edges
    _SIM_CTX["subchunks_node"] = subchunks_node
    _SIM_CTX["GPU_list"] = GPU_list


def renorm_rows(se: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Clamp to non-negative and L1-normalize each row to sum to 1."""
    se = se.clamp(min=0.0)
    row_sum = se.sum(dim=1, keepdim=True)
    return se / (row_sum + eps)


def sim_reward_from_se(se_tensor: torch.Tensor) -> float:
    """Convert SE matrix to simulator input and return the scalar reward.
    Requires set_sim_context(...) to be called beforehand in main().
    """
    if _SIM_CTX["G"] is None:
        raise RuntimeError("Simulator context not set. Call set_sim_context(...) before using sim_reward_from_se().")
    mats = outputs_to_dict(se_tensor, _SIM_CTX["subchunks_node"], _SIM_CTX["edges"])
    R = simulate_allgather_pipeline_bfs(
        G=_SIM_CTX["G"],
        packet_size_per_subchunk=(config.packet_size / config.num_chunk),
        subchunk_priority_mats=mats,
        gpu_nodes=_SIM_CTX["GPU_list"],
        verbose=False,
    )
    return R


def outputs_to_dict(outputs, subchunks_node, edges, thresh=0.0):
    """
    将 agent.forward() 的输出 [S, E] 转换为嵌套字典:
      { sub_id: { (u,v): weight, ... }, ... }
    - outputs: torch.Tensor [S, E]
    - subchunks_node: dict {scid: {...}} (顺序必须与预处理一致)
    - edges: dict {eid: {...}} (顺序必须与预处理一致)
    - thresh: 小于该阈值的权重会被忽略
    """
    mats = {}
    S, E = outputs.shape

    # 边 (u,v) 顺序必须和预处理一致
    edge_uv = [(int(edges[eid]["u"]), int(edges[eid]["v"])) for eid in edges]

    # subchunk 顺序必须和预处理一致
    sc_ids = list(subchunks_node.keys())

    for sid_idx, sid in enumerate(sc_ids):
        mats[sid] = {}
        for e_idx, (u, v) in enumerate(edge_uv):
            w = float(outputs[sid_idx, e_idx].item())
            if w > thresh:
                mats[sid][(u, v)] = w
    return mats

def main(policy):
    datacenter = load_topology(packet_size=config.packet_size, num_chunk=config.num_chunk, chassis=config.chassis, name=config.topology_name)
    G = datacenter.topology

    GPU_list = []
    for node in G.nodes:
        if G.nodes[node]['type'] == 'GPU' or G.nodes[node]['type'] == 'gpu':
            GPU_list.append(node)

    subchunk_weight_dict = build_subchunk_weights_from_policy(policy)
    # print(subchunk_weight_dict)
    reward = simulate_allgather_pipeline_bfs(G=G, packet_size_per_subchunk=config.packet_size / config.num_chunk,
                                             subchunk_priority_mats=subchunk_weight_dict, gpu_nodes=GPU_list,
                                             verbose=False)
    print('reward',reward)

    # adjacency as dict of neighbor lists
    adjacency = {n: list(G.neighbors(n)) for n in G.nodes()}
    node_degrees = {n: len(adjacency[n]) for n in G.nodes()}

    # Build nodes as a dict: {node_id: {"degree": ..., "mean_neighbor_degree": ..., "node_type": ...}}
    nodes = {}
    for n in G.nodes():
        neighbors = adjacency[n]
        mean_neighbor_degree = sum(node_degrees[nb] for nb in neighbors) / len(neighbors) if neighbors else 0
        nodes[n] = {
            "degree": float(node_degrees[n]),
            "mean_neighbor_degree": float(mean_neighbor_degree),
            "node_type": G.nodes[n].get('type', None)
        }

    # Build edges as a dict: {(u, v): {"u": u, "v": v, "tx_lat": ..., "prop_lat": ...}}
    edges = {}
    for u, v, data in G.edges(data=True):
        tx_lat = data.get('transmission_latency', None)
        prop_lat = data.get('propagation_latency', None)
        if isinstance(tx_lat, Decimal):
            tx_lat = float(tx_lat)
        if isinstance(prop_lat, Decimal):
            prop_lat = float(prop_lat)
        # Store the forward edge
        edges[(u, v)] = {
            "u": u,
            "v": v,
            "tx_lat": tx_lat,
            "prop_lat": prop_lat
        }
        # Store the reverse edge
        edges[(v, u)] = {
            "u": v,
            "v": u,
            "tx_lat": tx_lat,
            "prop_lat": prop_lat
        }
    # Build subchunk-to-node mapping with source_node and pos_in_node
    subchunks_node = {}
    num_nodes = len(nodes)
    DC_node = {}
    for node in G.nodes:
        DC = G.nodes[node]['DC']
        if DC not in DC_node.keys():
            DC_node[DC] = [node]
        else:
            DC_node[DC].append(node)
        # Collect the list of subchunks (buffer not None) for this node
        subchunk_ids = [value['buffer'] for id, value in G.nodes[node]['memory'].items() if value['buffer'] is not None]
        total_subchunks_at_node = len(subchunk_ids)
        for idx, subchunk_id in enumerate(subchunk_ids):
            # Compute pos_in_node as float division
            pos_in_node = idx / total_subchunks_at_node if total_subchunks_at_node > 0 else 0.0
            subchunks_node[subchunk_id] = {
                "source_node": node,
                "pos_in_node": pos_in_node
            }

    # Make simulator context available to module-scope helpers
    set_sim_context(G, edges, subchunks_node, GPU_list)
    # ===== 构建 agent =====
    agent = AllGatherAgent(nodes, edges, subchunks_node, DC_node)



    from GNN_DRL.utils.ppo_trainer import PPOConfig, ppo_fit

    cfg = PPOConfig()

    # 假设你已有: agent, simulate_fn, outputs_to_dict, G, packet_size_per_subchunk, subchunks_node, edges, gpu_nodes
    ppo_fit(
        agent=agent,
        outputs_to_dict_fn=outputs_to_dict,
        simulate_fn=simulate_allgather_pipeline_bfs,
        G=G,
        packet_size_per_subchunk=(config.packet_size / config.num_chunk),
        subchunks_node=subchunks_node,
        edges=edges,
        gpu_nodes=GPU_list,
        cfg=cfg,
        verbose=True,
    )

    # 训练完成后，拿最终策略评估一次

    # ===== 可微替身：RewardNet =====
    # try:
    #     from utils.reward_net import RewardNet  # 如果你把类放到了 utils/reward_net.py
    # except ImportError:
    #     # 如果直接把 RewardNet 类定义贴在了本文件上方，这里就直接用
    #     R

    # reward_net = RewardNet(hidden_dim=128, num_heads=4, num_layers=2).to(agent.sub_feat.device)
    # # opt_rn = torch.optim.AdamW(reward_net.parameters(), lr=1e-3, weight_decay=1e-4)
    # # opt_agent = torch.optim.AdamW(agent.parameters(), lr=3e-4, weight_decay=0.0)
    #
    # # scale = 300.0  # 把 [-几百,几百] 缩放到 ~[-1,1]
    # # lam_sparsity = 1e-4  # 轻微稀疏/稳定正则，防止SE爆满
    # # max_grad_norm = 1.0
    # # lam_anchor = 1e-3  # trust-region style anchor on SE changes
    #
    #
    #
    # # ===== 预训练 RewardNet：用大量样本单训替身 =====
    # scale = 300.0  # 把 [-几百,几百] 缩放到 ~[-1,1]
    # lam_anchor = 1e-4
    # lam_sparsity = 0.0
    # max_grad_norm = 1.0
    #
    # def sim_reward_from_se(se_tensor):
    #     mats = outputs_to_dict(se_tensor, subchunks_node, edges)
    #     return simulate_allgather_pipeline_bfs(
    #         G=G,
    #         packet_size_per_subchunk=(config.packet_size / config.num_chunk),
    #         subchunk_priority_mats=mats,
    #         gpu_nodes=GPU_list,
    #         verbose=False
    #     )
    #
    # @torch.no_grad()
    # def renorm_rows(se: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    #     # 保持非负，并按行归一化到和为1（避免退化）
    #     se = se.clamp(min=0.0)
    #     row_sum = se.sum(dim=1, keepdim=True)
    #     se = se / (row_sum + eps)
    #     return se
    #
    # @torch.no_grad()
    # def perturb_se(se: torch.Tensor, noise_std: float = 0.05, sprinkle_prob: float = 0.05) -> torch.Tensor:
    #     # 轻量扰动：加噪并随机在若干零位置撒少量权重，然后按行归一化
    #     z = se + noise_std * torch.randn_like(se)
    #     z = z.clamp(min=0.0)
    #     if sprinkle_prob > 0.0:
    #         mask_zero = (se < 1e-6).float()
    #         sprinkle = (torch.rand_like(se) < sprinkle_prob).float() * mask_zero
    #         z = z + 0.01 * sprinkle
    #     return renorm_rows(z)
    #
    # def build_dataset(agent, n_samples: int = 256):
    #     """
    #     采样数据集：50% 用 agent 当前策略生成（含少量扰动增强多样性），
    #     50% 用随机 SE（[0,1] 均匀采样后做按行归一）。
    #     返回 [(se_tensor, reward_float), ...]
    #     """
    #     dataset = []
    #     agent.eval()
    #     device = agent.sub_feat.device
    #     S, E = agent.S, agent.E
    #
    #     for i in range(n_samples):
    #         if (i % 2) == 0:
    #             # ---- 来自 agent 的样本（带轻量扰动以增加多样性）----
    #             with torch.no_grad():
    #                 se_pred, _ = agent.forward()  # [S, E]
    #             # 50% 原样 + 50% 轻扰动
    #             if (i // 2) % 2 == 0:
    #                 se_used = renorm_rows(se_pred)
    #             else:
    #                 se_used = perturb_se(se_pred)  # 你上面已实现的轻量扰动 + 归一
    #             src = "agent"
    #         else:
    #             # ---- 随机样本 ----
    #             se_used = torch.rand((S, E), device=device)
    #             se_used = renorm_rows(se_used)
    #             src = "random"
    #
    #         R_true = sim_reward_from_se(se_used)
    #         dataset.append((se_used.detach(), float(R_true)))
    #
    #         if (i + 1) % 32 == 0:
    #             print(f"[DATA] {i + 1:04d}/{n_samples} | src={src:6s} | R_true={R_true:.3f}")
    #     shuffle(dataset)
    #
    #     return dataset
    #
    # # 1) 构建数据集
    # n_samples = 2048
    # dataset = build_dataset(agent, n_samples=n_samples)
    #
    # # 2) 用数据集单独训练 RewardNet（监督拟合模拟器）
    # reward_net.train()
    # opt_rn = torch.optim.AdamW(reward_net.parameters(), lr=1e-3, weight_decay=1e-4)
    # rn_epochs = 64
    # for ep in range(1, rn_epochs + 1):
    #     total = 0.0
    #     for se_used, R_true in dataset:
    #         y = torch.tensor([R_true / scale], device=agent.sub_feat.device)
    #         R_hat = reward_net(agent, se_used) / scale
    #         loss_rn = F.smooth_l1_loss(R_hat.view_as(y), y)
    #         opt_rn.zero_grad()
    #         loss_rn.backward()
    #         torch.nn.utils.clip_grad_norm_(reward_net.parameters(), max_grad_norm)
    #         opt_rn.step()
    #         total += loss_rn.item()
    #     print(f"[RN] epoch {ep:02d}/{rn_epochs} | avg_loss={total/len(dataset):.6f}")
    #
    # # 3) 冻结 RewardNet，只训练 agent（监督把 \hat R 推到一个很大目标）
    # reward_net.eval()
    # for p in reward_net.parameters():
    #     p.requires_grad = False
    #
    # # 只训练 scoring head（最小改动；如需全量训练，可去掉这段冻结）
    # for p in agent.parameters():
    #     p.requires_grad = False
    # for p in agent.scoring.parameters():
    #     p.requires_grad = True
    #
    # opt_agent = torch.optim.AdamW(filter(lambda p: p.requires_grad, agent.parameters()), lr=3e-4, weight_decay=0.0)
    #
    # target_reward = 1000.0
    # target_scaled = torch.tensor([target_reward / scale], device=agent.sub_feat.device)
    #
    # se_prev = None
    # steps_agent = 1000
    # for t in range(1, steps_agent + 1):
    #     se_pred, _ = agent.forward()  # [S,E]
    #     R_hat = reward_net(agent, se_pred) / scale
    #     if se_prev is None:
    #         se_prev = se_pred.detach()
    #     anchor = (se_pred - se_prev).pow(2).mean()
    #     loss_agent = F.mse_loss(R_hat.view_as(target_scaled), target_scaled) + lam_anchor * anchor + lam_sparsity * se_pred.mean()
    #
    #     opt_agent.zero_grad()
    #     loss_agent.backward()
    #     torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    #     # 打印一下 scoring head 的梯度范数，确认在学习
    #     with torch.no_grad():
    #         grad_sq = 0.0
    #         for p in agent.scoring.parameters():
    #             if p.grad is not None:
    #                 grad_sq += (p.grad ** 2).sum().item()
    #         grad_norm = (grad_sq ** 0.5)
    #     opt_agent.step()
    #
    #     se_prev = se_pred.detach()
    #
    #     if t % 50 == 0:
    #         print(f"[AG] step {t:04d}/{steps_agent} | R_hat*={R_hat.item()*scale:8.3f} | loss={loss_agent.item():.6f} | grad_scoring={grad_norm:.3e} | mu[min={se_pred.min().item():.3f}, max={se_pred.max().item():.3f}, mean={se_pred.mean().item():.3f}]")

    # ===== 训练完成后，用最终策略评估一次 =====
    agent.eval()
    final_outputs, value = agent.forward()  # [S, E]
    final_mats = outputs_to_dict(final_outputs, subchunks_node, edges)
    final_reward = simulate_allgather_pipeline_bfs(
        G=G,
        packet_size_per_subchunk=(config.packet_size / config.num_chunk),
        subchunk_priority_mats=final_mats,
        gpu_nodes=GPU_list,
        verbose=True
    )
    print("final_mats: ", final_mats)
    print("Final reward:", final_reward)
    print(final_mats)















if __name__ == '__main__':
    policy = [[0, 4, 0, Decimal('0.0')], [0, 4, 5, Decimal('0.0')], [0, 4, 6, Decimal('0.0')], [0, 4, 8, Decimal('0.0')], [1, 5, 1, Decimal('0.0')], [1, 5, 4, Decimal('0.0')], [1, 5, 6, Decimal('0.0')], [2, 6, 4, Decimal('0.0')], [2, 6, 5, Decimal('0.0')], [2, 6, 7, Decimal('0.0')], [3, 7, 0, Decimal('0.0')], [3, 7, 6, Decimal('0.0')], [3, 7, 8, Decimal('0.0')], [4, 8, 0, Decimal('0.0')], [4, 8, 4, Decimal('0.0')], [4, 8, 7, Decimal('0.0')], [4, 8, 9, Decimal('0.0')], [5, 9, 1, Decimal('0.0')], [5, 9, 8, Decimal('0.0')], [6, 10, 2, Decimal('0.0')], [6, 10, 11, Decimal('0.0')], [7, 11, 3, Decimal('0.0')], [7, 11, 10, Decimal('0.0')], [7, 11, 12, Decimal('0.0')], [7, 11, 15, Decimal('0.0')], [8, 12, 2, Decimal('0.0')], [8, 12, 11, Decimal('0.0')], [8, 12, 13, Decimal('0.0')], [9, 13, 2, Decimal('0.0')], [9, 13, 12, Decimal('0.0')], [9, 13, 14, Decimal('0.0')], [9, 13, 15, Decimal('0.0')], [10, 14, 13, Decimal('0.0')], [10, 14, 15, Decimal('0.0')], [11, 15, 3, Decimal('0.0')], [11, 15, 11, Decimal('0.0')], [11, 15, 13, Decimal('0.0')], [11, 15, 14, Decimal('0.0')], [4, 0, 2, Decimal('7E-7')], [3, 0, 3, Decimal('7E-7')], [5, 1, 2, Decimal('7E-7')], [1, 1, 3, Decimal('7E-7')], [9, 2, 0, Decimal('7E-7')], [8, 2, 1, Decimal('7E-7')], [11, 3, 0, Decimal('7E-7')], [7, 3, 1, Decimal('7E-7')], [2, 4, 1, Decimal('0.0000397625')], [4, 4, 5, Decimal('0.0000397625')], [4, 4, 6, Decimal('0.0000397625')], [1, 4, 8, Decimal('0.0000397625')], [3, 6, 4, Decimal('0.0000397625')], [3, 6, 5, Decimal('0.0000397625')], [1, 6, 7, Decimal('0.0000397625')], [2, 7, 8, Decimal('0.0000397625')], [5, 8, 4, Decimal('0.0000397625')], [5, 8, 7, Decimal('0.0000397625')], [0, 8, 9, Decimal('0.0000397625')], [8, 11, 10, Decimal('0.0000397625')], [6, 11, 12, Decimal('0.0000397625')], [6, 11, 15, Decimal('0.0000397625')], [9, 12, 11, Decimal('0.0000397625')], [7, 12, 13, Decimal('0.0000397625')], [10, 13, 12, Decimal('0.0000397625')], [8, 13, 14, Decimal('0.0000397625')], [8, 13, 15, Decimal('0.0000397625')], [10, 15, 11, Decimal('0.0000397625')], [7, 15, 14, Decimal('0.0000397625')], [10, 13, 2, Decimal('0.000078125')], [0, 0, 2, Decimal('0.000078825')], [2, 1, 2, Decimal('0.000078825')], [6, 2, 0, Decimal('0.000078825')], [0, 6, 7, Decimal('0.000078825')], [3, 8, 9, Decimal('0.000078825')], [11, 11, 10, Decimal('0.000078825')], [11, 11, 12, Decimal('0.000078825')], [5, 4, 5, Decimal('0.0000795250')], [5, 4, 6, Decimal('0.0000795250')], [6, 12, 13, Decimal('0.0000795250')], [6, 15, 14, Decimal('0.0000795250')], [1, 8, 9, Decimal('0.0001178875')], [9, 11, 10, Decimal('0.0001178875')], [10, 2, 0, Decimal('0.000156950')], [2, 8, 9, Decimal('0.000156950')], [10, 11, 10, Decimal('0.000156950')], [9, 0, 4, Decimal('0.0007507')], [11, 0, 7, Decimal('0.0007507')], [11, 0, 8, Decimal('0.0007507')], [8, 1, 4, Decimal('0.0007507')], [7, 1, 5, Decimal('0.0007507')], [7, 1, 9, Decimal('0.0007507')], [4, 2, 10, Decimal('0.0007507')], [5, 2, 12, Decimal('0.0007507')], [5, 2, 13, Decimal('0.0007507')], [5, 2, 15, Decimal('0.0007507')], [3, 3, 11, Decimal('0.0007507')], [3, 3, 15, Decimal('0.0007507')], [11, 0, 4, Decimal('0.000828825')], [6, 0, 7, Decimal('0.000828825')], [6, 0, 8, Decimal('0.000828825')], [7, 1, 4, Decimal('0.000828825')], [8, 1, 5, Decimal('0.000828825')], [8, 1, 9, Decimal('0.000828825')], [5, 2, 10, Decimal('0.000828825')], [0, 2, 12, Decimal('0.000828825')], [0, 2, 13, Decimal('0.000828825')], [0, 2, 15, Decimal('0.000828825')], [1, 3, 11, Decimal('0.000828825')], [1, 3, 15, Decimal('0.000828825')], [9, 4, 5, Decimal('0.000829525')], [8, 4, 6, Decimal('0.000829525')], [8, 4, 8, Decimal('0.000829525')], [7, 5, 6, Decimal('0.000829525')], [11, 7, 6, Decimal('0.000829525')], [11, 8, 9, Decimal('0.000829525')], [7, 9, 8, Decimal('0.000829525')], [4, 10, 11, Decimal('0.000829525')], [3, 11, 10, Decimal('0.000829525')], [3, 11, 12, Decimal('0.000829525')], [5, 12, 11, Decimal('0.000829525')], [5, 13, 14, Decimal('0.000829525')], [3, 15, 13, Decimal('0.000829525')], [3, 15, 14, Decimal('0.000829525')], [9, 4, 6, Decimal('0.0008685875')], [9, 4, 8, Decimal('0.0008685875')], [11, 6, 5, Decimal('0.0008692875')], [7, 6, 7, Decimal('0.0008692875')], [8, 8, 7, Decimal('0.0008692875')], [4, 11, 12, Decimal('0.0008692875')], [4, 11, 15, Decimal('0.0008692875')], [6, 0, 4, Decimal('0.000906950')], [9, 0, 7, Decimal('0.000906950')], [10, 0, 8, Decimal('0.000906950')], [0, 2, 10, Decimal('0.000906950')], [2, 2, 12, Decimal('0.000906950')], [4, 2, 13, Decimal('0.000906950')], [2, 2, 15, Decimal('0.000906950')], [6, 7, 6, Decimal('0.000907650')], [6, 8, 9, Decimal('0.000907650')], [1, 11, 10, Decimal('0.000907650')], [0, 12, 11, Decimal('0.000907650')], [0, 13, 14, Decimal('0.000907650')], [1, 15, 13, Decimal('0.000907650')], [1, 15, 14, Decimal('0.000907650')], [1, 11, 12, Decimal('0.0009083500')], [9, 8, 9, Decimal('0.0009467125')], [4, 15, 14, Decimal('0.0009467125')], [6, 6, 5, Decimal('0.0009474125')], [10, 0, 4, Decimal('0.000985075')], [10, 0, 7, Decimal('0.000985075')], [2, 2, 10, Decimal('0.000985075')], [2, 2, 13, Decimal('0.000985075')], [10, 8, 9, Decimal('0.000985775')], [2, 12, 11, Decimal('0.000985775')], [2, 15, 14, Decimal('0.000985775')], [10, 4, 5, Decimal('0.001063900')], [10, 4, 6, Decimal('0.001063900')]]
    config.chassis = 2
    config.num_chunk = 1
    config.packet_size = Decimal(1/1024)
    config.connectivity = 0.5
    config.collective = 'ALLGATHER'
    config.topology_name = 'NVD2'

    main(policy=policy)