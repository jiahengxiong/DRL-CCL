# utils/es_trainer.py
import math
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class ESTrainer:
    def __init__(self, agent, sim_fn, sim_args: dict, pop_size=16, sigma=0.05, lr=1e-2, seed=42, device=None):
        """
        agent: 已构造好的 AllGatherAgent（含预处理好的特征）
        sim_fn: 一个函数 rollout(agent) -> reward
        sim_args: sim_fn 需要的额外参数字典（比如 G、GPU_list、config 等）
        pop_size: 每次迭代的扰动对数 N（总评估次数 2N）
        sigma: 扰动标准差
        lr: Adam 学习率
        """
        self.agent = agent
        self.sim_fn = sim_fn
        self.sim_args = sim_args
        self.pop_size = pop_size
        self.sigma = sigma
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.device = device or next(agent.parameters()).device

        # 初始化 θ & 优化器
        self.theta = parameters_to_vector(self.agent.parameters()).detach().clone()
        self.theta.requires_grad = False  # 我们手动设梯度
        self.optimizer = torch.optim.Adam([torch.nn.Parameter(self.theta)], lr=lr)

    @torch.no_grad()
    def _set_params(self, flat_params: torch.Tensor):
        """
        将展平参数写回模型
        flat_params 必须搬回到 self.device（比如 MPS）
        """
        flat_params = flat_params.to(self.device)  # 确保和模型同设备
        idx = 0
        for p in self.agent.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[idx: idx + numel].view_as(p))
            idx += numel

    def _rollout(self):
        # 让 agent 产出 [S, E]，转成 mats，进模拟器拿 reward
        outputs = self.agent.forward()                     # [S, E]
        mats = self.sim_args["outputs_to_dict"](outputs, self.sim_args["subchunks_node"], self.sim_args["edges"])
        reward = self.sim_args["simulate"](
            G=self.sim_args["G"],
            packet_size_per_subchunk=self.sim_args["packet_size_per_subchunk"],
            subchunk_priority_mats=mats,
            gpu_nodes=self.sim_args["GPU_list"],
            verbose=False
        )
        # 期望“越大越好”，如果你的 simulate 返回的是负 makespan，这里就直接用；否则可做 reward = -makespan
        return float(reward)

    # --- 放在 ESTrainer 类里 ---

    def get_flat_params(self) -> torch.Tensor:
        # 用 vectorize API 拉平参数；不指定 device，保持参数原设备
        return torch.nn.utils.parameters_to_vector(self.agent.parameters())

    def _set_params(self, flat_params: torch.Tensor):
        # 确保写回到 agent 参数时搬到 self.device（MPS），避免参数落在 CPU
        flat_params = flat_params.to(self.device)
        torch.nn.utils.vector_to_parameters(flat_params, self.agent.parameters())

    def step(self):
        """
        单步迭代（镜像采样）：只用每个方向的差分 r_plus - r_minus 做梯度，
        不用 batch 的均值做优化信号；均值仅用于日志。
        使用 top-k reward 样本估计梯度。
        """
        # 1) 取当前参数（放 CPU，避免 MPS 随机数/flatten 兼容坑）
        base_theta = self.get_flat_params().detach().cpu()

        # 2) 生成 N 个扰动（CPU 采样；MPS 也没问题）
        eps_list = []
        gen = getattr(self, "rng", None)
        for _ in range(self.pop_size):
            try:
                if gen is not None:
                    # 强制在 CPU 上采样，避免 MPS generator 兼容问题
                    eps_cpu = torch.randn(base_theta.numel(), dtype=base_theta.dtype, device="cpu", generator=gen)
                else:
                    eps_cpu = torch.randn(base_theta.numel(), dtype=base_theta.dtype, device="cpu")
            except RuntimeError:
                # fallback: 不带 generator，纯 CPU 随机
                eps_cpu = torch.randn(base_theta.numel(), dtype=base_theta.dtype, device="cpu")
            eps_list.append(eps_cpu.view_as(base_theta))  # 全在 CPU

        # 3) 评估镜像样本：θ±σ ε（rollout 内部会把参数搬到模型设备上）
        rewards_plus, rewards_minus = [], []
        for i in range(self.pop_size):
            theta_plus = base_theta + self.sigma * eps_list[i]
            self._set_params(theta_plus)
            r_plus = self._rollout()
            rewards_plus.append(r_plus)

            theta_minus = base_theta - self.sigma * eps_list[i]
            self._set_params(theta_minus)
            r_minus = self._rollout()
            rewards_minus.append(r_minus)

        # 恢复 θ
        self._set_params(base_theta)

        # 4) 计算差分优势（top-k reward），只作为梯度系数；不做 batch mean 回传
        r_plus_t  = torch.tensor(rewards_plus, dtype=torch.float32)  # CPU
        r_minus_t = torch.tensor(rewards_minus, dtype=torch.float32)

        # 合并所有 reward，只选最优的 top-1
        all_rewards = torch.cat([r_plus_t, r_minus_t])
        topk_val, topk_idx = torch.topk(all_rewards, 1)  # 只取最优样本
        k = 1

        # 5) 估计梯度：g = (1/(k*σ)) Σ_{topk} reward_i * eps_i
        g = torch.zeros_like(base_theta)  # CPU
        for idx in topk_idx.tolist():
            if idx < self.pop_size:
                # plus
                g.add_(r_plus_t[idx] * eps_list[idx])
            else:
                # minus
                j = idx - self.pop_size
                g.add_(r_minus_t[j] * (-eps_list[j]))
        g.div_(k * self.sigma)

        # 6) 用 Adam 做“奖励上升”：opt 是最小化，所以写入 -g
        self.optimizer.zero_grad(set_to_none=True)
        # 把 grad 写到 self.theta 上（保持设备一致）
        grad_dev = g.to(self.theta.device)
        if self.theta.grad is None:
            self.theta.grad = -grad_dev
        else:
            self.theta.grad.copy_(-grad_dev)
        self.optimizer.step()

        # 7) 同步 θ -> agent
        with torch.no_grad():
            self._set_params(self.theta)

        # 8) 日志（只打印，不参与学习信号）
        mean_r = 0.5 * (r_plus_t.mean().item() + r_minus_t.mean().item())
        best_r = max(r_plus_t.max().item(), r_minus_t.max().item())
        grad_norm = g.norm().item()
        print(f"[ES] iter={getattr(self, 'iter_idx', 0):04d}  mean_r={mean_r:.4f}  best_r={best_r:.4f}  grad_norm={grad_norm:.4e}")

        return mean_r, grad_norm

    def train(self, iters=200, log_every=10):
        history = []
        for t in range(1, iters + 1):
            mean_r, grad_norm = self.step()
            # Print every iteration
            print(f"[ES][step] iter={t:04d}  mean_reward={mean_r:.6f}  grad_norm={grad_norm:.4f}")
            history.append((t, mean_r, grad_norm))
            if (t % log_every) == 0:
                print(f"[ES] iter={t:04d}  mean_reward={mean_r:.6f}  grad_norm={grad_norm:.4f}")
        return history