# Allgather_AgentinBR/utils/critics.py
from typing import Optional
import torch
import torch.nn as nn

# 复用 Agent.py 里的组件，避免重复实现
from Allgather_AgentinBR.utils.Agent import AgentConfig, make_mlp, PositionalEncoding


class ValueCritic(nn.Module):
    """
    PPO Critic / Value Function.

    架构：
      - 每链路独立的 link encoder（不共享）
      - 共享 GRUCell 做每链路的时间记忆（各链路槽位独立更新）
      - TransformerEncoder 跨链路建模
      - 全局聚合（mean-pool） -> MLP -> 标量价值 V(s)

    输入（与 LinkPolicyAgent 保持一致的特征顺序）：
      link_states: [L, D] 或 [B, L, D]，D=6：
        [prop_s, tx_s, queue_len, hist_cnt, tail_finish_minus_now_s, current_time]

    输出：
      - 若输入 [L, D] -> 标量（0-D Tensor）
      - 若输入 [B, L, D] -> [B]
    """

    def __init__(self, num_links: int, cfg: AgentConfig = None):
        super().__init__()
        if cfg is None:
            # 与 Actor 完全一致：in_dim=6
            cfg = AgentConfig(in_dim=6)
        else:
            # 强制保持与 Actor 一致
            cfg.in_dim = 6
        self.cfg = cfg
        self.num_links = num_links

        # 每链路独立 encoder（不共享）
        self.link_encoders = nn.ModuleList([
            make_mlp(cfg.in_dim, cfg.link_layers, cfg.transformer_dim, cfg.activation, cfg.layernorm)
            for _ in range(num_links)
        ])
        self.num_links = len(self.link_encoders)

        # 每链路时间记忆（共享权重，按槽位独立更新）
        self.rnn_cell = None  # RNN removed

        # 跨链路 Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.transformer_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_dim * 4,
            dropout=cfg.transformer_dropout,
            batch_first=True,
            activation=cfg.activation,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=cfg.transformer_layers)
        self.posenc = PositionalEncoding(cfg.transformer_dim, cfg.transformer_dropout) if cfg.use_positional_encoding else nn.Identity()

        # 价值头：聚合后的 [B, T] -> 标量
        self.value_head = make_mlp(cfg.transformer_dim, cfg.head_layers, 1, cfg.activation, cfg.layernorm)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 隐藏态缓冲区: [B, num_links, T]
        self._h: Optional[torch.Tensor] = None

    # ---------- hidden state helpers ----------
    def reset_state(self) -> None:
        """清空内部 RNN 记忆（切换 episode 时调用）"""
        self._h = None

    def get_state(self) -> Optional[torch.Tensor]:
        """获取当前记忆的拷贝（用于异步 rollouts 的保存/恢复）"""
        return None if self._h is None else self._h.clone()

    def set_state(self, h: Optional[torch.Tensor]) -> None:
        """从外部设置记忆（会做拷贝）"""
        self._h = None if h is None else h.clone()

    def _ensure_state(self, B: int, device: torch.device, dtype: torch.dtype) -> None:
        """
        确保 self._h 存在且形状为 [B, num_links, T]。
        注意：运行时的 L 可以小于等于 num_links。
        """
        T = self.cfg.transformer_dim
        n_links = self.num_links
        h = self._h
        if (h is None) or (h.shape[0] != B) or (h.shape[1] != n_links) or (h.shape[2] != T) \
           or (h.device != device) or (h.dtype != dtype):
            self._h = torch.zeros(B, n_links, T, device=device, dtype=dtype)

    # ---------- preprocessing（与 Actor 完全一致） ----------
    def _normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]，特征顺序：
           [prop_s, tx_s, queue_len, hist_cnt, tail_finish_minus_now_s, current_time]
        策略：
          - t_ref = median(prop + tx)（每个样本按其 L 上取中位数）
          - prop/tx/tail 用 t_ref 缩放
          - hist_cnt、tail、current_time 做 log1p
          - 仅对前5列做轻度截断 + z-score，current_time 不做 z-score
        """
        if x.size(-1) < 6:
            raise ValueError(f"Expected 6 features per link, got D={x.size(-1)}")
        x = x.clone()
        eps = 1e-9

        # 参考时间：t_ref = median(prop + tx)
        prop = torch.clamp(x[:, :, 0], min=0.0)
        tx   = torch.clamp(x[:, :, 1], min=0.0)
        sum_pt = prop + tx
        t_ref = sum_pt.median(dim=1, keepdim=True).values
        t_ref = torch.clamp(t_ref, min=1e-6)

        # 按 t_ref 缩放时间列
        x[:, :, 0] = prop / t_ref       # prop
        x[:, :, 1] = tx   / t_ref       # tx
        x[:, :, 4] = torch.clamp(x[:, :, 4], min=0.0) / t_ref  # tail

        # 长尾压缩
        x[:, :, 3] = torch.log1p(torch.clamp(x[:, :, 3], min=0.0))  # hist_cnt
        x[:, :, 4] = torch.log1p(x[:, :, 4])                        # tail
        x[:, :, 5] = torch.log1p(torch.clamp(x[:, :, 5], min=0.0))  # current_time

        # 仅对前5列做截断与 z-score（避免抹平时间语义）
        head5 = x[:, :, :5]
        mean_approx = head5.mean(dim=1, keepdim=True)
        std_approx  = head5.std(dim=1, keepdim=True, unbiased=False) + eps
        upper = mean_approx + 3.0 * std_approx
        lower = mean_approx - 3.0 * std_approx
        head5 = torch.max(torch.min(head5, upper), lower)

        mean = head5.mean(dim=1, keepdim=True)
        std  = head5.std(dim=1, keepdim=True, unbiased=False) + eps
        head5 = (head5 - mean) / std

        x[:, :, :5] = head5
        return x

    # ---------- forward ----------
    def forward(self, link_states) -> torch.Tensor:
        """
        返回标量价值：
          - 输入 [L, D] -> 标量（0-D Tensor）
          - 输入 [B, L, D] -> [B]
        """
        # 统一输入
        if isinstance(link_states, (list, tuple)):
            link_states = torch.as_tensor(link_states, dtype=torch.float32)

        squeeze_out = False
        if link_states.dim() == 2:  # [L, D]
            link_states = link_states.unsqueeze(0)  # -> [1, L, D]
            squeeze_out = True
        elif link_states.dim() != 3:
            raise ValueError(f"link_states must be [L, D] or [B, L, D], got shape {tuple(link_states.shape)}")

        # 迁移到模型设备
        device = next(self.parameters()).device
        link_states = link_states.to(device=device, dtype=torch.float32)

        B, L, D = link_states.shape
        assert L <= self.num_links, f"L={L} > num_links={self.num_links}; re-init critic with more encoders or reduce input links"

        # 归一化
        link_states = self._normalize_inputs(link_states)

        # per-link: 仅 encoder，无 RNN
        tokens = []
        for l in range(L):
            enc = self.link_encoders[l](link_states[:, l, :])   # [B, T]
            tokens.append(enc.unsqueeze(1))                     # [B, 1, T]

        x = torch.cat(tokens, dim=1)    # [B, L, T]
        x = self.posenc(x)

        # 跨链路上下文
        ctx = self.tr(x)
        ctx = ctx.contiguous().clone()                           # [B, L, T]

        # 全局聚合（mean-pool）
        pooled = ctx.mean(dim=1)                                # [B, T]

        # 价值头 -> [B, 1] -> [B]
        values = self.value_head(pooled).squeeze(-1)            # [B]
        if squeeze_out:
            return values[0]
        return values


# -------------- minimal sanity check --------------
if __name__ == "__main__":
    # 6 维特征: [prop_s, tx_s, queue_len, hist_cnt, tail_finish_minus_now_s, current_time]
    feats = [
        [80e-6, 40e-6, 3.0, 5.0, 120e-6, 0.00],  # link 0
        [90e-6, 30e-6, 2.0, 1.0,  80e-6, 0.00],  # link 1
        [60e-6, 50e-6, 6.0, 9.0, 200e-6, 0.00],  # link 2
    ]
    critic = ValueCritic(num_links=4)   # 允许最多 4 条链路；运行时 L=3 也可以
    v1 = critic(feats)                  # 标量
    print("V(s) =", v1)
    v2 = critic([feats, feats])         # [B=2, L=3, D=6]
    print("V(s) batch shape =", v2)