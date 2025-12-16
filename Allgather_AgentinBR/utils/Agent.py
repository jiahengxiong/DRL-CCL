# Allgather_AgentinBR/utils/Agent.py
from typing import Tuple, Optional
import math
import torch
import torch.nn as nn

# -----------------------------
# Minimal helpers (network-only)
# -----------------------------

def make_mlp(in_dim: int, hidden, out_dim: int, activation: str = "relu", layernorm: bool = False) -> nn.Sequential:
    acts = {"relu": nn.ReLU, "gelu": nn.GELU}
    Act = acts.get(activation, nn.ReLU)
    layers = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), Act()]
        if layernorm:
            layers += [nn.LayerNorm(h)]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, T]
        return self.dropout(x + self.pe[:, :x.size(1), :])

# -----------------------------
# Config (only architecture knobs)
# -----------------------------
class AgentConfig:
    def __init__(
        self,
        in_dim: int = 6,                 # ← 改成6: [prop_s, tx_s, queue_len, hist_cnt, tail_finish_minus_now_s, current_time]
        link_layers = (64, 64),          # per-link encoder hidden sizes
        transformer_dim: int = 64,       # model dim for Transformer / RNN state
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.0,
        head_layers = (64,),             # scoring head hidden sizes
        layernorm: bool = True,
        activation: str = "relu",
        use_positional_encoding: bool = False,
    ):
        self.in_dim = in_dim
        self.link_layers = tuple(link_layers)
        self.transformer_dim = transformer_dim
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_dropout = transformer_dropout
        self.head_layers = tuple(head_layers)
        self.layernorm = layernorm
        self.activation = activation
        self.use_positional_encoding = use_positional_encoding

# -----------------------------
# Network-only Agent with per-link RNN memory
# -----------------------------
class LinkPolicyAgent(nn.Module):
    """
    Per-link encoders -> per-link GRUCell (temporal memory) -> Transformer over links -> shared head -> per-link score.
    Inputs:
      link_states: [B, L, D] or [L, D]
    Output:
      scores:      [B, L] or [L]
    """
    def __init__(self, num_links: int, cfg: AgentConfig = None):
        super().__init__()
        if cfg is None:
            cfg = AgentConfig()
        self.cfg = cfg

        # per-link encoders (index-aligned up to num_links)
        self.link_encoders = nn.ModuleList([
            make_mlp(cfg.in_dim, cfg.link_layers, cfg.transformer_dim, cfg.activation, cfg.layernorm)
            for _ in range(num_links)
        ])
        self.num_links = len(self.link_encoders)

        # per-link temporal memory via a shared GRUCell (applied independently on each link slot)
        self.rnn_cell = None  # RNN removed

        # cross-link context
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

        # shared per-link scoring head
        self.mip_score = make_mlp(cfg.transformer_dim, cfg.head_layers, 1, cfg.activation, cfg.layernorm)

        # params init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # hidden state buffer: Optional[Tensor] with shape [B, num_links, T]
        self._h: Optional[torch.Tensor] = None

    # ---------- hidden state helpers ----------
    def reset_state(self) -> None:
        """Clear internal RNN memory."""
        self._h = None
        return None

    def get_state(self) -> Optional[torch.Tensor]:
        """Return a clone of current memory [B, L, T] or None."""
        return None

    def set_state(self, h: Optional[torch.Tensor]) -> None:
        """Set memory from external tensor (will be cloned)."""
        self._h = None
        return None

    def _ensure_state(self, B: int, device: torch.device, dtype: torch.dtype) -> None:
        """
        No-op: RNN removed, no hidden state maintained.
        """
        return None  # RNN removed: no hidden state maintained

    # ---------- preprocessing ----------
    def _normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D], feature order:
          [prop_s, tx_s, queue_len, hist_cnt, tail_finish_minus_now_s, current_time]
        归一化策略：
          - t_ref = median(prop + tx) per sample；prop/tx/tail 用 t_ref 缩放
          - hist_cnt/tail 用 log1p 压缩
          - current_time 仅 log1p 压缩，不做 z-score（避免抹平时间语义）
          - 其他列轻微截断 + z-score（按 batch 内 over links）
        """
        if x.size(-1) < 6:
            raise ValueError(f"Expected 6 features per link, got D={x.size(-1)}")
        x = x.clone()
        eps = 1e-9

        # robust reference time (per sample in batch): t_ref = median(prop + tx)
        prop = torch.clamp(x[:, :, 0], min=0.0)
        tx   = torch.clamp(x[:, :, 1], min=0.0)
        sum_pt = prop + tx                                  # [B,L]
        t_ref = sum_pt.median(dim=1, keepdim=True).values   # [B,1]
        t_ref = torch.clamp(t_ref, min=1e-6)

        # scale time columns by t_ref (prop, tx, tail)
        x[:, :, 0] = prop / t_ref
        x[:, :, 1] = tx   / t_ref
        x[:, :, 4] = torch.clamp(x[:, :, 4], min=0.0) / t_ref  # tail_finish_minus_now_s

        # log1p compression
        x[:, :, 3] = torch.log1p(torch.clamp(x[:, :, 3], min=0.0))  # hist_cnt
        x[:, :, 4] = torch.log1p(x[:, :, 4])                        # tail
        x[:, :, 5] = torch.log1p(torch.clamp(x[:, :, 5], min=0.0))  # current_time

        # light clipping for first 5 columns (exclude current_time)
        head5 = x[:, :, :5]
        mean_approx = head5.mean(dim=1, keepdim=True)
        std_approx  = head5.std(dim=1, keepdim=True, unbiased=False) + eps
        upper = mean_approx + 3.0 * std_approx
        lower = mean_approx - 3.0 * std_approx
        head5 = torch.max(torch.min(head5, upper), lower)

        # z-score for first 5 columns
        mean = head5.mean(dim=1, keepdim=True)
        std  = head5.std(dim=1, keepdim=True, unbiased=False) + eps
        head5 = (head5 - mean) / std

        x[:, :, :5] = head5
        return x

    # ---------- forward ----------
    def forward(self, link_states) -> torch.Tensor:
        """
        Compute per-link scores.
        Accepts:
          - link_states: shape [L, D] (preferred) or [B, L, D]; also supports Python list/tuple inputs.
        Returns:
          - scores:      shape [L] (if input was [L, D]) or [B, L] (if input was [B, L, D]).
        """
        # Coerce inputs to tensors and unify shapes
        if isinstance(link_states, (list, tuple)):
            link_states = torch.as_tensor(link_states, dtype=torch.float32)
        squeeze_out = False
        if link_states.dim() == 2:  # [L, D]
            link_states = link_states.unsqueeze(0)  # -> [1, L, D]
            squeeze_out = True
        elif link_states.dim() != 3:
            raise ValueError(f"link_states must be [L, D] or [B, L, D], got shape {tuple(link_states.shape)}")

        # Move to model device and enforce float32
        device = next(self.parameters()).device
        link_states = link_states.to(device=device, dtype=torch.float32)

        B, L, D = link_states.shape
        assert L <= self.num_links, f"L={L} > num_links={self.num_links}; re-init agent with more encoders or reduce input links"

        # normalize input features per batch
        link_states = self._normalize_inputs(link_states)

        # ensure hidden state buffer
        # self._ensure_state(B=B, device=device, dtype=link_states.dtype)

        # per-link encode only (no RNN)
        tokens = []
        for l in range(L):
            enc = self.link_encoders[l](link_states[:, l, :])   # [B, T]
            tokens.append(enc.unsqueeze(1))                     # [B, 1, T]

        x = torch.cat(tokens, dim=1)  # [B, L, T]
        x = self.posenc(x)

        # context across links
        ctx = self.tr(x)  # [B, L, T]

        # scoring
        scores = self.mip_score(ctx).squeeze(-1)  # [B, L]
        if squeeze_out:
            return scores[0]  # [L]
        return scores

    @torch.no_grad()
    def scores_prob(self, link_states, method: str = "softmax") -> torch.Tensor:
        """
        Convert raw scores (logits) into 0–1 probabilities per link.
        Input/Output shapes follow `forward`: [L, D] -> [L]; [B, L, D] -> [B, L].
        """
        logits = self.forward(link_states)  # no mask
        squeeze_out = False
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)  # [1, L]
            squeeze_out = True
        if method == "softmax":
            probs = torch.softmax(logits, dim=1)
        elif method == "sigmoid":
            probs = torch.sigmoid(logits)
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'softmax' or 'sigmoid'.")
        if squeeze_out:
            return probs[0]
        return probs

    @torch.no_grad()
    def argmax_action(self, link_states) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (scores, action_idx)."""
        scores = self.forward(link_states)
        if scores.dim() == 1:  # [L]
            action = scores.argmax(dim=0)
            return scores, action
        action = scores.argmax(dim=1)
        return scores, action

# -----------------------------
# Minimal sanity check
# -----------------------------
if __name__ == "__main__":
    # toy features: [prop_s, tx_s, queue_len, hist_cnt, tail_finish_minus_now_s, current_time]
    link_feats_t1 = [
        [80e-6, 40e-6, 3.0, 5.0, 120e-6, 0.00],  # link 0
        [90e-6, 30e-6, 2.0, 1.0,  80e-6, 0.00],  # link 1
        [60e-6, 50e-6, 6.0, 9.0, 200e-6, 0.00],  # link 2
    ]
    L = len(link_feats_t1)
    net = LinkPolicyAgent(num_links=L)

    # t=1
    scores1, act1 = net.argmax_action(link_feats_t1)
    print("t=1 logits:", scores1, "action:", act1.item())
    print("t=1 probs :", net.scores_prob(link_feats_t1, method="softmax"))

    # t=2 (time grows; queue/tail也可能变化)
    link_feats_t2 = [
        [80e-6, 40e-6, 4.0, 6.0, 150e-6, 0.01],
        [90e-6, 30e-6, 2.0, 1.0,  60e-6, 0.01],
        [60e-6, 50e-6, 5.0, 9.0, 210e-6, 0.01],
    ]
    scores2, act2 = net.argmax_action(link_feats_t2)
    print("t=2 logits:", scores2, "action:", act2.item())
    print("t=2 probs :", net.scores_prob(link_feats_t2, method="softmax"))

    # reset memory if你切换 episode
    net.reset_state()