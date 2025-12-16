import torch
import torch.nn as nn
from typing import Optional, Tuple


class _PeakFeatureGRUCell(nn.Module):
    """
    Wrapper with the SAME call signature as nn.GRUCell:
        h_next = cell(x_raw, h_prev)
    where x_raw is (B,1).

    Internally, it builds 4-D peak-aware features:
        [x, dx, x-ema, mad]
    and feeds them to an inner GRUCell(input_size=4).

    Crucially, it maintains *one-step-delayed* feature-state snapshots so that
    external continual learning can safely call:
        cell(prev_x, prev_h_before)
    AFTER the model has already consumed the current x_t.
    """

    def __init__(self, hidden_dim: int, ema_alpha: float = 0.15):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.ema_alpha = float(ema_alpha)

        self.gru = nn.GRUCell(input_size=4, hidden_size=self.hidden_dim)

        # runtime feature buffers (current time)
        self._prev_x: Optional[torch.Tensor] = None  # (B,1)
        self._ema: Optional[torch.Tensor] = None     # (B,1)
        self._mad: Optional[torch.Tensor] = None     # (B,1)

        # --- two-stage cache to support "predict first, then update prev step" ---
        # pending: state BEFORE consuming prev_x  (used when CL recomputes p_t)
        self._pending_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None  # (prev_x, ema, mad)
        self._pending_x: Optional[torch.Tensor] = None  # (B,1) raw x for sanity/debug (not strictly needed)

        # next: state BEFORE consuming current x (becomes pending after prev is used)
        self._next_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self._next_x: Optional[torch.Tensor] = None

    def reset_feat_state(self):
        self._prev_x = None
        self._ema = None
        self._mad = None
        self._pending_state = None
        self._pending_x = None
        self._next_state = None
        self._next_x = None

    def detach_feat_state(self):
        def _det(x):
            return None if x is None else x.detach()

        self._prev_x = _det(self._prev_x)
        self._ema = _det(self._ema)
        self._mad = _det(self._mad)

        if self._pending_state is not None:
            self._pending_state = tuple(s.detach() for s in self._pending_state)
        self._pending_x = _det(self._pending_x)

        if self._next_state is not None:
            self._next_state = tuple(s.detach() for s in self._next_state)
        self._next_x = _det(self._next_x)

    def _ensure_buffers(self, B: int, device, dtype):
        if self._prev_x is None or self._prev_x.size(0) != B or self._prev_x.device != device or self._prev_x.dtype != dtype:
            self._prev_x = torch.zeros(B, 1, device=device, dtype=dtype)
            self._ema = torch.zeros(B, 1, device=device, dtype=dtype)
            self._mad = torch.zeros(B, 1, device=device, dtype=dtype)

    @staticmethod
    def _build_feat_from_state(x_raw: torch.Tensor,
                              prev_x: torch.Tensor,
                              ema: torch.Tensor,
                              mad: torch.Tensor,
                              alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build features using PROVIDED (prev_x, ema, mad) without mutating them.
        Returns:
          feat: (B,4)
          prev_x_new, ema_new, mad_new  (for normal runtime evolution)
        """
        dx = x_raw - prev_x
        ema_new = alpha * x_raw + (1.0 - alpha) * ema
        abs_dev = torch.abs(x_raw - ema_new)
        mad_new = alpha * abs_dev + (1.0 - alpha) * mad
        x_center = x_raw - ema_new
        feat = torch.cat([x_raw, dx, x_center, mad_new], dim=1)  # (B,4)
        return feat, x_raw, ema_new, mad_new

    def forward(self, x_raw: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        x_raw: (B,1)  (external unchanged)
        h_prev: (B,H)
        return: h_next (B,H)
        """
        if x_raw.dim() != 2 or x_raw.size(1) != 1:
            raise ValueError(f"cell input must be (B,1), got {tuple(x_raw.shape)}")
        B = x_raw.size(0)
        self._ensure_buffers(B, x_raw.device, x_raw.dtype)

        # Case A: Grad-enabled call => this is the EXTERNAL continual-learning recompute of prev step.
        # We MUST use the one-step delayed pending feature-state (state BEFORE consuming prev_x).
        if torch.is_grad_enabled():
            if self._pending_state is None:
                # Not enough cached history to recompute safely
                return self.gru(torch.cat([x_raw, x_raw * 0, x_raw * 0, x_raw * 0], dim=1), h_prev)  # harmless fallback

            prev_x0, ema0, mad0 = self._pending_state
            feat, _, _, _ = self._build_feat_from_state(x_raw, prev_x0, ema0, mad0, self.ema_alpha)

            h_next = self.gru(feat, h_prev)

            # After using pending once, promote next -> pending (so next update aligns correctly)
            self._pending_state = self._next_state
            self._pending_x = self._next_x
            self._next_state = None
            self._next_x = None

            return h_next

        # Case B: No-grad call => normal deployment / inference.
        # We must evolve the runtime feature buffers AND prepare caches for the next CL recompute.
        # Shift: if pending is empty and we already have a "next" (from previous step), promote it now.
        if self._pending_state is None and self._next_state is not None:
            self._pending_state = self._next_state
            self._pending_x = self._next_x
            self._next_state = None
            self._next_x = None

        # Save state BEFORE consuming current x_raw as "next" (it will be needed next time)
        state_before = (self._prev_x.detach(), self._ema.detach(), self._mad.detach())
        self._next_state = state_before
        self._next_x = x_raw.detach()

        feat, prev_x_new, ema_new, mad_new = self._build_feat_from_state(
            x_raw, self._prev_x, self._ema, self._mad, self.ema_alpha
        )

        # Evolve runtime buffers
        self._prev_x = prev_x_new
        self._ema = ema_new
        self._mad = mad_new

        return self.gru(feat, h_prev)


class OneStepRNN(nn.Module):
    """
    External API UNCHANGED:
      - step(x_t): x_t (B,1) -> p_{t+1} (B,1)
      - forward(x_seq, reset=True): x_seq (B,T,1) -> pred_seq (B,T,1)
    Hidden state and peak-aware feature state are maintained INSIDE.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, ema_alpha: float = 0.15):
        super().__init__()
        assert input_dim == 1, "Keep external input_dim=1."
        self.input_dim = input_dim
        self.hidden_dim = int(hidden_dim)

        # IMPORTANT: self.cell is now a wrapper that still accepts (B,1) input.
        self.cell = _PeakFeatureGRUCell(hidden_dim=self.hidden_dim, ema_alpha=ema_alpha)
        self.head = nn.Linear(self.hidden_dim, 1)

        self._h: Optional[torch.Tensor] = None  # (B,H)

    # -------------------------
    # state management
    # -------------------------
    def reset_state(self) -> None:
        self._h = None
        self.cell.reset_feat_state()

    def detach_state(self) -> None:
        if self._h is not None:
            self._h = self._h.detach()
        self.cell.detach_feat_state()

    def _init_state(self, B: int, device, dtype) -> None:
        self._h = torch.zeros(B, self.hidden_dim, device=device, dtype=dtype)

    # -------------------------
    # deployment: single step
    # -------------------------
    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        if x_t.dim() != 2 or x_t.size(-1) != 1:
            raise ValueError(f"x_t must be (B,1), got {tuple(x_t.shape)}")

        B = x_t.size(0)
        if self._h is None or self._h.size(0) != B or self._h.device != x_t.device or self._h.dtype != x_t.dtype:
            self._init_state(B, x_t.device, x_t.dtype)

        # cell wrapper will internally build peak-aware features and update its own buffers
        self._h = self.cell(x_t, self._h)
        return self.head(self._h)

    # -------------------------
    # training: full sequence
    # -------------------------
    def forward(self, x_seq: torch.Tensor, reset: bool = True) -> torch.Tensor:
        if x_seq.dim() != 3 or x_seq.size(-1) != 1:
            raise ValueError(f"x_seq must be (B,T,1), got {tuple(x_seq.shape)}")

        if reset:
            self.reset_state()

        B, T, _ = x_seq.shape
        if self._h is None or self._h.size(0) != B or self._h.device != x_seq.device or self._h.dtype != x_seq.dtype:
            self._init_state(B, x_seq.device, x_seq.dtype)

        preds = []
        for t in range(T):
            preds.append(self.step(x_seq[:, t, :]))

        return torch.stack(preds, dim=1)