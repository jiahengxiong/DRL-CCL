from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from model import OneStepRNN


def _infer_hidden_dim_from_state_dict(sd: dict) -> int:
    for k in ("cell.weight_hh", "cell.weight_hh_l0"):
        if k in sd:
            return int(sd[k].shape[1])
    for k, w in sd.items():
        if "weight_hh" in k:
            return int(w.shape[1])
    raise ValueError("Cannot infer hidden_dim from checkpoint state_dict keys.")


class RNNDeployer:
    """
    External usage pattern:

      deployer = RNNDeployer.from_checkpoint("rnn_snvang.pt", device=...)
      deployer.begin_day()  # optional (recommended)

      for x_t in day_truth:  # x_t is scalar (float) or shape (1,) etc.
          p_next = deployer.predict(x_t, continual_learning=True)

    Design:
      - predict() is the ONLY public runtime API you need.
      - continual_learning_update() is a separate function, called inside predict() when enabled.
      - By default we freeze GRUCell and only adapt head online (stable).
    """

    def __init__(
        self,
        model: OneStepRNN,
        lr: float = 1e-4,
        grad_clip: float = 1.0,
        freeze_cell: bool = True,
    ):
        self.model = model
        self.device = next(model.parameters()).device

        self.lr = float(lr)
        self.grad_clip = float(grad_clip)
        self.freeze_cell = bool(freeze_cell)

        self.loss_fn = nn.MSELoss()

        # optimizer + params (created lazily when continual_learning=True first time)
        self._optimizer = None
        self._optim_params = None

        # cache needed for update (uses p_t vs x_t after producing p_{t+1})
        self._prev_x = None           # torch.Tensor (1,1)
        self._prev_h_before = None    # torch.Tensor (1,H)

    # --------- lifecycle ---------
    def begin_day(self):
        """Call at the beginning of each day/segment."""
        self.model.reset_state()
        self._prev_x = None
        self._prev_h_before = None

    @staticmethod
    def from_checkpoint(ckpt_path: str, device: Optional[torch.device] = None) -> "RNNDeployer":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        hidden_dim = _infer_hidden_dim_from_state_dict(state_dict)
        model = OneStepRNN(input_dim=1, hidden_dim=hidden_dim).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.reset_state()

        return RNNDeployer(model=model)

    # --------- internal: optimizer setup ---------
    def _ensure_optimizer(self):
        if self._optimizer is not None:
            return

        if self.freeze_cell:
            for p in self.model.cell.parameters():
                p.requires_grad_(False)
            for p in self.model.head.parameters():
                p.requires_grad_(True)
            self._optim_params = list(self.model.head.parameters())
        else:
            for p in self.model.parameters():
                p.requires_grad_(True)
            self._optim_params = list(self.model.parameters())

        self._optimizer = torch.optim.Adam(self._optim_params, lr=self.lr)

    # --------- internal: online update ---------
    def continual_learning_update(self, x_t: torch.Tensor):
        """
        The simplest online update:
          - always do ONE gradient step on (p_t vs x_t)
          - p_t is recomputed from cached (prev_x, prev_h_before)
        """

        if self.lr == 0:
            return
        if self._prev_x is None or self._prev_h_before is None:
            return

        self._ensure_optimizer()
        self.model.train()

        # recompute p_t under current params
        h_tmp = self.model.cell(self._prev_x, self._prev_h_before)
        p_t = self.model.head(h_tmp)

        loss = (p_t - x_t).pow(2).mean()  # plain MSE
        # loss = torch.nn.functional.smooth_l1_loss(p_t, x_t, beta=0.02)


        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self._optim_params, self.grad_clip)
        self._optimizer.step()

        self.model.detach_state()
        self.model.eval()

    # --------- public: single-step prediction API ---------
    def predict(self, x_t, continual_learning: bool = False) -> float:
        """
        The ONLY runtime API:
          - input: current ground-truth x_t (scalar / numpy / tensor ok)
          - output: prediction p_{t+1} as python float
          - if continual_learning=True, will call continual_learning_update() after predicting.

        Note: we follow your requirement:
          1) predict next step first (so downstream can use it)
          2) then do online update using current truth
        """
        # normalize x_t -> torch (1,1)
        if isinstance(x_t, torch.Tensor):
            xt = x_t.detach().to(self.device).float()
            if xt.numel() != 1:
                raise ValueError(f"x_t must be scalar, got shape {tuple(xt.shape)}")
            xt = xt.view(1, 1)
        else:
            xt = torch.tensor([[float(x_t)]], device=self.device, dtype=torch.float32)

        # ---- 1) inference first: produce p_{t+1} and advance hidden state ----
        with torch.no_grad():
            h_before = self.model._h.detach() if self.model._h is not None else None
            p_next = self.model.step(xt)      # updates internal hidden state
            p_next_f = float(p_next.item())

        # ---- 2) optional online update: using (p_t vs x_t) ----
        if continual_learning:
            self.continual_learning_update(xt)

        # ---- 3) refresh cache for next update ----
        self._prev_x = xt.detach()
        if h_before is None:
            # hidden before consuming first sample is zeros (conceptually)
            self._prev_h_before = torch.zeros(1, self.model.hidden_dim, device=self.device, dtype=torch.float32)
        else:
            self._prev_h_before = h_before.detach()

        return p_next_f


# ---------------- example usage ----------------
if __name__ == "__main__":
    from load_dataset import load_snvang_dataset_by_day

    data = load_snvang_dataset_by_day("SNVAng__STTLng.txt", train_ratio=0.8, scale_range=(0.01, 1.0))
    dayseg = data["test_days"][1]
    y = dayseg.x[:, 0].astype(np.float32)  # (T,)

    deployer = RNNDeployer.from_checkpoint("rnn_snvang.pt")
    deployer.begin_day()

    preds = []
    for xt in y:
        preds.append(deployer.predict(xt, continual_learning=True))

    preds = np.asarray(preds, dtype=np.float32)
    mse = float(np.mean((preds[:-1] - y[1:]) ** 2))
    print(dayseg.day, mse)
    print(y[0:10])
    print(preds[0:10])