import torch
import torch.nn as nn
import numpy as np

from NewAgent.load_dataset import load_snvang_dataset_by_day
from NewAgent.model import OneStepRNN


def train_one_epoch(model, optimizer, train_days, device):
    model.train()
    loss_fn = nn.MSELoss()

    xs = [dayseg.x for dayseg in train_days if getattr(dayseg, "x", None) is not None and len(dayseg.x) > 0]
    if len(xs) == 0:
        return 0.0

    x_np = np.concatenate(xs, axis=0)
    x = torch.tensor(x_np, dtype=torch.float32, device=device).view(1, -1, 1)  # (B=1, T, 1)
    T = x.shape[1]
    if T < 3:
        return 0.0

    tbptt = 256
    optimizer.zero_grad(set_to_none=True)
    model.reset_state()

    total_loss = 0.0
    total_points = 0
    chunk_loss = None
    chunk_steps = 0

    for t in range(T - 1):
        p_next = model.step(x[:, t, :])
        y_next = x[:, t + 1, :]
        step_loss = loss_fn(p_next, y_next)

        if chunk_loss is None:
            chunk_loss = step_loss
        else:
            chunk_loss = chunk_loss + step_loss

        total_loss += float(step_loss.item())
        total_points += 1
        chunk_steps += 1

        if chunk_steps >= tbptt:
            (chunk_loss / chunk_steps).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model.detach_state()
            chunk_loss = None
            chunk_steps = 0

    if chunk_steps > 0 and chunk_loss is not None:
        (chunk_loss / chunk_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    model.reset_state()
    return total_loss / max(1, total_points)


@torch.no_grad()
def eval_by_day(model, test_days, device):
    model.eval()
    loss_fn = nn.MSELoss()

    xs = [dayseg.x for dayseg in test_days if getattr(dayseg, "x", None) is not None and len(dayseg.x) > 0]
    if len(xs) == 0:
        return 0.0

    x_np = np.concatenate(xs, axis=0)
    x = torch.tensor(x_np, dtype=torch.float32, device=device).view(1, -1, 1)
    T = x.shape[1]
    if T < 3:
        return 0.0

    model.reset_state()
    total_loss = 0.0
    total_points = 0
    for t in range(T - 1):
        p_next = model.step(x[:, t, :])
        y_next = x[:, t + 1, :]
        total_loss += float(loss_fn(p_next, y_next).item())
        total_points += 1
    model.reset_state()
    return total_loss / max(1, total_points)


def train_rnn_from_path_and_save(
    data_path: str,
    save_path: str,
    train_ratio: float = 0.5,
    scale_range=(0.01, 1.0),
    hidden_dim: int = 64,
    lr: float = 1e-3,
    epochs: int = 5,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    device: torch.device = None,
):
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_snvang_dataset_by_day(
        data_path,
        train_ratio=train_ratio,
        scale_range=scale_range
    )
    train_days = data["train_days"]
    test_days = data["test_days"]
    model = OneStepRNN(input_dim=1, hidden_dim=hidden_dim).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_te = float("inf")
    best_state = None
    no_improve = 0
    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, optimizer, train_days, dev)
        te = eval_by_day(model, test_days, dev) if len(test_days) > 0 else float("nan")
        print(f"epoch {ep:03d} | train_mse={tr:.6f} | test_mse={te:.6f}")
        if early_stop_patience > 0 and np.isfinite(te):
            if te < (best_te - float(early_stop_min_delta)):
                best_te = float(te)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= int(early_stop_patience):
                    print(f"early_stop at epoch {ep:03d} | best_test_mse={best_te:.6f}")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "scale_info": data["scale_info"]}, save_path)
    return {"train_days": len(train_days), "test_days": len(test_days), "save_path": save_path}


def train_rnn_from_loaded_data_and_save(
    data: dict,
    save_path: str,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    epochs: int = 5,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    device: torch.device = None,
):
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_days = data.get("train_days", [])
    test_days = data.get("test_days", [])
    model = OneStepRNN(input_dim=1, hidden_dim=hidden_dim).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_te = float("inf")
    best_state = None
    no_improve = 0
    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, optimizer, train_days, dev)
        te = eval_by_day(model, test_days, dev) if len(test_days) > 0 else float("nan")
        print(f"epoch {ep:03d} | train_mse={tr:.6f} | test_mse={te:.6f}")
        if early_stop_patience > 0 and np.isfinite(te):
            if te < (best_te - float(early_stop_min_delta)):
                best_te = float(te)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= int(early_stop_patience):
                    print(f"early_stop at epoch {ep:03d} | best_test_mse={best_te:.6f}")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "scale_info": data.get("scale_info")}, save_path)
    return {"train_days": len(train_days), "test_days": len(test_days), "save_path": save_path}


def main():
    # ====== config ======
    data_path = "SNVAng__STTLng.txt"
    train_ratio = 0.5
    scale_range = (0.01, 1.0)

    hidden_dim = 64
    lr = 1e-3
    epochs = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== load data (by day) ======
    data = load_snvang_dataset_by_day(
        data_path,
        train_ratio=train_ratio,
        scale_range=scale_range
    )
    train_days = data["train_days"]
    test_days = data["test_days"]

    print(f"days train/test: {len(train_days)}/{len(test_days)}")
    print("scale_info:", data["scale_info"])

    # ====== model ======
    model = OneStepRNN(input_dim=1, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ====== train ======
    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, optimizer, train_days, device)
        te = eval_by_day(model, test_days, device) if len(test_days) > 0 else float("nan")
        print(f"epoch {ep:03d} | train_mse={tr:.6f} | test_mse={te:.6f}")

    # 你想保存模型就加这句
    torch.save({"state_dict": model.state_dict(), "scale_info": data["scale_info"]}, "rnn_snvang.pt")


if __name__ == "__main__":
    main()
