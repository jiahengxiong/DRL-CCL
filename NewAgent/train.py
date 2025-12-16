import torch
import torch.nn as nn

from load_dataset import load_snvang_dataset_by_day   # 你之前的按天 loader
from model import OneStepRNN                          # 你之前的 RNN 模型


def train_one_epoch(model, optimizer, train_days, device):
    model.train()
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_points = 0

    for dayseg in train_days:
        x = torch.tensor(dayseg.x, dtype=torch.float32, device=device).view(1, -1, 1)  # (B=1, T, 1)
        T = x.shape[1]
        if T < 3:
            continue

        model.reset_state()               # 每天一个连续片段：从零状态开始
        pred = model(x)                   # (1, T, 1), pred[:, t] 预测 x_{t+1}

        # 对齐：用 pred[:, :-1] 对齐 x[:, 1:]
        loss = loss_fn(pred[:, :-1, :], x[:, 1:, :])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 训练完这一天就结束了，state 不需要跨天保留
        model.reset_state()

        total_loss += float(loss.item()) * (T - 1)
        total_points += (T - 1)

    return total_loss / max(1, total_points)


@torch.no_grad()
def eval_by_day(model, test_days, device):
    model.eval()
    loss_fn = nn.MSELoss(reduction="sum")

    total_loss = 0.0
    total_points = 0

    for dayseg in test_days:
        x = torch.tensor(dayseg.x, dtype=torch.float32, device=device).view(1, -1, 1)
        T = x.shape[1]
        if T < 3:
            continue

        model.reset_state()
        pred = model(x)
        loss = loss_fn(pred[:, :-1, :], x[:, 1:, :])

        total_loss += float(loss.item())
        total_points += (T - 1)

    return total_loss / max(1, total_points)


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