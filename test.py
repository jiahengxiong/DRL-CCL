import numpy as np
import torch
import torch.nn as nn

from NewAgent.load_dataset import DaySegment
from NewAgent.model import OneStepRNN
from NewAgent.train import train_one_epoch
from NewAgent.deploy import RNNDeployer

torch.manual_seed(0)
np.random.seed(0)

# =========================
# 1. 用公式生成数据
# =========================
def generate(T=1000):
    t = np.arange(T)
    k = np.floor(t / 10)

    sum_mid = 0.55
    sum_amp = 0.45
    sum0 = sum_mid + sum_amp * ((-1) ** k)
    sum1 = sum_mid - sum_amp * ((-1) ** k)

    frac0 = 0.80 + 0.05 * np.sin(0.10 * t)
    frac1 = 0.80 + 0.05 * np.cos(0.07 * t)
    d0 = sum0 * frac0
    d1 = sum1 * frac1

    r02 = (sum0 + d0 * ((-1) ** t)) / 2
    r03 = (sum0 - d0 * ((-1) ** t)) / 2
    r12 = (sum1 + d1 * ((-1) ** (t + 1))) / 2
    r13 = (sum1 - d1 * ((-1) ** (t + 1))) / 2

    data = np.stack([r02, r03, r12, r13], axis=1).astype(np.float32)
    return data


def main():
    T = 10000
    data = generate(T)
    n_print = min(200, data.shape[0])
    print(f"First {n_print} values table")
    print("i\t(0,2)\t(0,3)\t(0,2)+(0,3)\t(1,2)\t(1,3)\t(1,2)+(1,3)")
    for i in range(n_print):
        r02, r03, r12, r13 = (float(v) for v in data[i, :4])
        s0 = r02 + r03
        s1 = r12 + r13
        print(f"{i}\t{r02:.6f}\t{r03:.6f}\t{s0:.6f}\t{r12:.6f}\t{r13:.6f}\t{s1:.6f}")

    series = data[:, 0].astype(np.float32)
    split = int(series.shape[0] * 0.5)
    train_series = series[:split]
    test_series = series[split:]

    train_days = [DaySegment(day="train", x=train_series.reshape(-1, 1))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OneStepRNN(input_dim=1, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 200
    eval_len = min(800, len(test_series))
    for epoch in range(epochs):
        train_mse = train_one_epoch(model, optimizer, train_days, device)
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                x_eval = torch.tensor(test_series[:eval_len], dtype=torch.float32, device=device).view(1, -1, 1)
                pred_eval = model(x_eval, reset=True)
                loss_eval = loss_fn(pred_eval[:, :-1, :], x_eval[:, 1:, :]).item()
                model.reset_state()
            print(f"Epoch {epoch:3d} | train_mse={train_mse:.8f} | test_mse@{eval_len}={loss_eval:.8f}")

    model.eval()
    deployer = RNNDeployer(model=model, lr=0.0, freeze_cell=True)
    deployer.begin_day()
    pred = np.empty((len(test_series),), dtype=np.float32)
    for i, xt in enumerate(test_series):
        pred[i] = np.float32(deployer.predict(float(xt), continual_learning=False))

    test_mse = float(np.mean((pred[:-1] - test_series[1:]) ** 2))
    test_mae = float(np.mean(np.abs(pred[:-1] - test_series[1:])))
    print("\nRNN prediction on (0,2) link (test set)")
    print(f"Test MSE: {test_mse:.10f}")
    print(f"Test MAE: {test_mae:.10f}")
    n_show = min(50, len(test_series) - 1)
    print("i\tgt\tpred\tabs_err")
    for i in range(n_show):
        gt = float(test_series[i + 1])
        pv = float(pred[i])
        print(f"{i}\t{gt:.6f}\t{pv:.6f}\t{abs(pv - gt):.6f}")

if __name__ == "__main__":
    main()
