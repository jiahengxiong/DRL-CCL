import numpy as np
import matplotlib.pyplot as plt

from load_dataset import load_snvang_dataset_by_day
from deploy import RNNDeployer
import math


def concat_day_segments(day_segs):
    """
    day_segs: List[DaySegment]
    return: np.ndarray, shape (T_total,)
    """
    xs = []
    for seg in day_segs:
        x = seg.x
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        xs.append(x.astype(np.float32))
    if len(xs) == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(xs, axis=0)


def run_stream(deployer, y_stream: np.ndarray, continual_learning: bool):
    """
    Long-running deployment stream (no per-day reset).
    Only reset once at the beginning.
    """
    deployer.begin_day()
    preds = np.empty((len(y_stream),), dtype=np.float32)
    for i, xt in enumerate(y_stream):
        preds[i] = deployer.predict(float(xt), continual_learning=continual_learning)
    return preds


def print_tail_metrics(y, preds_off, preds_on, prefix=""):
    """
    Print tail metrics (P95/P99 etc.) for one-step prediction errors.
    Alignment: preds[t] predicts y[t+1], so compare preds[:-1] vs y[1:].
    """
    e_off = preds_off[:-1] - y[1:]
    e_on  = preds_on[:-1]  - y[1:]

    ae_off = np.abs(e_off)
    ae_on  = np.abs(e_on)
    se_off = e_off ** 2
    se_on  = e_on ** 2

    qs = [50, 90, 95, 99, 99.5]

    def _line(name, arr_off, arr_on):
        print(f"\n{prefix}{name}")
        for q in qs:
            v_off = float(np.percentile(arr_off, q))
            v_on  = float(np.percentile(arr_on,  q))
            impr = 100.0 * (1.0 - (v_on / max(v_off, 1e-12)))
            print(f"  P{q:>4}: off={v_off:.6g} | on={v_on:.6g} | improv={impr:+.2f}%")

    _line("Absolute Error |e|", ae_off, ae_on)
    _line("Squared Error  e^2", se_off, se_on)

    # Tail-mean (CVaR-like) on absolute error
    def tail_mean(arr, tail_frac):
        k = max(1, int(np.ceil(len(arr) * tail_frac)))
        return float(np.mean(np.partition(arr, -k)[-k:]))

    for frac, name in [(0.05, "Top 5% mean"), (0.01, "Top 1% mean")]:
        off = tail_mean(ae_off, frac)
        on  = tail_mean(ae_on, frac)
        impr = 100.0 * (1.0 - on / max(off, 1e-12))
        print(f"\n{prefix}Tail {name} (AE): off={off:.6g} | on={on:.6g} | improv={impr:+.2f}%")


def plot_stream_results(y, preds_offline, preds_online, label: str):
    T = len(y)
    t = np.arange(T)

    # -------- Figure 1: time series --------
    plt.figure(figsize=(12, 4))
    plt.plot(t, y, label="Ground truth", color="black", linewidth=2)
    plt.plot(t, preds_offline, label="Prediction (offline)", linestyle="--")
    plt.plot(t, preds_online, label="Prediction (online CL)", linestyle="-")
    plt.xlabel("Time step")
    plt.ylabel("Traffic (scaled)")
    plt.title(f"Prediction on test stream ({label})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------- Figure 2: cumulative error --------
    # one-step alignment: preds[t] predicts y[t+1] => compare preds[:-1] vs y[1:]
    # 绝对误差
    err_off = np.abs(preds_offline[:-1] - y[1:])
    err_on = np.abs(preds_online[:-1] - y[1:])

    # 前 10 个最大误差（数值）
    top10_off = np.sort(err_off)[-30:]
    top10_on = np.sort(err_on)[-30:]

    print("OFFLINE top-10 AE:", top10_off)
    print("ONLINE  top-10 AE:", top10_on)

    cum_err_off = np.cumsum(err_off)
    cum_err_on  = np.cumsum(err_on)

    plt.figure(figsize=(12, 4))
    plt.plot(cum_err_off, label="Offline predictor", linestyle="--")
    plt.plot(cum_err_on, label="Online predictor (CL)", linestyle="-")
    plt.xlabel("Time step")
    plt.ylabel("Cumulative error")
    plt.title(f"Cumulative one-step error on test stream ({label})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(err_off, label="Offline predictor", linestyle="--")
    plt.plot(err_on, label="Online predictor (CL)", linestyle="-")
    plt.xlabel("Time step")
    plt.ylabel("Error")
    plt.title(f"Error on test stream ({label})")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # -------- load data --------
    data = load_snvang_dataset_by_day(
        "SNVAng__STTLng.txt",
        train_ratio=0.5,
        scale_range=(0.01, 1.0),
    )

    # concat ALL test segments into one long stream
    test_days = data["test_days"]
    y = concat_day_segments(test_days)

    if y.shape[0] < 2:
        raise ValueError(f"Test stream too short: {y.shape}")

    # label: show how many segments and date range
    start_day = test_days[0].day if len(test_days) > 0 else "N/A"
    end_day = test_days[-1].day if len(test_days) > 0 else "N/A"
    label = f"{len(test_days)} segments, {start_day} → {end_day}"

    # -------- run offline --------
    deployer_off = RNNDeployer.from_checkpoint("rnn_snvang.pt")
    preds_off = run_stream(deployer_off, y, continual_learning=False)

    # -------- run online --------
    deployer_on = RNNDeployer.from_checkpoint("rnn_snvang.pt")
    preds_on = run_stream(deployer_on, y, continual_learning=True)

    # quick scalar metrics
    mse_off = float(np.mean((preds_off[:-1] - y[1:]) ** 2))
    mse_on  = float(np.mean((preds_on[:-1]  - y[1:]) ** 2))
    print("OFFLINE MSE:", mse_off)
    print("ONLINE  MSE:", mse_on)

    # -------- tail metrics --------
    print_tail_metrics(y, preds_off, preds_on, prefix="[Test stream] ")

    # -------- plot --------
    plot_stream_results(
        y=y,
        preds_offline=preds_off,
        preds_online=preds_on,
        label=label,
    )