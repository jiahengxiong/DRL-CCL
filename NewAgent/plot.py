import numpy as np
import matplotlib.pyplot as plt
import torch
import copy

from load_dataset import load_snvang_dataset_by_day
from deploy import RNNDeployer
from train import train_one_epoch
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


def run_stream_with_replay(deployer, test_days, initial_train_days=None, continual_learning=False):
    """
    Run stream with Experience Replay Buffer and periodic retraining.
    
    Args:
        deployer: RNNDeployer instance
        test_days: List[DaySegment] for testing
        initial_train_days: List[DaySegment] for initial training buffer
        continual_learning: bool, whether to enable online fine-tuning (single step) IN ADDITION to replay.
                            Default is False (replay only).
    """
    deployer.begin_day()
    
    # Initialize replay buffer with initial training data if provided
    # If initial_train_days is provided, we should respect the max_buffer_size immediately
    max_buffer_size = 10
    
    buffer = []
    if initial_train_days:
        # Keep only the last N segments if initial buffer is too large
        buffer = list(initial_train_days)[-max_buffer_size:]
    
    # Store all predictions
    all_preds = []
    
    # Hyperparameters for retraining
    retrain_interval = 5  # segments
    soft_update_alpha = 0.5  # new_weight * alpha + old_weight * (1 - alpha)
    retrain_lr = 1e-3 # Back to standard LR
    retrain_epochs = 1 # As requested
    
    # Setup optimizer for retraining (we need a new one or reuse? train_one_epoch takes optimizer)
    # We will create a fresh optimizer each time we retrain to reset momentum etc., 
    # or we can keep one. train.py creates one per run. Let's create one inside the loop or reuse.
    # Since we are soft-updating the model weights manually, the optimizer state might get out of sync 
    # if we persist it. Safer to re-create optimizer for each retraining session.
    
    print(f"Starting Replay Stream: buffer size={len(buffer)}, retrain_interval={retrain_interval}")
    
    for i, dayseg in enumerate(test_days):
        # 1. Predict for current day (segment)
        # We need to predict point-by-point to mimic streaming, 
        # or we can just predict the whole day if we don't do intra-day online updates.
        # But run_stream does point-by-point. Let's be consistent.
        
        y_day = dayseg.x
        if y_day.ndim == 2 and y_day.shape[1] == 1:
            y_day = y_day[:, 0]
        y_day = y_day.astype(np.float32)
        
        preds_day = np.empty((len(y_day),), dtype=np.float32)
        
        # Reset state for the day (assuming independent days for RNN state, as per deployer.begin_day)
        deployer.begin_day()
        
        for t, xt in enumerate(y_day):
            # Predict
            preds_day[t] = deployer.predict(float(xt), continual_learning=continual_learning)
        
        all_preds.append(preds_day)
        
        # 2. Add to buffer
        buffer.append(dayseg)
        # Enforce max buffer size (FIFO)
        if len(buffer) > max_buffer_size:
            buffer.pop(0)
        
        # 3. Periodic Retraining
        if (i + 1) % retrain_interval == 0:
            print(f"[{i+1}/{len(test_days)}] Retraining on buffer (size {len(buffer)})...")
            
            # Backup current weights
            old_state_dict = copy.deepcopy(deployer.model.state_dict())
            
            # Train (we use the model in-place, then soft-update back)
            # Create a fresh optimizer for this retraining phase
            # Note: train_one_epoch sets model to train() mode.
            optimizer = torch.optim.Adam(deployer.model.parameters(), lr=retrain_lr)
            
            for _ in range(retrain_epochs):
                train_one_epoch(deployer.model, optimizer, buffer, deployer.device)
            
            # Get new weights
            new_state_dict = deployer.model.state_dict()
            
            # Soft Update
            mixed_state_dict = {}
            with torch.no_grad():
                for key in old_state_dict:
                    # theta_new = alpha * theta_trained + (1 - alpha) * theta_old
                    # Note: user said "new model weight * 0.4 + old model weight"
                    # We assume user meant: final = 0.4 * new + 0.6 * old
                    w_old = old_state_dict[key]
                    w_new = new_state_dict[key]
                    mixed_state_dict[key] = soft_update_alpha * w_new + (1.0 - soft_update_alpha) * w_old
            
            # Load mixed weights
            deployer.model.load_state_dict(mixed_state_dict)
            
            # Critical: Reset the online optimizer because weights have changed abruptly!
            # The momentum states in the old optimizer are now invalid/stale.
            deployer._optimizer = None 
            
            deployer.model.eval() # Switch back to eval mode
            
    return np.concatenate(all_preds, axis=0)


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


def print_tail_metrics_comparison(y, preds_dict, prefix=""):
    """
    Print tail metrics for multiple predictors.
    Prints two tables: Absolute Error and Squared Error.
    """
    qs = [50, 90, 95, 99, 99.5]
    
    # Calculate errors
    abs_errors = {}
    sq_errors = {}
    for name, preds in preds_dict.items():
        e = preds[:-1] - y[1:]
        abs_errors[name] = np.abs(e)
        sq_errors[name] = e ** 2

    # Helper for improvement calculation
    def get_improvement(base_val, new_val):
        if base_val < 1e-12: return 0.0
        return (base_val - new_val) / base_val * 100.0

    # --- Table 1: Absolute Error ---
    print(f"\n{prefix}=== Tail Metrics Comparison (Absolute Error) ===")
    header = f"{'Metric':<10}"
    for name in preds_dict.keys():
        header += f" | {name:<20}"
    # Add improvement columns
    header += " | Improv(FT)    | Improv(CL)"
    print(header)
    print("-" * len(header))
    
    for q in qs:
        row = f"P{q:<9}"
        vals = {}
        for name in preds_dict.keys():
            val = float(np.percentile(abs_errors[name], q))
            vals[name] = val
            row += f" | {val:.6f}           "
        
        # Calculate improvement vs Offline
        imp_ft = get_improvement(vals["Offline"], vals["Online FT"])
        imp_cl = get_improvement(vals["Offline"], vals["Continual Learning"])
        row += f" | {imp_ft:+.2f}%       | {imp_cl:+.2f}%"
        print(row)

    # --- Table 2: Squared Error ---
    print(f"\n{prefix}=== Tail Metrics Comparison (Squared Error) ===")
    print(header) # Reuse header
    print("-" * len(header))
    
    for q in qs:
        row = f"P{q:<9}"
        vals = {}
        for name in preds_dict.keys():
            val = float(np.percentile(sq_errors[name], q))
            vals[name] = val
            row += f" | {val:.6f}           "
            
        # Calculate improvement vs Offline
        imp_ft = get_improvement(vals["Offline"], vals["Online FT"])
        imp_cl = get_improvement(vals["Offline"], vals["Continual Learning"])
        row += f" | {imp_ft:+.2f}%       | {imp_cl:+.2f}%"
        print(row)
        
    # Tail Means (CVaR) - usually on Absolute Error
    print(f"\n{prefix}=== Tail Mean (Top k% Absolute Error) ===")
    print(header)
    print("-" * len(header))
    
    def tail_mean(arr, tail_frac):
        k = max(1, int(np.ceil(len(arr) * tail_frac)))
        return float(np.mean(np.partition(arr, -k)[-k:]))

    for frac, label in [(0.05, "Top 5%"), (0.01, "Top 1%")]:
        row = f"{label:<10}"
        vals = {}
        for name in preds_dict.keys():
            val = tail_mean(abs_errors[name], frac)
            vals[name] = val
            row += f" | {val:.6f}           "
        
        # Calculate improvement vs Offline
        imp_ft = get_improvement(vals["Offline"], vals["Online FT"])
        imp_cl = get_improvement(vals["Offline"], vals["Continual Learning"])
        row += f" | {imp_ft:+.2f}%       | {imp_cl:+.2f}%"
        print(row)


def plot_combined_cumulative_error(y, preds_dict, styles_dict):
    """
    Plots all cumulative error lines in a single chart.
    """
    plt.figure(figsize=(12, 6))
    
    for name, preds in preds_dict.items():
        err = np.abs(preds[:-1] - y[1:])
        cum_err = np.cumsum(err)
        style = styles_dict.get(name, {"color": "black", "style": "-"})
        plt.plot(cum_err, label=name, color=style["color"], linestyle=style["style"], linewidth=2)

    plt.xlabel("Time step")
    plt.ylabel("Cumulative Absolute Error")
    plt.title("Overall Comparison: Cumulative Absolute Error")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_pair_comparison(y, preds1, name1, preds2, name2, label, color1="blue", color2="red", style1="--", style2="-"):
    """
    Helper to plot a pair of predictors against Ground Truth.
    (Cumulative error is now handled separately)
    """
    T = len(y)
    t = np.arange(T)
    
    # 1. Time Series
    plt.figure(figsize=(12, 4))
    plt.plot(t, y, label="Ground truth", color="black", linewidth=1.5, alpha=0.5)
    plt.plot(t, preds1, label=name1, color=color1, linestyle=style1)
    plt.plot(t, preds2, label=name2, color=color2, linestyle=style2)
    plt.xlabel("Time step")
    plt.ylabel("Traffic (scaled)")
    plt.title(f"Comparison: {name1} vs {name2} ({label})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Absolute Error (Instantaneous)
    err1 = np.abs(preds1[:-1] - y[1:])
    err2 = np.abs(preds2[:-1] - y[1:])
    plt.figure(figsize=(12, 4))
    plt.plot(err1, label=name1, color=color1, linestyle=style1, alpha=0.7, linewidth=0.8)
    plt.plot(err2, label=name2, color=color2, linestyle=style2, alpha=0.7, linewidth=0.8)
    plt.xlabel("Time step")
    plt.ylabel("Absolute Error")
    plt.title(f"Instantaneous Error: {name1} vs {name2}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_stream_results_comparison(y, preds_dict, label: str):
    """
    Generates comparisons as requested:
    1. A combined Cumulative Error plot for all methods.
    2. Pairwise comparisons for Time Series and Instantaneous Error.
    """
    preds_off = preds_dict["Offline"]
    preds_ft = preds_dict["Online FT"]
    preds_cl = preds_dict["Continual Learning"]
    
    # Styles as requested:
    # Offline: Red dashed
    # Online FT: Green solid
    # Continual Learning: Yellow solid
    
    style_off = {"color": "red", "style": "--"}
    style_ft = {"color": "green", "style": "-"}
    style_cl = {"color": "yellow", "style": "-"}
    
    styles_dict = {
        "Offline": style_off,
        "Online FT": style_ft,
        "Continual Learning": style_cl
    }

    # -------- 1. Combined Cumulative Error --------
    print("\nPlotting Combined Cumulative Error...")
    plot_combined_cumulative_error(y, preds_dict, styles_dict)
    
    # -------- 2. Pairwise comparisons --------
    print("\nPlotting Pairwise 1: Offline vs Online FT")
    plot_pair_comparison(y, preds_off, "Offline", preds_ft, "Online FT", label, 
                         color1=style_off["color"], color2=style_ft["color"],
                         style1=style_off["style"], style2=style_ft["style"])
    
    print("\nPlotting Pairwise 2: Offline vs Continual Learning")
    plot_pair_comparison(y, preds_off, "Offline", preds_cl, "Continual Learning", label, 
                         color1=style_off["color"], color2=style_cl["color"],
                         style1=style_off["style"], style2=style_cl["style"])
    
    print("\nPlotting Pairwise 3: Online FT vs Continual Learning")
    plot_pair_comparison(y, preds_ft, "Online FT", preds_cl, "Continual Learning", label, 
                         color1=style_ft["color"], color2=style_cl["color"],
                         style1=style_ft["style"], style2=style_cl["style"])


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
    
    train_days = data["train_days"]

    # 1. Offline Only (No updates)
    print("\n--- Running Scenario 1: Offline Only ---")
    deployer_off = RNNDeployer.from_checkpoint("rnn_snvang.pt")
    preds_off = run_stream(deployer_off, y, continual_learning=False)

    # 2. Offline + Online Fine-tuning (No Replay Buffer)
    print("\n--- Running Scenario 2: Offline + Online Fine-tuning (No Replay) ---")
    deployer_ft = RNNDeployer.from_checkpoint("rnn_snvang.pt")
    preds_ft = run_stream(deployer_ft, y, continual_learning=True)

    # 3. Offline + Online Fine-tuning + Replay Buffer (Periodic Retraining)
    print("\n--- Running Scenario 3: Offline + Online FT + Replay Buffer ---")
    deployer_replay = RNNDeployer.from_checkpoint("rnn_snvang.pt")
    preds_replay = run_stream_with_replay(
        deployer_replay, 
        test_days, 
        initial_train_days=None, # User requested: "不要包含offline的训练数据"
        continual_learning=False # Enable online fine-tuning + Replay
    )

    # -------- Metrics & Plot --------
    preds_dict = {
        "Offline": preds_off,
        "Online FT": preds_ft,
        "Continual Learning": preds_replay
    }

    # MSE & MAE Calculation
    print("\n=== Metrics Comparison (MSE & MAE) ===")
    print(f"{'Method':<20} | {'MSE':<12} | {'MAE':<12}")
    print("-" * 50)
    for name, preds in preds_dict.items():
        mse = float(np.mean((preds[:-1] - y[1:]) ** 2))
        mae = float(np.mean(np.abs(preds[:-1] - y[1:])))
        print(f"{name:<20} | {mse:.6f}     | {mae:.6f}")

    # Tail Metrics
    print_tail_metrics_comparison(y, preds_dict, prefix="[Test stream] ")

    # Plot
    plot_stream_results_comparison(y, preds_dict, label)
