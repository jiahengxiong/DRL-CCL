from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime


_TS_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass
class DaySegment:
    day: str                 # "YYYY-mm-dd"
    x: np.ndarray            # (T, 1)  float32, scaled


def _parse_by_day(path: str) -> List[Tuple[str, np.ndarray]]:
    """
    Parse file into per-day raw sequences (no scaling).
    Returns list of (day, x_raw) where x_raw is (T,) float32.
    """
    bucket: Dict[str, List[Tuple[str, float]]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or "," not in line:
                continue

            ts, v = line.split(",", 1)
            ts = ts.strip()
            v = v.strip()

            if len(ts) < 19:
                continue
            ts = ts[:19]

            try:
                datetime.strptime(ts, _TS_FMT)
                val = float(v)
            except ValueError:
                continue

            day = ts[:10]
            bucket.setdefault(day, []).append((ts, val))

    days = sorted(bucket.keys())
    out: List[Tuple[str, np.ndarray]] = []

    for d in days:
        rows = bucket[d]
        if len(rows) < 2:
            continue

        # sort within day
        rows.sort(key=lambda z: z[0])
        x = np.asarray([v for _, v in rows], dtype=np.float32)
        out.append((d, x))

    if len(out) == 0:
        raise ValueError(f"No valid day sequences parsed from {path}")

    return out


def _global_minmax_scale(
    seqs: List[Tuple[str, np.ndarray]],
    low: float,
    high: float,
) -> Tuple[List[DaySegment], Dict[str, float]]:
    assert high > low
    all_x = np.concatenate([x for _, x in seqs], axis=0)

    vmin = float(all_x.min())
    vmax = float(all_x.max())
    if not (vmax > vmin):
        raise ValueError("Degenerate series: global max == min")

    def scale(x: np.ndarray) -> np.ndarray:
        return low + (x - vmin) / (vmax - vmin) * (high - low)

    out: List[DaySegment] = []
    for day, x in seqs:
        xs = scale(x).astype(np.float32).reshape(-1, 1)  # (T, 1)
        out.append(DaySegment(day=day, x=xs))

    info = {"vmin": vmin, "vmax": vmax, "low": low, "high": high}
    return out, info


def _split_days(
    days: List[DaySegment],
    train_ratio: float,
) -> Tuple[List[DaySegment], List[DaySegment]]:
    """
    Split by chronological day order.
    """
    assert 0.0 < train_ratio < 1.0
    n = len(days)
    if n == 1:
        return days, []

    n_train = int(n * train_ratio)
    n_train = max(1, min(n - 1, n_train))
    return days[:n_train], days[n_train:]


def load_snvang_dataset_by_day(
    path: str,
    train_ratio: float = 0.8,
    scale_range: Tuple[float, float] = (0.01, 1.0),
) -> Dict[str, object]:
    """
    Returns:
      train_days: List[DaySegment], each x is (T,1)
      test_days:  List[DaySegment], each x is (T,1)
      scale_info: dict for inverse scaling if needed
    """
    raw = _parse_by_day(path)
    all_days, scale_info = _global_minmax_scale(
        raw, low=scale_range[0], high=scale_range[1]
    )

    train_days, test_days = _split_days(all_days, train_ratio)

    return {
        "train_days": train_days,
        "test_days": test_days,
        "scale_info": scale_info,
    }


if __name__ == "__main__":
    data = load_snvang_dataset_by_day(
        "SNVAng__STTLng.txt", train_ratio=0.8, scale_range=(0.01, 1.0)
    )
    print("train days:", len(data["train_days"]))
    print("test days:", len(data["test_days"]))
    print("first train day:", data["train_days"][0].day, data["train_days"][0].x.shape)