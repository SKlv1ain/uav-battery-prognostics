"""
Loaders for the NASA PCoE Battery Aging dataset (B0005-B0018 .mat files).

Two extraction modes:
  - Multi-cycle summarised features (one row per discharge cycle).
  - Single-cycle raw curves (V/I/T resampled to a fixed length per cycle).
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io


DEFAULT_CHANNELS: Tuple[str, ...] = (
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
)


def resample_1d(arr: np.ndarray, n: int) -> np.ndarray:
    """Linearly interpolate a 1-D array to exactly n equally-spaced points."""
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, arr).astype(np.float32)


def _battery_key(b_id: int) -> str:
    return f"B00{b_id:02d}"


def _mat_path(data_dir: str, b_id: int) -> str:
    return os.path.join(data_dir, f"{_battery_key(b_id)}.mat")


def count_discharge_cycles(b_id: int, data_dir: str) -> int:
    """Fast discharge-cycle count for a battery — no full feature extraction."""
    key = _battery_key(b_id)
    mat = scipy.io.loadmat(_mat_path(data_dir, b_id))
    return sum(
        1
        for c in mat[key][0, 0]["cycle"][0]
        if c["type"][0] == "discharge"
        and "Capacity" in c["data"][0, 0].dtype.names
    )


def build_multicycle_features(b_id: int, data_dir: str) -> pd.DataFrame:
    """
    Extract per-discharge-cycle summary features for one battery.

    Returns a DataFrame sorted by cycle with columns:
        battery_id, cycle, capacity, duration, avg_temp, v_drop,
        v_decay_rate, temp_stability, cumulative_duration, capacity_fade
    """
    key = _battery_key(b_id)
    mat = scipy.io.loadmat(_mat_path(data_dir, b_id))
    cycles = mat[key][0, 0]["cycle"][0]
    rows = []

    for i, c in enumerate(cycles):
        if c["type"][0] != "discharge":
            continue
        cd = c["data"][0, 0]
        try:
            cap = float(cd["Capacity"][0][0])
        except (KeyError, IndexError, TypeError):
            continue
        t = cd["Time"][0]
        v = cd["Voltage_measured"][0]
        T = cd["Temperature_measured"][0]
        rows.append(
            {
                "battery_id": key,
                "cycle": i + 1,
                "capacity": cap,
                "duration": float(t[-1] - t[0]),
                "avg_temp": float(np.mean(T)),
                "v_drop": float(v[0] - v[-1]),
            }
        )

    df = pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)
    df["v_decay_rate"] = df["v_drop"] / df["duration"]
    df["temp_stability"] = df["avg_temp"].rolling(3).std().fillna(0)
    df["cumulative_duration"] = df["duration"].cumsum()
    df["capacity_fade"] = df["capacity"].diff().fillna(0)
    return df


def build_multicycle_dataset(
    battery_ids: Sequence[int], data_dir: str
) -> pd.DataFrame:
    """Concatenate `build_multicycle_features` across multiple batteries."""
    frames = [build_multicycle_features(b, data_dir) for b in battery_ids]
    return pd.concat(frames, ignore_index=True)


def load_single_cycle_dataset(
    battery_ids: Sequence[int],
    data_dir: str,
    n_timesteps: int = 1000,
    channels: Sequence[str] = DEFAULT_CHANNELS,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract every discharge cycle as a raw resampled curve.

    Returns:
        X    — (n_cycles, n_timesteps, len(channels)) float32
        y    — (n_cycles,)  capacity in Ahr
        meta — DataFrame with battery_id, cycle_idx
    """
    X_list, y_list, meta_rows = [], [], []

    for b_id in battery_ids:
        key = _battery_key(b_id)
        mat = scipy.io.loadmat(_mat_path(data_dir, b_id))
        cycles = mat[key][0, 0]["cycle"][0]

        for i, cyc in enumerate(cycles):
            if cyc["type"][0] != "discharge":
                continue
            cd = cyc["data"][0, 0]
            try:
                cap = float(cd["Capacity"][0][0])
            except (KeyError, IndexError, TypeError):
                continue

            channel_arrs: List[np.ndarray] = []
            ok = True
            for ch in channels:
                try:
                    raw = cd[ch][0].astype(np.float32)
                    if len(raw) < 10:
                        ok = False
                        break
                    channel_arrs.append(resample_1d(raw, n_timesteps))
                except (KeyError, IndexError):
                    ok = False
                    break
            if not ok:
                continue

            X_list.append(np.stack(channel_arrs, axis=-1))
            y_list.append(cap)
            meta_rows.append({"battery_id": key, "cycle_idx": i + 1})

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


def load_raw_cycles(
    b_id: int,
    data_dir: str,
    n_timesteps: int = 1000,
    channels: Sequence[str] = DEFAULT_CHANNELS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-battery variant of `load_single_cycle_dataset`."""
    X, y, meta = load_single_cycle_dataset(
        [b_id], data_dir, n_timesteps=n_timesteps, channels=channels
    )
    cycle_nums = meta["cycle_idx"].to_numpy(dtype=np.int32)
    return X, y, cycle_nums
