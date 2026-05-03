"""
Auto-routing inference pipeline.

Given a battery's discharge history, route to:
    - multi_cycle  → windowed BiLSTM + LR + XGBoost on summarised features
    - single_cycle → BiLSTM + LR + XGBoost on raw V/I/T curves

Routing rule: use multi_cycle if at least `window_size` discharge cycles are
available, otherwise fall back to single_cycle (which works on a single cycle).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import joblib
import numpy as np

from ..preprocessing import (
    apply_channel_scalers,
    build_multicycle_features,
    count_discharge_cycles,
    load_raw_cycles,
)


def load_bundle(save_dir: str) -> Optional[Dict[str, Any]]:
    """Load every artifact produced by a training run, or None if missing."""
    config_path = os.path.join(save_dir, "config.json")
    if not os.path.exists(config_path):
        return None

    with open(config_path) as f:
        bundle: Dict[str, Any] = {"config": json.load(f)}

    from tensorflow.keras.models import load_model

    for name in ("lstm_model.keras", "bilstm_model.keras", "cnn_lstm_model.keras"):
        path = os.path.join(save_dir, name)
        if os.path.exists(path):
            bundle["lstm_model"] = load_model(path)
            break

    for fname, key in (
        ("lr_model.pkl", "lr_model"),
        ("xgb_model.pkl", "xgb_model"),
        ("scaler_X.pkl", "scaler_X"),
        ("scaler_y.pkl", "scaler_y"),
        ("channel_scalers.pkl", "channel_scalers"),
    ):
        path = os.path.join(save_dir, fname)
        if os.path.exists(path):
            bundle[key] = joblib.load(path)

    return bundle


def select_pipeline(n_cycles: int, window_size: int) -> str:
    return "multi_cycle" if n_cycles >= window_size else "single_cycle"


def predict_multicycle(
    b_id: int, data_dir: str, bundle: Dict[str, Any]
) -> Dict[str, Any]:
    """Summarised features + sliding-window BiLSTM (with LR / XGBoost)."""
    df = build_multicycle_features(b_id, data_dir)
    feats = bundle["config"]["features"]
    W = bundle["config"]["window_size"]

    Xs = bundle["scaler_X"].transform(df[feats])
    cycle_nums = df["cycle"].to_numpy()
    actual = df["capacity"].to_numpy()

    lr_preds = bundle["scaler_y"].inverse_transform(
        bundle["lr_model"].predict(Xs).reshape(-1, 1)
    ).ravel()
    xgb_preds = bundle["scaler_y"].inverse_transform(
        bundle["xgb_model"].predict(Xs).reshape(-1, 1)
    ).ravel()

    if "lstm_model" in bundle and len(Xs) >= W:
        windows = np.stack([Xs[i : i + W] for i in range(len(Xs) - W)])
        lstm_sc = bundle["lstm_model"].predict(windows, verbose=0).ravel()
        lstm_raw = bundle["scaler_y"].inverse_transform(
            lstm_sc.reshape(-1, 1)
        ).ravel()
        # XGB fills the first W cycles where the LSTM has no window.
        lstm_hybrid = np.concatenate([xgb_preds[:W], lstm_raw])
        # Anchor to the first actual to remove systematic offset.
        lstm_hybrid += actual[0] - lstm_hybrid[0]
    else:
        lstm_hybrid = xgb_preds

    return {
        "cycle_nums": cycle_nums,
        "actual": actual,
        "preds": {
            "Linear Regression": lr_preds,
            "XGBoost": xgb_preds,
            "BiLSTM Hybrid": lstm_hybrid,
        },
    }


def predict_singlecycle(
    b_id: int, data_dir: str, bundle: Dict[str, Any]
) -> Dict[str, Any]:
    """Raw V/I/T curves → LR / XGBoost / BiLSTM (no window required)."""
    n_ts = bundle["config"].get("n_timesteps", 1000)
    channels = bundle["config"].get(
        "channels",
        ["Voltage_measured", "Current_measured", "Temperature_measured"],
    )

    X_raw, actual, cycle_nums = load_raw_cycles(
        b_id, data_dir, n_timesteps=n_ts, channels=channels
    )
    X_scaled = apply_channel_scalers(X_raw, bundle["channel_scalers"])
    X_flat = X_scaled.reshape(len(X_scaled), -1)

    lr_preds = bundle["scaler_y"].inverse_transform(
        bundle["lr_model"].predict(X_flat).reshape(-1, 1)
    ).ravel()
    xgb_preds = bundle["scaler_y"].inverse_transform(
        bundle["xgb_model"].predict(X_flat).reshape(-1, 1)
    ).ravel()

    if "lstm_model" in bundle:
        bilstm_sc = bundle["lstm_model"].predict(X_scaled, verbose=0).ravel()
        bilstm_raw = bundle["scaler_y"].inverse_transform(
            bilstm_sc.reshape(-1, 1)
        ).ravel()
        bilstm_raw += actual[0] - bilstm_raw[0]
    else:
        bilstm_raw = xgb_preds

    return {
        "cycle_nums": cycle_nums,
        "actual": actual,
        "preds": {
            "Linear Regression": lr_preds,
            "XGBoost": xgb_preds,
            "BiLSTM": bilstm_raw,
        },
    }


def run_pipeline(
    b_id: int,
    data_dir: str,
    multi_bundle: Optional[Dict[str, Any]],
    single_bundle: Optional[Dict[str, Any]],
    window_size: int = 15,
) -> Optional[Dict[str, Any]]:
    """Auto-detect cycle count → route → predict → return results."""
    key = f"B00{b_id:02d}"
    n_cycles = count_discharge_cycles(b_id, data_dir)
    pipeline = select_pipeline(n_cycles, window_size)

    if pipeline == "multi_cycle":
        if multi_bundle is None:
            return None
        out = predict_multicycle(b_id, data_dir, multi_bundle)
    else:
        if single_bundle is None:
            return None
        out = predict_singlecycle(b_id, data_dir, single_bundle)

    return {
        "battery": key,
        "pipeline": pipeline,
        "n_cycles": n_cycles,
        **out,
    }
