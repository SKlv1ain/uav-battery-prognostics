"""
End-to-end training for the single-cycle (raw-curve BiLSTM + LR + XGBoost) stack.

Run as a module:
    python -m src.training.train_singlecycle --config configs/single_cycle_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import yaml
from sklearn.preprocessing import MinMaxScaler

from ..models import (
    build_linear_regression,
    build_singlecycle_bilstm,
    build_xgboost,
)
from ..preprocessing import (
    DEFAULT_CHANNELS,
    fit_apply_channel_scalers,
    load_single_cycle_dataset,
)


def _load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def train_singlecycle(config: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = config["data"]
    train_cfg = config["training"]
    arch_cfg = config["architecture"]

    data_dir: str = data_cfg["data_dir"]
    battery_ids: List[int] = data_cfg["battery_ids"]
    train_keys: List[str] = data_cfg["train_batteries"]
    test_key: str = data_cfg["test_battery"]
    n_timesteps: int = arch_cfg["n_timesteps"]
    channels: List[str] = data_cfg.get("channels", list(DEFAULT_CHANNELS))

    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    X_all, y_all, meta = load_single_cycle_dataset(
        battery_ids, data_dir, n_timesteps=n_timesteps, channels=channels
    )

    train_mask = meta["battery_id"].isin(train_keys).values
    test_mask = (meta["battery_id"] == test_key).values

    X_train_raw = X_all[train_mask]
    X_test_raw = X_all[test_mask]
    y_train = y_all[train_mask]
    y_test = y_all[test_mask]

    X_train, X_test, channel_scalers = fit_apply_channel_scalers(
        X_train_raw, X_test_raw
    )

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    # Baselines on the flattened curve.
    lr_model = build_linear_regression()
    lr_model.fit(X_train_flat, y_train_scaled)

    xgb_kwargs = {"colsample_bytree": 0.3, **config.get("xgboost", {})}
    xgb_model = build_xgboost(**xgb_kwargs)
    xgb_model.fit(
        X_train_flat,
        y_train_scaled,
        eval_set=[
            (
                X_test_flat,
                scaler_y.transform(y_test.reshape(-1, 1)).ravel(),
            )
        ],
        verbose=False,
    )

    # Stacked BiLSTM on the curve directly.
    bilstm_model = build_singlecycle_bilstm(
        n_timesteps=n_timesteps,
        n_channels=len(channels),
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        units=tuple(arch_cfg.get("units", [128, 64, 32])),
        dropouts=tuple(arch_cfg.get("dropouts", [0.3, 0.3, 0.2])),
        head_units=arch_cfg.get("head_units", 32),
    )

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=train_cfg.get("patience", 12),
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    bilstm_model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=train_cfg.get("epochs", 100),
        batch_size=train_cfg.get("batch_size", 32),
        callbacks=callbacks,
        verbose=train_cfg.get("verbose", 1),
    )

    bilstm_model.save(save_dir / "lstm_model.keras")
    joblib.dump(lr_model, save_dir / "lr_model.pkl")
    joblib.dump(xgb_model, save_dir / "xgb_model.pkl")
    joblib.dump(channel_scalers, save_dir / "channel_scalers.pkl")
    joblib.dump(scaler_y, save_dir / "scaler_y.pkl")

    bundle_config = {"n_timesteps": n_timesteps, "channels": list(channels)}
    with open(save_dir / "config.json", "w") as f:
        json.dump(bundle_config, f)

    return {
        "save_dir": str(save_dir),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join("configs", "single_cycle_config.yaml"),
    )
    args = parser.parse_args()
    info = train_singlecycle(_load_config(args.config))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
