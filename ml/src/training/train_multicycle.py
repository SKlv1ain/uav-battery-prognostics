"""
End-to-end training for the multi-cycle (windowed BiLSTM + LR + XGBoost) stack.

Replicates the proof-of-concept notebook:
    - Build summarised features for B0005..B0018.
    - Hold out one battery, train LR / XGBoost on flat features and a stacked
      BiLSTM on a sliding window of `window_size` cycles.
    - Save artifacts under `save_dir` so the routing pipeline can load them.

Run as a module:
    python -m src.training.train_multicycle --config configs/multi_cycle_config.yaml
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
    build_multicycle_bilstm,
    build_xgboost,
)
from ..preprocessing import build_multicycle_dataset, create_sequences


def _load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def train_multicycle(config: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = config["data"]
    train_cfg = config["training"]
    arch_cfg = config["architecture"]

    data_dir: str = data_cfg["data_dir"]
    battery_ids: List[int] = data_cfg["battery_ids"]
    train_keys: List[str] = data_cfg["train_batteries"]
    test_key: str = data_cfg["test_battery"]
    features: List[str] = data_cfg["features"]
    target: str = data_cfg.get("target", "capacity")
    window_size: int = arch_cfg["window_size"]

    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    df = build_multicycle_dataset(battery_ids, data_dir).sort_values(
        ["battery_id", "cycle"]
    )

    train_df = df[df["battery_id"].isin(train_keys)].copy()
    test_df = df[df["battery_id"] == test_key].copy()

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(train_df[features])
    y_train = scaler_y.fit_transform(train_df[[target]])
    X_test = scaler_X.transform(test_df[features])
    y_test_scaled = scaler_y.transform(test_df[[target]])

    # Baselines on flat features.
    lr_model = build_linear_regression()
    lr_model.fit(X_train, y_train)

    xgb_kwargs = config.get("xgboost", {})
    xgb_model = build_xgboost(**xgb_kwargs)
    xgb_model.fit(X_train, y_train.ravel())

    # BiLSTM on windowed sequences.
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, window_size)
    X_test_seq, _ = create_sequences(X_test, y_test_scaled, window_size)

    lstm_model = build_multicycle_bilstm(
        window_size=window_size,
        n_features=len(features),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        units=tuple(arch_cfg.get("units", [128, 64, 32])),
        dropout=arch_cfg.get("dropout", 0.2),
    )

    from tensorflow.keras.callbacks import EarlyStopping

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=train_cfg.get("patience", 15),
        restore_best_weights=True,
    )

    lstm_model.fit(
        X_train_seq,
        y_train_seq.reshape(-1, 1),
        epochs=train_cfg.get("epochs", 1000),
        validation_split=train_cfg.get("validation_split", 0.2),
        callbacks=[early_stop],
        verbose=train_cfg.get("verbose", 1),
    )

    lstm_model.save(save_dir / "lstm_model.keras")
    joblib.dump(lr_model, save_dir / "lr_model.pkl")
    joblib.dump(xgb_model, save_dir / "xgb_model.pkl")
    joblib.dump(scaler_X, save_dir / "scaler_X.pkl")
    joblib.dump(scaler_y, save_dir / "scaler_y.pkl")

    bundle_config = {"features": features, "window_size": window_size}
    with open(save_dir / "config.json", "w") as f:
        json.dump(bundle_config, f)

    return {
        "save_dir": str(save_dir),
        "n_train": int(len(X_train_seq)),
        "n_test": int(len(X_test_seq)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join("configs", "multi_cycle_config.yaml"),
    )
    args = parser.parse_args()
    info = train_multicycle(_load_config(args.config))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
