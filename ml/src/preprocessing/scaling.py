"""Per-channel scaling for raw discharge curves."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def fit_channel_scalers(X: np.ndarray) -> List[MinMaxScaler]:
    """Fit one MinMaxScaler per channel on X with shape (N, T, C)."""
    scalers: List[MinMaxScaler] = []
    n_ts, n_ch = X.shape[1], X.shape[2]
    for ch in range(n_ch):
        sc = MinMaxScaler()
        sc.fit(X[:, :, ch].reshape(-1, 1))
        scalers.append(sc)
    return scalers


def apply_channel_scalers(
    X: np.ndarray, scalers: List[MinMaxScaler]
) -> np.ndarray:
    """Apply pre-fit per-channel scalers to X with shape (N, T, C)."""
    n_ts = X.shape[1]
    out = X.copy()
    for ch, sc in enumerate(scalers):
        flat = X[:, :, ch].reshape(-1, 1)
        out[:, :, ch] = sc.transform(flat).reshape(-1, n_ts)
    return out


def fit_apply_channel_scalers(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[MinMaxScaler]]:
    scalers = fit_channel_scalers(X_train)
    return (
        apply_channel_scalers(X_train, scalers),
        apply_channel_scalers(X_test, scalers),
        scalers,
    )
