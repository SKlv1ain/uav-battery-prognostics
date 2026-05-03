"""Sliding-window utilities for sequence models."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def create_sequences(
    data: np.ndarray, target: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) pairs from a time-ordered feature matrix.

    For each i in [0, len(data) - window) emit:
        X[i] = data[i : i + window]
        y[i] = target[i + window]
    """
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(target[i + window])
    return np.asarray(X), np.asarray(y)
