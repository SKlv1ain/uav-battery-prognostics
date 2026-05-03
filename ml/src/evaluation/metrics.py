"""SOH and RUL helpers shared across the prediction pipeline and notebooks."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# NASA absolute end-of-life threshold (Ahr).
EOL_ABS = 1.4
# Nominal rated capacity used by the proof-of-concept (Ahr).
RATED_CAPACITY_NOMINAL = 2.0
EOL_THRESHOLD = EOL_ABS / RATED_CAPACITY_NOMINAL  # 0.70 → 70 % SOH


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    n = min(len(y_true), len(y_pred))
    yt, yp = y_true[:n], y_pred[:n]
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
    }


def compute_soh(
    capacity_preds: np.ndarray, rated: float = RATED_CAPACITY_NOMINAL
) -> np.ndarray:
    """Convert capacity (Ahr) to State of Health (%) given a rated capacity."""
    return np.asarray(capacity_preds, dtype=float) / rated * 100.0


def compute_rul(
    capacity_preds: np.ndarray,
    cycle_nums: np.ndarray,
    rated: float = RATED_CAPACITY_NOMINAL,
    eol_abs: float = EOL_ABS,
    current_cycle_idx: int = 0,
) -> Dict[str, Optional[float]]:
    """
    Remaining Useful Life from a capacity-prediction series.

    EOL is the first index (>= current_cycle_idx) where capacity drops below
    the absolute threshold (default 1.4 Ahr). Returns RUL in cycles, the EOL
    index, and the SOH at the current cycle.
    """
    preds = np.asarray(capacity_preds, dtype=float)
    soh = preds / rated * 100.0

    below = np.where(preds <= eol_abs)[0]
    below = below[below >= current_cycle_idx]

    soh_now = float(soh[current_cycle_idx])

    if len(below) == 0:
        return {
            "rul_cycles": None,
            "eol_index": None,
            "soh_now": soh_now,
            "note": "EOL not reached within window",
        }

    eol_idx = int(below[0])
    rul_cycles = int(cycle_nums[eol_idx] - cycle_nums[current_cycle_idx])
    return {
        "rul_cycles": rul_cycles,
        "eol_index": eol_idx,
        "soh_now": soh_now,
        "note": "EOL reached within prediction window",
    }
