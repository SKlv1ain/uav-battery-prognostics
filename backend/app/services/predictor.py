"""Wrap ml/src pipeline functions into API-friendly dicts."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from typing import Any, Dict, Optional

# Make ml/ importable regardless of working directory.
_ML_ROOT = Path(__file__).parents[3] / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from src.evaluation.metrics import compute_rul, compute_soh  # noqa: E402
from src.evaluation.pipeline import (  # noqa: E402
    count_discharge_cycles,
    predict_multicycle,
    predict_singlecycle,
    run_pipeline,
    select_pipeline,
)


def _build_response(
    b_id: int,
    pipeline: str,
    n_cycles: int,
    raw: Dict[str, Any],
    eol_threshold_soh: float = 70.0,
    target_cycle: Optional[int] = None,
) -> Dict[str, Any]:
    cycle_nums = raw["cycle_nums"]
    actual = raw["actual"]
    preds = raw["preds"]

    soh: Dict[str, list] = {}
    rul: Dict[str, dict] = {}
    predictions: Dict[str, list] = {}

    rated = float(actual[0]) if len(actual) > 0 else 2.0
    eol_abs = (eol_threshold_soh / 100.0) * rated

    current_idx = 0
    if target_cycle is not None:
        idx_list = np.where(cycle_nums == target_cycle)[0]
        if len(idx_list) > 0:
            current_idx = int(idx_list[0])
        else:
            idx_list = np.where(cycle_nums <= target_cycle)[0]
            current_idx = int(idx_list[-1]) if len(idx_list) > 0 else 0

    for model_name, pred_arr in preds.items():
        predictions[model_name] = pred_arr.tolist()
        soh[model_name] = compute_soh(pred_arr, rated=rated).tolist()
        rul[model_name] = compute_rul(
            pred_arr,
            cycle_nums,
            rated=rated,
            eol_abs=eol_abs,
            current_cycle_idx=current_idx,
        )

    return {
        "battery_id": f"B00{b_id:02d}",
        "pipeline": pipeline,
        "n_cycles": n_cycles,
        "cycle_nums": cycle_nums.tolist(),
        "actual_capacity": actual.tolist(),
        "predictions": predictions,
        "soh": soh,
        "rul": rul,
    }


def run_single_cycle(
    b_id: int,
    data_dir: str,
    bundle: Dict[str, Any],
    eol_threshold_soh: float = 70.0,
    target_cycle: Optional[int] = None,
) -> Dict[str, Any]:
    raw = predict_singlecycle(b_id, data_dir, bundle)
    n_cycles = len(raw["cycle_nums"])
    return _build_response(b_id, "single_cycle", n_cycles, raw, eol_threshold_soh, target_cycle)


def run_multi_cycle(
    b_id: int,
    data_dir: str,
    bundle: Dict[str, Any],
    eol_threshold_soh: float = 70.0,
    target_cycle: Optional[int] = None,
) -> Dict[str, Any]:
    raw = predict_multicycle(b_id, data_dir, bundle)
    n_cycles = len(raw["cycle_nums"])
    return _build_response(b_id, "multi_cycle", n_cycles, raw, eol_threshold_soh, target_cycle)


def run_auto(
    b_id: int,
    data_dir: str,
    multi_bundle: Dict[str, Any],
    single_bundle: Dict[str, Any],
    window_size: int = 15,
    eol_threshold_soh: float = 70.0,
    target_cycle: Optional[int] = None,
) -> Dict[str, Any]:
    result = run_pipeline(b_id, data_dir, multi_bundle, single_bundle, window_size)
    if result is None:
        raise RuntimeError(f"Pipeline returned no result for battery {b_id}")
    n_cycles = result["n_cycles"]
    pipeline = result["pipeline"]
    raw = {k: result[k] for k in ("cycle_nums", "actual", "preds")}
    return _build_response(b_id, pipeline, n_cycles, raw, eol_threshold_soh, target_cycle)
