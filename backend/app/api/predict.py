from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import (
    AutoPredictRequest,
    MultiCycleRequest,
    PredictionResponse,
    SingleCycleRequest,
)
from app.services.predictor import run_auto, run_multi_cycle, run_single_cycle

router = APIRouter(tags=["Predictions"])


@router.post("/predict/single-cycle", response_model=PredictionResponse)
def predict_single_cycle(body: SingleCycleRequest, request: Request):
    """Run the single-cycle pipeline on raw V/I/T curves from one discharge cycle."""
    bundle = request.app.state.single_bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="Single-cycle model not loaded")
    try:
        return run_single_cycle(body.battery_id, request.app.state.data_dir, bundle)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/predict/multi-cycle", response_model=PredictionResponse)
def predict_multi_cycle(body: MultiCycleRequest, request: Request):
    """Run the multi-cycle pipeline on windowed cycle-summary features."""
    bundle = request.app.state.multi_bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="Multi-cycle model not loaded")
    try:
        return run_multi_cycle(body.battery_id, request.app.state.data_dir, bundle)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/predict/auto", response_model=PredictionResponse)
def predict_auto(body: AutoPredictRequest, request: Request):
    """Auto-route: uses multi-cycle if ≥ window_size cycles exist, else single-cycle."""
    if request.app.state.multi_bundle is None and request.app.state.single_bundle is None:
        raise HTTPException(status_code=503, detail="No models loaded")
    try:
        return run_auto(
            body.battery_id,
            request.app.state.data_dir,
            request.app.state.multi_bundle,
            request.app.state.single_bundle,
            body.window_size,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
