from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SingleCycleRequest(BaseModel):
    battery_id: int = Field(..., description="Battery ID number (e.g. 5 → B0005)", examples=[5])


class MultiCycleRequest(BaseModel):
    battery_id: int = Field(..., description="Battery ID number (e.g. 5 → B0005)", examples=[5])


class AutoPredictRequest(BaseModel):
    battery_id: int = Field(..., description="Battery ID number (e.g. 5 → B0005)", examples=[5])
    window_size: int = Field(15, description="Minimum cycles required to use multi-cycle pipeline")


class RULResult(BaseModel):
    rul_cycles: Optional[int] = Field(None, description="Remaining useful life in cycles")
    eol_index: Optional[int] = Field(None, description="Index in prediction array where EOL occurs")
    soh_now: float = Field(..., description="State of Health at current cycle (%)")
    note: str = Field(..., description="Human-readable RUL note")


class PredictionResponse(BaseModel):
    battery_id: str = Field(..., description="Battery label e.g. 'B0005'")
    pipeline: str = Field(..., description="Pipeline used: 'single_cycle' or 'multi_cycle'")
    n_cycles: int = Field(..., description="Total number of discharge cycles")
    cycle_nums: List[int] = Field(..., description="Cycle index numbers")
    actual_capacity: List[float] = Field(..., description="Measured capacity (Ahr) per cycle")
    predictions: Dict[str, List[float]] = Field(
        ..., description="Predicted capacity (Ahr) per cycle, keyed by model name"
    )
    soh: Dict[str, List[float]] = Field(
        ..., description="State of Health (%) per cycle, keyed by model name"
    )
    rul: Dict[str, RULResult] = Field(
        ..., description="RUL result per model"
    )


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
