from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.predict import router as predict_router
from app.core.config import DATA_DIR, MULTI_MODEL_DIR, SINGLE_MODEL_DIR
from app.models.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Import here so TF loads once at startup, not on every request.
    from app.services.predictor import _ML_ROOT  # noqa: F401 — ensures sys.path patched
    from src.evaluation.pipeline import load_bundle  # type: ignore[import]

    app.state.data_dir = DATA_DIR
    app.state.single_bundle = load_bundle(SINGLE_MODEL_DIR)
    app.state.multi_bundle = load_bundle(MULTI_MODEL_DIR)
    yield


app = FastAPI(
    title="UAV Battery Prognostics API",
    version="1.0.0",
    description=(
        "REST API exposing the dual-pipeline (Single-Cycle BiLSTM + Multi-Cycle BiLSTM) "
        "battery State-of-Health and Remaining Useful Life prediction system."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api")


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health():
    return {
        "status": "ok",
        "models_loaded": {
            "single_cycle": app.state.single_bundle is not None,
            "multi_cycle": app.state.multi_bundle is not None,
        },
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
