import os
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[3]

DATA_DIR = os.getenv(
    "DATA_DIR",
    str(_REPO_ROOT / "ml" / "data" / "raw" / "BatteryAgingARC-FY08Q4"),
)
SINGLE_MODEL_DIR = os.getenv(
    "SINGLE_MODEL_DIR",
    str(_REPO_ROOT / "ml" / "saved_models" / "single_cycle"),
)
MULTI_MODEL_DIR = os.getenv(
    "MULTI_MODEL_DIR",
    str(_REPO_ROOT / "ml" / "saved_models" / "multi_cycle"),
)
ML_SRC_PATH = str(_REPO_ROOT / "ml")
