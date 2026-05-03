# ML Pipeline

Machine learning pipeline for UAV battery health prediction. Refactored from
the standalone proof-of-concept notebooks into importable modules.

## Setup

```bash
cd ml/
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Structure

```
ml/
├── data/
│   ├── raw/BatteryAgingARC-FY08Q4/  # NASA .mat files (gitignored)
│   ├── processed/                   # Feature-engineered cycle-level data
│   └── hardware/                    # ESP32-collected telemetry
├── notebooks/
│   ├── 01_train_single_cycle.ipynb  # raw V/I/T curve BiLSTM
│   ├── 02_train_multi_cycle.ipynb   # windowed-features BiLSTM
│   └── 03_pipeline.ipynb            # auto-routing inference demo
├── src/
│   ├── preprocessing/   # NASA loaders, windowing, channel scaling
│   ├── models/          # BiLSTM (multi & single cycle), LR, XGBoost
│   ├── training/        # train_multicycle / train_singlecycle scripts
│   └── evaluation/      # metrics, SOH/RUL, auto-routing pipeline
├── configs/
│   ├── multi_cycle_config.yaml
│   └── single_cycle_config.yaml
└── saved_models/
    ├── multi_cycle/     # lstm + lr + xgb + scalers + config.json
    └── single_cycle/    # lstm + lr + xgb + channel_scalers + scaler_y
```

## Pipelines

| Pipeline      | Input                                       | When to use                       |
|---------------|---------------------------------------------|-----------------------------------|
| Multi-cycle   | sliding window of summarised cycle features | ≥ `window_size` (15) past cycles  |
| Single-cycle  | one resampled discharge curve (V/I/T × 1000)| < 15 cycles (fresh battery)       |

`src.evaluation.pipeline.select_pipeline` routes between them automatically
based on the number of available discharge cycles.

## Training

Each pipeline trains three models (Linear Regression, XGBoost, BiLSTM). Saved
artifacts are loaded by `src.evaluation.pipeline.load_bundle`.

```bash
cd ml/

# Multi-cycle stack (windowed BiLSTM + baselines)
python -m src.training.train_multicycle --config configs/multi_cycle_config.yaml

# Single-cycle stack (raw-curve BiLSTM + baselines)
python -m src.training.train_singlecycle --config configs/single_cycle_config.yaml
```

## Inference / evaluation

```bash
cd ml/
python -m src.evaluation.run_pipeline \
    --data-dir data/raw/BatteryAgingARC-FY08Q4 \
    --batteries 5 6 7 18
```

Model weights and `mlruns/` are gitignored — run the training commands above
once to populate `saved_models/multi_cycle/` and `saved_models/single_cycle/`
before invoking the routing pipeline.

## Experiment tracking

MLflow tracks runs to `mlflow.db` at the repo root.

```bash
mlflow ui --backend-store-uri sqlite:///../mlflow.db   # http://localhost:5000
```
