# ML Pipeline

Machine learning pipeline for UAV battery health prediction.

## Setup

```bash
cd ml/
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Structure

```
ml/
├── data/
│   ├── raw/           # Original datasets (gitignored, see README)
│   ├── processed/     # Feature-engineered cycle-level data
│   └── hardware/      # ESP32-collected telemetry
├── notebooks/         # Jupyter notebooks for EDA & experiments
├── src/
│   ├── preprocessing/ # Feature engineering pipeline
│   ├── models/        # Model definitions (LSTM, XGBoost, RF, LR)
│   ├── training/      # Training scripts & configs
│   └── evaluation/    # Evaluation (LOBO, cross-dataset, HITL)
├── configs/           # Hyperparameter YAML files
└── saved_models/      # Exported model weights
```

## Models

| Model | Task | Input |
|-------|------|-------|
| LSTM | RUL + Multi-step Capacity Forecast | Sliding window of k cycles |
| XGBoost | SOH estimation | Current cycle features |
| Random Forest | SOH baseline | Current cycle features |
| Linear Regression | Performance baseline | Current cycle features |

## Experiment Tracking

We use MLflow for experiment tracking:

```bash
mlflow ui  # Open http://localhost:5000
```
