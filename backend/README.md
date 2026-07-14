# Backend — FastAPI

REST API serving battery health predictions (SOH, RUL, Capacity Forecast).  
Swagger documentation is auto-generated and available at `http://localhost:8000/docs`.

## Setup

```bash
cd backend/
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check + model load status |
| `POST` | `/api/predict/single-cycle` | Predict using raw V/I/T curves (single discharge cycle) |
| `POST` | `/api/predict/multi-cycle` | Predict using windowed cycle-summary features |
| `POST` | `/api/predict/auto` | Auto-route to best pipeline based on available cycle count |

## Request / Response

### `POST /api/predict/auto`

**Request:**
```json
{
  "battery_id": 5,
  "window_size": 15
}
```

**Response:**
```json
{
  "battery_id": "B0005",
  "pipeline": "multi_cycle",
  "n_cycles": 168,
  "cycle_nums": [1, 2, 3, "..."],
  "actual_capacity": [1.856, 1.847, "..."],
  "predictions": {
    "Linear Regression": [1.843, "..."],
    "XGBoost": [1.852, "..."],
    "BiLSTM Hybrid": [1.849, "..."]
  },
  "soh": {
    "BiLSTM Hybrid": [92.4, "..."]
  },
  "rul": {
    "BiLSTM Hybrid": {
      "rul_cycles": 48,
      "eol_index": 120,
      "soh_now": 88.3,
      "note": "EOL reached within prediction window"
    }
  }
}
```

`battery_id` maps to NASA PCoE batteries: `5 → B0005`, `6 → B0006`, `7 → B0007`, `18 → B0018`.

## Routing Logic

```
n_cycles >= window_size  →  multi_cycle pipeline (BiLSTM Hybrid + XGBoost + LR)
n_cycles <  window_size  →  single_cycle pipeline (BiLSTM + XGBoost + LR)
```
