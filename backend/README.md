# Backend — FastAPI

REST API serving battery health predictions (SOH, RUL, Capacity Forecast) to the mobile dashboard.

## Setup

```bash
cd backend/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/batteries` | List registered batteries |
| GET | `/api/batteries/{id}/predictions` | Get SOH, RUL, capacity forecast |
| POST | `/api/batteries/{id}/telemetry` | Ingest new telemetry data |

## Database

PostgreSQL (or SQLite for development). Run migrations:

```bash
alembic upgrade head
```
