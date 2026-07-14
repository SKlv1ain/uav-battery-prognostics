# UAV Lithium-Ion Battery Health Monitoring and Prognostics

An AI-driven battery health monitoring and prognostics system for UAV Li-Ion batteries.  
Predicts **SOH** (State of Health), **RUL** (Remaining Useful Life), and **Multi-step Capacity Forecast** using LSTM and ensemble ML models.

## Team

- **Sai Khun Main** (Meen)
- **Peerawat Theerasakul**

**Advisor:** Asst. Prof. Dr. Supaporn Erjongmanee  
**Co-Advisor:** Dr. Chaiwat Klampol  
**Lab:** Data Analysis and Knowledge Discovery Lab (DAKDL)  
**Department:** Computer Engineering, Faculty of Engineering, Kasetsart University  
**Academic Year:** 2025

---

## Architecture

The system is organized into three layers:

| Layer | Description | Tech Stack |
|-------|-------------|------------|
| **Hardware Data Acquisition** | ESP32 + Smart BMS (Daly/JBD) via UART | C++/Arduino, PlatformIO |
| **ML Prediction (Offline)** | Feature engineering → Model inference | Python, TensorFlow/Keras, XGBoost, Scikit-learn |
| **Backend API** | Serves predictions to mobile app | FastAPI, PostgreSQL |
| **Dashboard** | Mobile app for operators | React Native or Flutter |

## Project Structure

```
uav-battery-prognostics/
├── firmware/          # ESP32 + BMS firmware (C++/Arduino)
├── ml/                # ML pipeline (Python)
│   ├── data/          # Raw, processed, hardware-collected data
│   ├── notebooks/     # EDA and experiment notebooks
│   ├── src/           # Preprocessing, models, training, evaluation
│   ├── configs/       # Hyperparameter YAML files
│   └── saved_models/  # Exported model weights
├── backend/           # FastAPI backend (REST API)
├── ui/                # Streamlit dashboard
└── docs/              # SRS, architecture diagrams, reports
```

## Datasets

| Dataset | Purpose | Cell Type |
|---------|---------|-----------|
| NASA PCoE | Primary training | 18650 cylindrical |
| Oxford Battery Degradation | Cross-dataset validation | Pouch cell |
| CALCE | Cross-dataset validation | Various |
| Self-collected (ESP32 + BMS) | Real-world validation | UAV pouch cell |

> Raw datasets are **not** included in this repo. See [`ml/data/raw/README.md`](ml/data/raw/README.md) for download instructions.

## Evaluation Strategy

Three-tier validation:
1. **Intra-dataset:** Leave-One-Battery-Out (LOBO) on NASA PCoE
2. **Inter-dataset:** Train on NASA → Test on Oxford/CALCE
3. **Real-world:** Hardware-in-the-loop (HITL) with ESP32 telemetry

## Metrics

- **RMSE** and **MAE** for RUL and Capacity Forecast
- **MAPE** for SOH estimation
- **R²** for overall model fit

## Getting Started

See individual README files in each subdirectory for setup instructions:
- [`firmware/README.md`](firmware/README.md) — Hardware setup
- [`ml/README.md`](ml/README.md) — ML pipeline setup
- [`backend/README.md`](backend/README.md) — Backend API setup
- [`ui/`](ui/) — Streamlit dashboard

## License

This project is developed as a senior project at Kasetsart University.
