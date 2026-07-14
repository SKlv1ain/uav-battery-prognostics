# Mobile Dashboard

Unified mobile application for UAV battery health monitoring.

## Screens

1. **Fleet Overview** — All registered batteries with health status
2. **SOH Gauge** — Current State of Health percentage
3. **RUL Countdown** — Remaining cycles until EOL (80%)
4. **Capacity Forecast** — Multi-step capacity trend chart (next 10 cycles)
5. **Historical Logs** — Raw telemetry viewer (voltage, current, temperature)

## Setup

### React Native
```bash
cd mobile/
npm install
npx react-native run-android  # or run-ios
```

### Flutter (alternative)
```bash
cd mobile/
flutter pub get
flutter run
```

## Tech Stack

- React Native or Flutter (Dart)
- Recharts / Victory Native for charts
- Axios / Dio for API communication
