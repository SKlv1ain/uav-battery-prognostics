# Firmware — ESP32 + Smart BMS

ESP32 firmware for reading battery telemetry from Smart BMS (Daly/JBD) via UART.

## Hardware

- **MCU:** ESP32 DevKit
- **BMS:** Daly Smart BMS (UART @ 9600 bps)
- **Sensors:** INA226 (voltage/current), DS18B20 (temperature)
- **Storage:** SD Card module
- **Power:** KIS3R33S buck converter
- **Shunt:** 50A shunt resistor

## Setup

This project uses [PlatformIO](https://platformio.org/).

```bash
cd firmware/
pio run           # Build
pio run -t upload # Flash to ESP32
pio device monitor # Serial monitor
```

## Data Output

The firmware outputs CSV records with the following fields:
`timestamp, cycle, cell_v1, cell_v2, ..., pack_voltage, current, temperature, soc`
