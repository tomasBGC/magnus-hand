# LoadScale

Serial interface and diagnostics for an HX711-based load cell, with a Python host library and an Arduino-compatible microcontroller firmware.

## Overview

This project provides two pieces that work together over a USB-serial link:

| Component | File(s) | Role |
|---|---|---|
| **MCU firmware** | `MCU/src/main.cpp` | Reads the HX711 ADC, streams mass values, responds to commands |
| **Python library** | `load_scale.py` | Auto-connects, reads force, sends tare/calibration commands |
| **Python GUI diagnostics** | `load_cell_check.py` | Live weight display, tare, manual & computed calibration |

## Hardware

- **Microcontroller:** ATmega328P (e.g. Arduino Nano) — configurable via `platformio.ini`
- **ADC:** HX711 24-bit load-cell amplifier
- **Load cell:** Standard strain-gauge load cell (4-wire)

### Wiring

| HX711 | MCU Pin |
|-------|---------|
| DOUT  | D2      |
| SCK   | D3      |
| VCC   | 5V      |
| GND   | GND     |

## Serial Protocol

The MCU firmware uses a simple text-based protocol at **57600 baud**.

### Handshake (startup)

1. MCU prints the device code `LDSCL-001` every 500 ms.
2. Host sends any single byte (the Python library sends `1\n`).
3. MCU stops announcing and begins streaming data.

### Commands (single characters sent over serial)

| Char | Action |
|------|--------|
| `t`  | Tare (zero the scale) |
| `r`  | Full interactive calibration wizard |
| `c`  | Set a new calibration factor manually |

After `c`, the MCU expects a float value followed by `y` to save to EEPROM (or `n` to discard).

### Data stream

Once connected, the MCU continuously streams one floating-point line per reading (mass in grams). The Python `read_force()` method drains the buffer and returns the most recent value.

## Python Setup

```bash
# Create a virtual environment and install (using uv)
uv sync
```

Minimal dependencies: only **pyserial** (`tkinter` is used by the GUI tool and ships with Python).

## Python Usage

### Library (`load_scale.py`)

```python
from load_scale import LoadScaleReader

reader = LoadScaleReader()           # auto-discovers the serial port
reader.tare()                        # zero the scale
value = reader.read_force()          # read latest mass in grams
print(f"Force: {value:.2f} g")

reader.set_calibration_factor(696.0) # set & save calibration factor
reader.close()
```

For background data collection use `loadscalegrab()` which runs in a subprocess.
Here's a complete example that logs force readings to a timestamped CSV:

```python
"""Collect force readings to a timestamped CSV file."""
import csv
import multiprocessing
import queue
from datetime import datetime
from pathlib import Path

from load_scale import loadscalegrab

out_path = Path(f"force_data_{datetime.now():%Y%m%d_%H%M%S}.csv")

stop = multiprocessing.Event()
q = multiprocessing.Queue()

proc = multiprocessing.Process(target=loadscalegrab, args=(stop, q))
proc.start()

print(f"Logging to {out_path} — press Ctrl+C to stop.")
with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "force_g"])
    try:
        while True:
            try:
                data = q.get(timeout=0.5)
            except queue.Empty:
                continue
            writer.writerow([data["timestamp"], data["force"]])
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        proc.join(timeout=2)
        print(f"Done. {out_path.stat().st_size} bytes written.")
```

### GUI Diagnostics (`load_cell_check.py`)

```bash
python load_cell_check.py              # auto-connect
python load_cell_check.py --port COM3  # prefer a specific port
```

The GUI shows:
- Live weight reading with min/max tracking
- Inferred calibration factor (from MCU boot message)
- Estimated raw HX711 delta counts
- **Tare** button
- **Set calibration factor** manually
- **Compute + Apply** — place a known mass, click, and it calculates the correct factor

## MCU Firmware

Uses [PlatformIO](https://platformio.org/) with the [HX711_ADC](https://github.com/olkal/HX711_ADC) library.

### Build & Upload

```bash
cd LoadScale
platformio run          # compile
platformio run --target upload   # compile & flash
```

### Configuration

- **Pins:** `HX711_dout = 4`, `HX711_sck = 5` (edit in `main.cpp`)
- **Baud rate:** 57600
- **EEPROM:** Calibration factor is persisted at EEPROM address 0
- **LED:** Built-in LED lights up when mass exceeds 500 g

## Calibration

### Quick method (GUI)

1. Open `load_cell_check.py`
2. Place a known mass on the load cell
3. Enter the mass in "Known mass (g)" and click **Compute + Apply**

### Manual method (serial)

1. Connect with any serial terminal (57600 baud)
2. Send `r` to start the interactive calibration wizard
3. Follow the prompts: tare → place known mass → enter weight → save to EEPROM

## Project Structure

```
LoadScale/
├── load_scale.py              Python library (LoadScaleReader + worker)
├── load_cell_check.py         Tkinter GUI diagnostics tool
├── pyproject.toml             Python project metadata & deps
├── README.md
└── MCU/
    ├── platformio.ini         PlatformIO build config
    └── src/
        └── main.cpp           MCU firmware
```
