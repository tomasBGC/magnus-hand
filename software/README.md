# eFlesh Sensor System

End-to-end pipeline for the eFlesh tactile fingertip: collect paired magnetic +
force sensor data, train a model that maps raw sensor readings to contact
Force/Theta/Phi, then run that model live to drive the AmazingHand.

| Stage | Hardware | Python Interface |
|-------|----------|-------------------|
| **1. Magnetic sensing** — 4×5 MLX90393 array via TCA9546A mux | Teensy 4.0 | [`MagneticSensors/`](MagneticSensors/README.md) |
| **2. Force reference** — HX711 load cell | ATmega328P | [`LoadScale/`](LoadScale/README.md) |
| **3. Data collection** — both sensors together, saved to `recordings/` | — | `unified_controller.py` + `unified_gui.py` |
| **4. Model training + live inference** — sensor readings → Force/Theta/Phi | — | [`model_v1/`](model_v1/README.md) |
| **5. Hand control** — servo-driven fingers, torque/position | Feetech SCS0009 | [`AmazingHand/`](AmazingHand/README.md) |

`unified_gui.py`/`unified_controller.py` (root, stage 3) are the *data-collection*
tools — not to be confused with `model_v1/sensing_gui.py` or
`model_v1/hand_control_gui.py` (stage 4/5), the *live hand-control* tools. See
[`model_v1/README.md`](model_v1/README.md) for which script to actually run.

> **Note on this repository's layout.** This README describes the original
> working folder (`Sensor System/`). In this repository that folder's code is
> under `software/`, the firmware under `../firmware/`, the models under
> `../models/` and the datasets under `../data/` (raw recordings, `recordings/<session>/`,
> are not included). Paths below use the original layout.

## Directory Structure

```
Sensor System/
├── README.md                   ← this file
├── pyproject.toml              ← deps for the data-collection tool
├── unified_controller.py       ← backend: connects both sensors, saves JSONs
├── unified_gui.py              ← GUI for recording with metadata (stage 3)
├── recording_analysis.py       ← auto-runs after each recording (plots/CSVs), not a standalone tool
│
├── MagneticSensors/            ← magnetic sensor array sub-project (stage 1)
│   ├── README.md
│   ├── pyproject.toml
│   ├── eflesh_controller.py    ← hardware control layer
│   ├── eflesh_gui.py           ← real-time magnetic visualizer
│   └── MCU/                    ← Teensy 4.0 firmware
│       ├── platformio.ini
│       ├── src/main.cpp
│       └── lib/
│
├── LoadScale/                  ← force sensor sub-project (stage 2)
│   ├── README.md
│   ├── pyproject.toml
│   ├── load_scale.py           ← hardware control library
│   ├── load_cell_check.py      ← diagnostics GUI (tare, calibrate)
│   └── MCU/                    ← ATmega328P firmware
│       ├── platformio.ini
│       └── src/main.cpp
│
├── recordings/                 ← raw + processed recording data (stage 3 output)
│   ├── master_characterisation*.csv   ← raw source data, one row per sample
│   └── <recording_name>/*.json        ← one folder per recording session
│
├── model_v1/                   ← model training + live inference (stage 4)
│   └── README.md                      ← see this for what's live vs. experiment history
│
└── AmazingHand/                ← servo/hand control (stage 5)
    └── README.md
```

## Quick Start — Unified Data Collection

### Install

```bash
cd eflesh_multiplexed
uv sync
```

### Run the unified GUI

```bash
uv run unified_gui.py --output-dir ./recordings
```

The GUI lets you:

1. **Connect** to both sensors — auto-discover or specify ports via
   `--mag-port` / `--load-port`
2. **Set metadata** — θ (theta) and φ (phi) angles, plus an optional
   free-text label
3. **Record** — captures a snapshot on every magnetic frame; LoadScale force is
   polled at the same cadence, so force and magnetic data are paired
4. **Save** — JSON files are written to the output directory with timestamped
   filenames (e.g. `recording_t45_p30_20260618_143052.json`)

### JSON Output Format

> **Note:** The `sensors` booleans reflect the *actual connection state* at save
> time.  Because recording requires both sensors to be connected, these will
> always be `true` in any saved file.

```json
{
  "recording_info": {
    "total_snapshots": 1420,
    "recorded_at_iso": "2026-06-18T14:30:52.123456+00:00",
    "sensors": { "magnetic": true, "loadscale": true }
  },
  "metadata": { "theta": 45.0, "phi": 30.0, "label": "example run" },
  "snapshots": [
    {
      "timestamp": 1720000000.123,
      "timestamp_iso": "2026-06-18T14:30:50.123456+00:00",
      "force_g": 123.4,
      "groups": [
        {
          "group_index": 0,
          "sensors": [
            { "x": 1.2, "y": -3.4, "z": 5.6, "magnitude": 6.7 },
            { "x": 0.0, "y": 0.0, "z": 0.0, "magnitude": 0.0 },
            ...
          ]
        },
        ...
      ],
      "metadata": { "theta": 45.0, "phi": 30.0 }
    },
    ...
  ]
}
```

### CLI usage (headless)

```bash
# Auto-discover both sensors
uv run unified_controller.py -o ./data --theta 45 --phi 30 --duration 10

# Specify ports explicitly
uv run unified_controller.py -o ./data --mag-port COM3 --load-port COM5 --duration 5
```

Press **Ctrl+C** at any time to stop recording early — accumulated
snapshots are saved automatically before exit.  The `--duration` acts as
a maximum; if you don't interrupt, recording stops automatically after
that many seconds.

## After Data Collection: Training a Model & Running the Hand

Once you have recordings in `recordings/` (or are using the ones already
there), the rest of the pipeline lives in `model_v1/` and `AmazingHand/`:

```bash
# Live sensing only, no GUI, no servos — sanity-check the model:
uv run model_v1/live_predictor_test.py

# Live sensing GUI (4-finger monitor, no servo control):
uv run model_v1/sensing_gui.py

# Full demo — live sensing + servo control:
uv run model_v1/hand_control_gui.py
```

See **[model_v1/README.md](model_v1/README.md)** for how the model is built from
`recordings/` and which files are "live" vs. superseded training experiments,
and **[AmazingHand/README.md](AmazingHand/README.md)** for the servo control
layer and hardware safety notes.

## Individual Sensor Tools

See sub-project READMEs for details:

- **[MagneticSensors/README.md](MagneticSensors/README.md)** — firmware upload,
  handshake protocol, binary frame format, real-time visualizer
- **[LoadScale/README.md](LoadScale/README.md)** — force sensor calibration,
  diagnostics GUI, subprocess data collection
