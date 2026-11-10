# eFlesh 4×5 Multiplexed Sensor Array

Reads **4 groups of 5 MLX90393 magnetometers** (20 total) through a **TCA9546A I2C multiplexer** on a **Teensy 4.0**, streaming calibrated magnetic field data to a Python GUI for real-time visualization and sensor identification.

```
┌──────────────────┐     ┌───────────────┐     ┌──────────────────────────────┐
│  4 eFlesh boards │     │  TCA9546A     │     │  Teensy 4.0                  │
│  (5 MLX90393 ea) │────▶│  4-ch I2C Mux │────▶│  src/main.cpp                │
│  = 20 sensors    │     │  addr: 0x70   │     │  + lib/handshake (auto-conn) │
└──────────────────┘     └───────────────┘     └──────────┬───────────────────┘
                                                          │ USB Serial (2 Mbps)
                                                          │ ① HSK → ② ACK → ③ RDY
                                                          ▼
                                               ┌──────────────────────┐
                                               │  eflesh_controller   │
                                               │  (hw control layer)  │
                                               │         │            │
                                               │  eflesh_gui.py       │
                                               │  (4-panel live view) │
                                               └──────────────────────┘
```

## Directory Structure

```
MagneticSensors/
├── eflesh_controller.py        # Hardware control layer (discovery + serial)
├── eflesh_gui.py               # Real-time Python visualization GUI
├── pyproject.toml               # Python dependencies (UV)
├── README.md
└── MCU/
    ├── platformio.ini           # PlatformIO config for Teensy 4.0
    ├── src/
    │   └── main.cpp             # Firmware: handshake + orchestration
    └── lib/
        ├── handshake/           # Auto-connect handshake protocol (MCU)
        ├── tca/                 # TCA9546A I2C multiplexer driver
        ├── eflesh/              # Sensor array (discovery, read pipeline, framing)
        └── MLX90393/            # MLX90393 driver library
```

## Hardware Setup

### Wiring

| TCA9546A Pin | Connect To            |
|-------------|----------------------|
| VCC         | Teensy 3.3V          |
| GND         | Teensy GND           |
| SDA         | Teensy Pin 18 (SDA)  |
| SCL         | Teensy Pin 19 (SCL)  |
| A0, A1, A2  | GND (address = 0x70) |
| SD0/SC0     | eFlesh Board 0 I2C   |
| SD1/SC1     | eFlesh Board 1 I2C   |
| SD2/SC2     | eFlesh Board 2 I2C   |
| SD3/SC3     | eFlesh Board 3 I2C   |

Each eFlesh board has 5 MLX90393 sensors at addresses 0x0C–0x10
(or 0x0C,0x10–0x13 for "white" boards).

### Teensy 4.0 Pin Reference

- **I2C (Wire)**: SDA=18, SCL=19
- **I2C (Wire1)**: SDA=17, SCL=16
- **I2C (Wire2)**: SDA=25, SCL=24

## Firmware (PlatformIO)

### Upload

Use the vscode platformio plugin to upload the code to the teensy 4.0

### Monitor (text output only, for debug)

```bash
pio device monitor --baud 2000000
```

### Handshake Protocol

Before data streaming begins, the MCU and host perform a short handshake
to ensure the correct device is connected:

```
  MCU (Teensy)                           Host (Python)
  ═══════════                            ═════════════
  Broadcast "EFLESH_HSK" every 150 ms
                                         Scan COM ports → send "EFLESH_ACK"
  Receive ACK → send "EFLESH_RDY"
                                         Receive RDY → handshake complete ✓
  Send text header + binary frames
```

Handshake strings are defined in `MCU/lib/handshake/handshake.h` (MCU) and
`eflesh_controller.py` (host).

### Serial Protocol

Binary frames, 324 bytes each:

| Offset | Size | Field              |
|--------|------|--------------------|
| 0      | 1    | Magic: 0xAA        |
| 1      | 1    | Magic: 0x55        |
| 2      | 2    | Sequence (uint16 LE)|
| 4      | 320  | 4 groups × 5 sensors × (float t, x, y, z) |

Sensor data layout: `[group0_s0][group0_s1]...[group0_s4][group1_s0]...[group3_s4]`

Each sensor is 16 bytes: `float t, float x, float y, float z` (all little-endian).

## Python GUI

### Install Dependencies (UV)

Install [UV](https://docs.astral.sh/uv/) if you haven't already:

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then sync the project dependencies:

```bash
cd eflesh_multiplexed/MagneticSensors
uv sync
```

### Run

```bash
# From the MagneticSensors directory:
uv run eflesh_gui.py

# Or from the repo root:
uv run MagneticSensors/eflesh_gui.py

# Connect to a specific port:
uv run eflesh_gui.py --port COM3
```

The GUI also has an **🔍 Auto-Connect** button that scans all COM ports for an
eFlesh device and connects automatically.

### GUI Layout

```
┌──────────────────────┬──────────────────────┐
│  Group 0 (Red)       │  Group 1 (Green)     │
│  [S0][S1][S2][S3][S4]│  [S0][S1][S2][S3][S4]│
│  X/Y/Z bar per sensor│  X/Y/Z bar per sensor│
├──────────────────────┼──────────────────────┤
│  Group 2 (Blue)      │  Group 3 (Gold)      │
│  [S0][S1][S2][S3][S4]│  [S0][S1][S2][S3][S4]│
│  X/Y/Z bar per sensor│  X/Y/Z bar per sensor│
└──────────────────────┴──────────────────────┘
```

**Identifying sensors**: Touch one physical eFlesh board — its corresponding
panel in the GUI will spike. The group color tells you which mux channel it's on.

- **FPS counter**: top-right, shows frame rate
- **Drop counter**: shows missed/dropped frames
- Each sensor shows X (red), Y (green), Z (blue) as bar heights with numeric values

## How It Works

1. **Firmware** (`MCU/src/main.cpp` + `MCU/lib/`):
   - `MCU/lib/tca/` — drives the TCA9546A 4-channel I2C multiplexer
   - `MCU/lib/eflesh/` — data model, I2C sensor discovery, single-shot read pipeline,
     and binary frame serialization
   - `MCU/lib/handshake/` — broadcasts `EFLESH_HSK` over USB serial until the host
     acknowledges
   - `MCU/src/main.cpp` — thin orchestrator: handshake → init → stream
   - Scans the I2C bus on each mux channel to find the 5 MLX90393 sensors
   - Single-shot mode only — burst is incompatible with the mux (sensors keep
     converting while electrically isolated during channel switching)
   - Uses a **pipelined read loop**: triggers all 20 sensors first, then sends
     USB data during the shared conversion wait, then reads all 20 results.
     This avoids paying the conversion delay 4 separate times.
   - Sends a text header, then streams 324-byte binary frames over USB serial

2. **Hardware Controller** (`eflesh_controller.py`):
   - Auto-discovers the device by scanning COM ports and performing the handshake
   - Reads the text header, then parses binary frames in a background thread
   - Exposes data via plain callbacks — framework-agnostic (no Qt dependency)
   - Can be reused in CLI tools, loggers, or any Python application

3. **GUI** (`eflesh_gui.py`):
   - Wraps the controller with a thin `ControllerAdapter` for Qt signals
   - Syncs to binary frames using 0xAA 0x55 magic bytes
   - Updates 4 group panels with real-time bar charts
   - Shows X, Y, Z magnetic field components per sensor
   - Tracks FPS and dropped frames

## Performance

- **Operating mode**: Single-shot, pipelined across all 4 mux channels
- **I2C bus speed**: 1 MHz (Fast-mode Plus)
- **Conversion time**: ~4.15 ms at DIG_FILT=4, OSR=0, RES=2
- **Frame time**: trigger (~2 ms) + USB send (hidden) + wait (~4.5 ms) + read (~3 ms) ≈ 9.5 ms
- **Frame rate**: ~106 fps at current settings
- **Serial throughput**: 324 B × 106 Hz ≈ 34 KB/s (well within USB full speed)

> ⚠️ Burst mode does **not** work with the I2C multiplexer. Sensors run continuous
> conversions while the mux disconnects them during channel switching, producing
> inconsistent readings. See `main.cpp` header comments for details.

### Tuning for higher speed

| Change | Effect | Trade-off |
|--------|--------|-----------|
| `DIG_FILT=0` (was 4) | ~200+ fps | More noise (can average in software) |
| `OSR=0`, `RES=0` | Slightly faster | Lower resolution |
| SPI instead of I2C | 400+ fps | Requires hardware redesign |

- **Typical GUI frame rate**: 100–106 FPS (limited by firmware, not the GUI)

## Credits

Based on:
- Original `5X_eflesh_stream.ino` by Venkatesh P
- `arduino-MLX90393` library by Theodore Yapo / Tess Hellebrekers
- ReSkin / AnySkin projects (https://any-skin.github.io)
