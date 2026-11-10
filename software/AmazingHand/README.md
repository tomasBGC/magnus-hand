# AmazingHand — Servo Control

Drives the [pollen-robotics AmazingHand](https://github.com/pollen-robotics/AmazingHand)'s
Feetech SCS0009 finger servos. This is the layer that actually moves the
hand — see [`model_v1/README.md`](../model_v1/README.md) for the sensing
side that decides *when* to close/open based on live tactile predictions.

## How it connects

There is **no separate serial connection** for the servos. `hand_controller.py`
relays servo commands over the same USB serial connection that
`MagneticSensors/eflesh_controller.py`'s `EfleshController` already opened
and is reading sensor frames from (see `TeensyServoLink` — write-only, so it
never competes with the sensor reader thread for the port). You must connect
the sensor controller first, then call `connect()` to share its connection:

```python
from MagneticSensors.eflesh_controller import EfleshController
from AmazingHand.hand_controller import connect, enable_index, close_index, open_index

mag = EfleshController()
mag.start()                     # begins streaming sensor frames
# ... wait until mag.connected_port is set ...
connect(mag)                    # shares mag's serial connection
enable_index()                  # torque-enable both servos in the finger
close_index()                   # close to 100% by default
open_index(50)                  # open 50% of the way
```

In practice you won't call this directly — `model_v1/hand_control_gui.py`
already wires this up (servo control panel + live sensing in one GUI).

## Fingers and functions

Four fingers, each two SCS0009 servos (`index`, `middle`, `ring`, `thumb`
module-level `Finger` instances). For each finger `X`:

| Function | Effect |
|---|---|
| `enable_X()` | Torque-enable both servos in that finger |
| `close_X(percent=100)` | Close by `percent` (0-100) at `DEFAULT_SPEED` |
| `open_X(percent=100)` | Open by `percent` (0-100) at `DEFAULT_SPEED` |

**Safety, by design:** each finger is driven only by an explicit function
call — there is intentionally no "all fingers" loop/helper here. Test one
finger at a time on the physically mounted hand before wiring up the rest.

## Register protocol (for reference / porting)

Not invented here — mirrors the official
`pollen-robotics/AmazingHand/PythonExample/AmazingHand_FingerTest.py`, with
register addresses and raw-value formulas from that project's rustypot
`Scs0009PyController` (`src/servo/feetech/scs0009.rs`):

| Register | Addr | Size | Raw value formula |
|---|---|---|---|
| `torque_enable` | 40 | 1 byte | — |
| `goal_position` | 42 | i16 | `int(1024 * angle_rad / rad(300) + 511)` |
| `goal_speed` | 46 | u16 | `int(value / (rad(300) / 1024))` |

Both 2-byte registers go out **big-endian** on the wire (SCS0009-specific —
unlike the little-endian STS3215). Frames are `[0xFA, 0xCE, servo_id, addr,
length, ...data_bytes, checksum]`, checksum = `(servo_id + addr + length +
sum(data_bytes)) & 0xFF`.
