"""Cross-finger magnetic crosstalk data collection.
=====================================================
Standalone, additive tool -- does not import, modify, or depend on
unified_hand_demo.py, hand_sensing_gui.py, live_predictor_test.py,
live_phiclass_predictor.py, unified_controller.py, or any existing model/
CSV file. Only touches:
  - AmazingHand/hand_controller.py   (imported, not modified -- per-finger
    enable/close/open functions, used exactly as hand_control_gui.py
    already uses them)
  - MagneticSensors/eflesh_controller.py (imported, not modified -- raw
    EfleshController frame stream)
  - model_v1/crosstalk/raw/*.json    (new files written here only)

Purpose
-------
Characterise how much one finger's magnet position bleeds into the OTHER
3 fingers' magnetic sensor readings, purely from proximity as the hand's
geometry changes -- NOT from any real contact/touch. Every recording in
this tool is idle-only: nothing is pressed or touched while capturing.

Protocol (20 recordings total)
-------------------------------
For each of the 4 fingers (Index, Middle, Ring, Thumb), one at a time:
  1. Move that finger through 5 positions: 0%, 25%, 50%, 75%, 100%
     (0% = fully open, 100% = fully closed, 50% = neutral/middle_pos --
     same 0-100 scale hand_control_gui.py's position readout already
     uses, see move_to_position() below).
  2. The other 3 fingers stay fixed at the reference position (0%, fully
     open) throughout.
  3. At each of the 5 positions, ~4s of raw magnetic frames (all 4
     fingers/mux groups) are captured after a settle delay, with no
     contact happening.

4 fingers x 5 positions = 20 recordings, one JSON file each, saved to
model_v1/crosstalk/raw/ as crosstalk_movingfinger{N}_pos{P}.json, where N
is the moving finger's 1-indexed Finger/mux-group number (Index=1,
Middle=2, Ring=3, Thumb=4 -- the same mapping confirmed in
model_v1/hand_control_gui.py's FINGER_CONTROLS) and P is the 0/25/50/75/100
position just described.

This script is data collection ONLY: it does not build, fit, or evaluate
any crosstalk model, and it is never imported by the live sensing
pipeline. See README.md in this folder for the on-disk data format and
how to pick this back up later.

Usage:
    python model_v1/crosstalk/record_crosstalk.py [--port COM7] [--baud 2000000]
                                                    [--record-s 4.0] [--settle-s 1.5]

Press Ctrl+C to abort early -- fingers are always returned to the
reference (0%, open) position before the script exits, whether it
finished normally, was interrupted, or hit an error.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from MagneticSensors.eflesh_controller import EfleshController, FrameData, NUM_GROUPS
from AmazingHand import hand_controller as hc

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "raw"

# 0% = fully open, 100% = fully closed, 50% = neutral (middle_pos) -- same
# scale as hand_control_gui.py's per-finger position readout (see that
# module's POSITION_LABEL_NEUTRAL), so "position" means the same physical
# hand shape for every finger, including Index (see FINGER_DEFS' swapped
# flag below).
POSITIONS = [0, 25, 50, 75, 100]
REFERENCE_POSITION = 0

RECORD_S_DEFAULT = 4.0    # length of each idle capture window
SETTLE_S_DEFAULT = 1.5    # pause after a servo move before capturing, so
                           # captured frames reflect a mechanically settled
                           # finger, not motion/vibration transients

# (label, enable_fn, Finger object, mux group index, swapped) -- one row
# per finger, in Index/Middle/Ring/Thumb order. Confirmed mapping (from
# model_v1/hand_control_gui.py's FINGER_CONTROLS): Index = Finger 1
# (group index 0), Middle = Finger 2 (index 1), Ring = Finger 3 (index 2),
# Thumb = Finger 4 (index 3).
#
# `swapped` mirrors hand_control_gui.py's _physical_close_open: Index's
# hand_controller.Finger motion math is the original, unreversed form (no
# reverse=True), under which finger.close()/finger.open() physically move
# Index opposite to their names -- so for Index only, "move toward closed"
# means calling finger.open(), and vice versa. Applying the same swap here
# keeps `position_percent` meaning the same physical hand shape (0%=open,
# 100%=closed) for every finger, including Index.
FINGER_DEFS = [
    ("Index", hc.enable_index, hc.index, 0, True),
    ("Middle", hc.enable_middle, hc.middle, 1, False),
    ("Ring", hc.enable_ring, hc.ring, 2, False),
    ("Thumb", hc.enable_thumb, hc.thumb, 3, False),
]


def move_to_position(entry: tuple, position_percent: int, speed: int, sleep_after: bool = True) -> None:
    """Moves one FINGER_DEFS entry to `position_percent` on the 0=open /
    100=closed / 50=neutral scale described above. percent=0 on either
    close() or open() leaves the finger at its calibrated middle_pos, so
    this is the single pivot both directions agree on -- see
    hand_control_gui.py's POSITION_LABEL_NEUTRAL for the same reasoning.

    `sleep_after=False` skips the trailing settle sleep inside Finger.close/
    open -- for batch-dispatching several fingers' commands back-to-back
    (see move_all()/move_all_simultaneously() callers) so an earlier finger
    in the batch doesn't get a head start on a later one purely from Python-
    side sleep time (2026-09-03 staircase-closing fix; see
    verified-ground-truth-artifact-critical memory family). Callers doing
    that must still sleep once themselves after the whole batch is sent.

    2026-09-03: a per-swapped-finger SPEED SCALING correction (an angle-
    magnitude-ratio theory for why Index lags under incremental stepping)
    was tried here and REMOVED -- live testing across three tuning attempts
    (pure ratio, +1.75x, +1.35x) never converged (lag -> lead -> still-lag,
    "horrible") and made things worse than the sleep_after fix alone. Do
    NOT reintroduce a speed multiplier without a confirmed mechanism (e.g.
    actual timestamped per-finger position logs showing where real time is
    going) -- see staircase-close-fix memory for the full dead-end history.
    """
    _label, _enable_fn, finger, _group_idx, swapped = entry
    close_fn, open_fn = (finger.open, finger.close) if swapped else (finger.close, finger.open)
    position_percent = max(0, min(100, position_percent))
    if position_percent <= 50:
        open_fn(percent=(50 - position_percent) * 2, speed=speed, sleep_after=sleep_after)
    else:
        close_fn(percent=(position_percent - 50) * 2, speed=speed, sleep_after=sleep_after)


class FrameRecorder:
    """Collects raw frames from EfleshController while `active`, via a
    single on_frame callback installed once for the whole session.

    Kept deliberately simple (no baseline subtraction, no glitch guard,
    no calibration) -- this tool captures raw sensor values only; any
    baseline-relative crosstalk analysis happens later, offline, when the
    actual crosstalk model is built (see README.md).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active = False
        self._frames: list[FrameData] = []

    def on_frame(self, frame: FrameData) -> None:
        with self._lock:
            if self._active:
                self._frames.append(frame)

    def capture(self, duration_s: float) -> list[FrameData]:
        """Blocks for `duration_s`, collecting every frame that arrives
        during that window, then returns them."""
        with self._lock:
            self._frames = []
            self._active = True
        time.sleep(duration_s)
        with self._lock:
            self._active = False
            frames = list(self._frames)
        return frames


def frame_to_dict(frame: FrameData) -> dict:
    """Same groups -> sensors -> x/y/z/magnitude shape as
    unified_controller.py's DataRecord serialisation, for consistency with
    the rest of the codebase's recording format -- but standalone here
    (per-frame, no force field, no LoadScale dependency), since this tool
    intentionally shares no code with unified_controller.py.
    """
    return {
        "seq": frame.seq,
        "timestamp": frame.timestamp,
        "groups": [
            {
                "group_index": gi,
                "sensors": [
                    {"x": s.x, "y": s.y, "z": s.z, "magnitude": s.magnitude}
                    for s in frame.groups[gi]
                ],
            }
            for gi in range(NUM_GROUPS)
        ],
    }


def save_recording(out_dir: Path, entry: tuple, position_percent: int,
                    frames: list[FrameData], args: argparse.Namespace) -> Path:
    label, _enable_fn, _finger, group_idx, _swapped = entry
    other_entries = [e for e in FINGER_DEFS if e[3] != group_idx]

    filename = f"crosstalk_movingfinger{group_idx + 1}_pos{position_percent:03d}.json"
    path = out_dir / filename

    output = {
        "recording_info": {
            "tool": "record_crosstalk.py",
            "purpose": (
                "Cross-finger magnetic crosstalk characterisation -- pure "
                "magnet-proximity coupling between fingers as they move "
                "relative to each other, NOT contact. Idle only: no "
                "touching/pressing happened during this capture."
            ),
            "recorded_at_iso": datetime.now(timezone.utc).isoformat(),
            "num_frames": len(frames),
            "requested_duration_s": args.record_s,
            "settle_s": args.settle_s,
            "servo_speed": args.speed,
            "moving_finger": {
                "label": label,
                "group_index": group_idx,
                "position_percent": position_percent,
            },
            "other_fingers": [
                {"label": e[0], "group_index": e[3], "position_percent": REFERENCE_POSITION}
                for e in other_entries
            ],
        },
        "frames": [frame_to_dict(f) for f in frames],
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    return path


def print_summary(label: str, group_idx: int, position_percent: int,
                   frames: list[FrameData], group_label: dict[int, str]) -> None:
    if not frames:
        print(f"  [WARNING] {label} @ {position_percent:3d}%: 0 frames captured!")
        return
    print(f"  {label} (Finger {group_idx + 1}) @ {position_percent:3d}%: "
          f"{len(frames)} frames. Other fingers' sensor magnitude range (min .. max, mean):")
    for gi in range(NUM_GROUPS):
        if gi == group_idx:
            continue
        mags = [s.magnitude for f in frames for s in f.groups[gi]]
        if mags:
            print(f"    {group_label[gi]:>6s} (Finger {gi + 1}): "
                  f"{min(mags):8.2f} .. {max(mags):8.2f}   (mean {sum(mags) / len(mags):8.2f})")
        else:
            print(f"    {group_label[gi]:>6s} (Finger {gi + 1}): no data")


def wait_for_connection(connected_event: threading.Event, timeout_s: float = 10.0) -> bool:
    """Waits on the on_connected callback firing, NOT on ctrl.connected_port.

    EfleshController.start() sets self._port (and therefore connected_port)
    synchronously during port discovery, before its background thread has
    even opened the serial handle, let alone completed the handshake --
    polling connected_port races ahead of the actual connection and returns
    True too early. on_connected only fires once handshake/header parsing
    has fully succeeded (see EfleshController._emit_connected), so waiting
    on that event here is the same signal unified_controller.py's own
    _wait_for_magnetic_connection uses.
    """
    return connected_event.wait(timeout=timeout_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="eFlesh cross-finger magnetic crosstalk recorder")
    parser.add_argument("--port", help="Teensy COM port (auto-discovers if omitted)")
    parser.add_argument("--baud", type=int, default=2_000_000, help="Serial baud rate")
    parser.add_argument("--record-s", type=float, default=RECORD_S_DEFAULT,
                         help="Idle capture duration per position (seconds)")
    parser.add_argument("--settle-s", type=float, default=SETTLE_S_DEFAULT,
                         help="Pause after each servo move before capturing (seconds)")
    parser.add_argument("--speed", type=int, default=hc.DEFAULT_SPEED, help="Servo speed (1-6)")
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR), help="Output directory for JSON files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctrl = EfleshController()
    connected_event = threading.Event()
    ctrl.on_status = print
    ctrl.on_error = print
    ctrl.on_connected = lambda port: (print(f"Connected on {port}"), connected_event.set())
    ctrl.on_disconnected = lambda: print("Disconnected.")

    recorder = FrameRecorder()
    group_label = {entry[3]: entry[0] for entry in FINGER_DEFS}

    print("Connecting to eFlesh sensor array...")
    ctrl.on_frame = recorder.on_frame  # installed once; only appends while recorder is active
    ctrl.start(port=args.port, baud=args.baud)
    if not wait_for_connection(connected_event):
        print("ERROR: could not connect to eFlesh sensor array.")
        sys.exit(1)

    hc.connect(ctrl)
    for _label, enable_fn, *_ in FINGER_DEFS:
        enable_fn()

    print("Moving all fingers to reference position (0% = fully open)...")
    for entry in FINGER_DEFS:
        move_to_position(entry, REFERENCE_POSITION, args.speed)
    time.sleep(args.settle_s)

    saved_paths: list[Path] = []
    try:
        for entry in FINGER_DEFS:
            label, _enable_fn, _finger, group_idx, _swapped = entry
            print(f"\n=== Sweeping {label} (Finger {group_idx + 1}) through {POSITIONS} "
                  f"(other 3 fingers held at {REFERENCE_POSITION}% / open) ===")
            for pos in POSITIONS:
                move_to_position(entry, pos, args.speed)
                time.sleep(args.settle_s)
                frames = recorder.capture(args.record_s)
                path = save_recording(out_dir, entry, pos, frames, args)
                saved_paths.append(path)
                print(f"  Saved -> {path.name}")
                print_summary(label, group_idx, pos, frames, group_label)
            print(f"Returning {label} to reference position (0%)...")
            move_to_position(entry, REFERENCE_POSITION, args.speed)
            time.sleep(args.settle_s)
    finally:
        print("\nReturning all fingers to reference position (0%) before exit...")
        for entry in FINGER_DEFS:
            move_to_position(entry, REFERENCE_POSITION, args.speed)
        time.sleep(args.settle_s)
        ctrl.stop()

    print(f"\nDone. {len(saved_paths)} recordings saved to {out_dir}")


if __name__ == "__main__":
    main()
