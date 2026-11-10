"""Combined-movement validation recordings for the additive crosstalk-model
assumption flagged in crosstalk_model_report.md (Index, Middle, and Ring
each have 2 significant crosstalk sources -- additivity was never tested
since every raw/ recording moved exactly one finger at a time).

Reuses record_crosstalk.py's connection/servo/capture logic directly (via
import -- record_crosstalk.py itself is NOT modified) and drives TWO
fingers to matching positions simultaneously instead of one:

  - Index + Middle together: both moved to the SAME position (25%, 50%,
    75% -- 3 combinations), Ring and Thumb held fixed at 0%.
  - Middle + Ring together: same approach, Index and Thumb held fixed at
    0%.

3 positions x 2 pairs = 6 new recordings, saved to raw_combined/ (kept
separate from raw/'s single-finger sweeps, since this is a validation
set, not additional training data for train_crosstalk_model.py).

Usage:
    python model_v1/crosstalk/record_crosstalk_combined.py [--port COM7]
        [--baud 2000000] [--record-s 4.0] [--settle-s 1.5]

Press Ctrl+C to abort early -- fingers are always returned to the
reference (0%, open) position before the script exits.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from MagneticSensors.eflesh_controller import EfleshController
from AmazingHand import hand_controller as hc

# Reused, not reimplemented -- record_crosstalk.py is not modified by this
# script; only imported as a library.
from record_crosstalk import (
    FINGER_DEFS,
    REFERENCE_POSITION,
    FrameRecorder,
    frame_to_dict,
    move_to_position,
    wait_for_connection,
)

OUTPUT_DIR = SCRIPT_DIR / "raw_combined"

# (source finger A, source finger B) -- both driven to the SAME position
# simultaneously, one pair at a time. Chosen to match the 2 finger pairs
# crosstalk_model_report.md flagged as having 2 significant sources each
# on a common affected finger (Middle is affected-by-both in the Index+
# Middle... no -- see below: each pair here tests additivity ON the pair's
# own two OTHER fingers, i.e. Index+Middle moving together is measured on
# Ring+Thumb, and Middle+Ring moving together is measured on Index+Thumb).
COMBINED_PAIRS = [("Index", "Middle"), ("Middle", "Ring")]

# Fewer than the original 5-position sweep (0/25/50/75/100) -- this is a
# small, targeted validation set, not a full re-characterisation. 0% is
# skipped since both fingers are already at 0% during the reference/
# settle step (trivially delta=0 for both, same as the original sweeps).
POSITIONS = [25, 50, 75]

RECORD_S_DEFAULT = 4.0
SETTLE_S_DEFAULT = 1.5

FINGER_BY_LABEL = {entry[0]: entry for entry in FINGER_DEFS}


def save_recording(out_dir: Path, entry_a: tuple, entry_b: tuple, position: int,
                    frames: list, args: argparse.Namespace) -> Path:
    label_a, _enable_a, _finger_a, group_a, _swapped_a = entry_a
    label_b, _enable_b, _finger_b, group_b, _swapped_b = entry_b
    moving_group_indices = {group_a, group_b}
    other_entries = [e for e in FINGER_DEFS if e[3] not in moving_group_indices]

    filename = f"crosstalk_combined_{label_a.lower()}-{label_b.lower()}_pos{position:03d}.json"
    path = out_dir / filename

    output = {
        "recording_info": {
            "tool": "record_crosstalk_combined.py",
            "purpose": (
                "Combined-movement additivity validation set -- tests "
                "whether the additive crosstalk model (train_crosstalk_model.py) "
                "holds when 2 fingers move SIMULTANEOUSLY, unlike every raw/ "
                "recording (which moved exactly one finger). Idle only: no "
                "touching/pressing happened during this capture."
            ),
            "recorded_at_iso": datetime.now(timezone.utc).isoformat(),
            "num_frames": len(frames),
            "requested_duration_s": args.record_s,
            "settle_s": args.settle_s,
            "servo_speed": args.speed,
            "moving_fingers": [
                {"label": label_a, "group_index": group_a, "position_percent": position},
                {"label": label_b, "group_index": group_b, "position_percent": position},
            ],
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


def print_summary(label_a: str, label_b: str, position: int, frames: list,
                   other_entries: list) -> None:
    if not frames:
        print(f"  [WARNING] {label_a}+{label_b} @ {position:3d}%: 0 frames captured!")
        return
    print(f"  {label_a}+{label_b} @ {position:3d}% (together): {len(frames)} frames. "
          f"Other fingers' sensor magnitude range (min .. max, mean):")
    for e in other_entries:
        label, _enable_fn, _finger, gi, _swapped = e
        mags = [s.magnitude for f in frames for s in f.groups[gi]]
        if mags:
            print(f"    {label:>6s} (Finger {gi + 1}): "
                  f"{min(mags):8.2f} .. {max(mags):8.2f}   (mean {sum(mags) / len(mags):8.2f})")
        else:
            print(f"    {label:>6s} (Finger {gi + 1}): no data")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="eFlesh combined-movement crosstalk additivity validation recorder")
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

    print("Connecting to eFlesh sensor array...")
    ctrl.on_frame = recorder.on_frame
    ctrl.start(port=args.port, baud=args.baud)
    if not wait_for_connection(connected_event):
        print("ERROR: could not connect to eFlesh sensor array.")
        sys.exit(1)

    hc.connect(ctrl)
    for _label, enable_fn, *_ in FINGER_DEFS:
        enable_fn()

    print("Moving all fingers to reference position (0% = fully open)...")
    for entry in FINGER_DEFS:
        move_to_position(entry, REFERENCE_POSITION, args.speed, sleep_after=False)
    time.sleep(args.settle_s)

    saved_paths: list[Path] = []
    try:
        for label_a, label_b in COMBINED_PAIRS:
            entry_a = FINGER_BY_LABEL[label_a]
            entry_b = FINGER_BY_LABEL[label_b]
            other_entries = [e for e in FINGER_DEFS if e[0] not in (label_a, label_b)]
            print(f"\n=== Combined sweep: {label_a} + {label_b} together through {POSITIONS} "
                  f"(other 2 fingers held at {REFERENCE_POSITION}% / open) ===")
            for pos in POSITIONS:
                # Both goal_position writes sent back-to-back, no artificial
                # delay between them -- these two targets don't collide with
                # each other, and sending them together is the whole point:
                # this test needs both fingers actually moving at the same
                # time, not staggered. (2026-09-03: sleep_after=False now
                # actually delivers that -- move_to_position previously
                # still ran Finger.close/open's default 0.01s sleep between
                # these two calls despite this comment's claim otherwise.)
                move_to_position(entry_a, pos, args.speed, sleep_after=False)
                move_to_position(entry_b, pos, args.speed, sleep_after=False)
                time.sleep(args.settle_s)
                frames = recorder.capture(args.record_s)
                path = save_recording(out_dir, entry_a, entry_b, pos, frames, args)
                saved_paths.append(path)
                print(f"  Saved -> {path.name}")
                print_summary(label_a, label_b, pos, frames, other_entries)
            print(f"Returning {label_a} + {label_b} to reference position (0%)...")
            move_to_position(entry_a, REFERENCE_POSITION, args.speed, sleep_after=False)
            move_to_position(entry_b, REFERENCE_POSITION, args.speed, sleep_after=False)
            time.sleep(args.settle_s)
    finally:
        print("\nReturning all fingers to reference position (0%) before exit...")
        for entry in FINGER_DEFS:
            move_to_position(entry, REFERENCE_POSITION, args.speed, sleep_after=False)
        time.sleep(args.settle_s)
        ctrl.stop()

    print(f"\nDone. {len(saved_paths)} combined recordings saved to {out_dir}")


if __name__ == "__main__":
    main()
