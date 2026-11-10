"""Third combined-movement validation batch -- closes out the 2 remaining
originally-flagged CAUTION pairs that record_crosstalk_combined.py's first
batch (Index+Middle, Middle+Ring) didn't reach as the affected finger:

  - Index's own flagged CAUTION pair is (Middle, Thumb) -- never tested as
    the affected finger before this. Tested here via Middle+Thumb moving
    together, measured on Index (and Ring, incidentally).
  - Middle's own flagged CAUTION pair is (Index, Ring) -- never tested as
    the affected finger at all before this (Middle was always one of the
    two movers in the first batch). Tested here via Index+Ring moving
    together, measured on Middle (and Thumb, incidentally).

Plus one extra check beyond the originally-flagged pairs, for additional
confidence: Index+Thumb moving together, under a hard safety cap (see
below), measured on whichever of Middle/Ring shows a significant response.

Reuses record_crosstalk.py's connection/servo/capture code AND
record_crosstalk_combined.py's save_recording()/print_summary()/
FINGER_BY_LABEL/OUTPUT_DIR directly (imported, not modified or
duplicated) -- this script only adds a new, explicit list of (source A,
source B, positions) combinations to run, plus a hard safety cap enforced
IN CODE (not merely by the choice of positions in TESTS) for the one pair
that cannot be commanded past a certain combined travel without a
physical collision.

SAFETY: Index and Thumb physically collide if both are closed past ~30%
AT THE SAME TIME (confirmed hardware constraint on this hand, not a
software/servo-range limit). SAFETY_CAPS below is checked before EVERY
combined move for EVERY tested pair (not just Index+Thumb), both at
startup (before any hardware is touched) and again immediately before
each individual move -- so a future edit to TESTS that accidentally
exceeds the cap is refused rather than silently sent to the servos.

Saves into the SAME raw_combined/ directory as record_crosstalk_combined.py's
first batch, using the exact same filename convention
(crosstalk_combined_{a}-{b}_pos{P}.json). build_combined_dataset.py and
validate_additivity.py are UNMODIFIED and pick up all recordings in that
folder automatically the next time they're rerun -- no code changes
needed there to include this batch's data.

Usage:
    python model_v1/crosstalk/record_crosstalk_combined_round2.py
        [--port COM7] [--baud 2000000] [--record-s 4.0] [--settle-s 1.5]

Press Ctrl+C to abort early -- fingers are always returned to the
reference (0%, open) position before the script exits.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from MagneticSensors.eflesh_controller import EfleshController
from AmazingHand import hand_controller as hc

# Reused, not reimplemented -- neither module is modified by this script.
from record_crosstalk import (
    FINGER_DEFS,
    REFERENCE_POSITION,
    FrameRecorder,
    move_to_position,
    wait_for_connection,
)
from record_crosstalk_combined import (
    FINGER_BY_LABEL,
    OUTPUT_DIR,
    print_summary,
    save_recording,
)

RECORD_S_DEFAULT = 4.0
SETTLE_S_DEFAULT = 1.5

# (source A, source B, [positions]) -- each pair moved together through its
# OWN position list. Unlike record_crosstalk_combined.py's first batch (one
# shared POSITIONS list applied to both pairs), this batch needs per-pair
# positions because of the Index+Thumb safety cap below.
TESTS: list[tuple[str, str, list[int]]] = [
    # Closes out Index's own flagged CAUTION pair (Middle, Thumb).
    ("Middle", "Thumb", [25, 50, 75]),
    # Closes out Middle's own flagged CAUTION pair (Index, Ring).
    ("Index", "Ring", [25, 50, 75]),
    # Extra check beyond the originally-flagged pairs -- 10/20/30% only,
    # never 50/75% (see SAFETY_CAPS).
    ("Index", "Thumb", [10, 20, 30]),
]

# Hard collision limit: Index and Thumb physically crash into each other if
# both are closed past ~30% AT THE SAME TIME. Enforced per-move (see
# move_pair_safely) and again up front in main() before ANY hardware is
# touched, independent of whatever TESTS happens to list, so this can
# never be silently bypassed by a future edit.
SAFETY_CAPS: dict[frozenset[str], int] = {frozenset({"Index", "Thumb"}): 30}


def check_safety_cap(label_a: str, label_b: str, position: int) -> None:
    cap = SAFETY_CAPS.get(frozenset({label_a, label_b}))
    if cap is not None and position > cap:
        raise SystemExit(
            f"SAFETY ABORT: refusing {label_a}+{label_b} at {position}% simultaneously -- "
            f"exceeds the {cap}% collision-risk cap for this finger pair. No servo command "
            f"was sent for this position."
        )


def move_pair_safely(entry_a: tuple, entry_b: tuple, position: int, speed: int) -> None:
    check_safety_cap(entry_a[0], entry_b[0], position)
    # sleep_after=False on both -- these need to move genuinely together,
    # not staggered by Finger.close/open's default settle sleep (2026-09-03
    # staircase-closing fix). Callers already sleep(args.settle_s) right
    # after this returns.
    move_to_position(entry_a, position, speed, sleep_after=False)
    move_to_position(entry_b, position, speed, sleep_after=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="eFlesh combined-movement crosstalk validation recorder -- round 2 "
                    "(closes out Index's and Middle's own CAUTION pairs, plus an extra "
                    "Index+Thumb check under a hard 30% simultaneous-movement safety cap)")
    parser.add_argument("--port", help="Teensy COM port (auto-discovers if omitted)")
    parser.add_argument("--baud", type=int, default=2_000_000, help="Serial baud rate")
    parser.add_argument("--record-s", type=float, default=RECORD_S_DEFAULT,
                         help="Idle capture duration per position (seconds)")
    parser.add_argument("--settle-s", type=float, default=SETTLE_S_DEFAULT,
                         help="Pause after each servo move before capturing (seconds)")
    parser.add_argument("--speed", type=int, default=hc.DEFAULT_SPEED, help="Servo speed (1-6)")
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR),
                         help="Output directory for JSON files (default: same raw_combined/ "
                              "as record_crosstalk_combined.py's first batch)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fail fast, before ANY hardware is touched, if TESTS was ever edited to
    # violate a safety cap -- the same check move_pair_safely enforces per
    # move, run here up front too so a bad edit is caught at startup, not
    # partway through a live session with fingers already moving.
    for label_a, label_b, positions in TESTS:
        cap = SAFETY_CAPS.get(frozenset({label_a, label_b}))
        if cap is not None:
            over = [p for p in positions if p > cap]
            if over:
                raise SystemExit(
                    f"SAFETY ABORT: TESTS lists {label_a}+{label_b} positions {over} which "
                    f"exceed the {cap}% collision-risk cap for this pair. Fix TESTS before running."
                )

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
        for label_a, label_b, positions in TESTS:
            entry_a = FINGER_BY_LABEL[label_a]
            entry_b = FINGER_BY_LABEL[label_b]
            other_entries = [e for e in FINGER_DEFS if e[0] not in (label_a, label_b)]
            cap = SAFETY_CAPS.get(frozenset({label_a, label_b}))
            cap_note = f" [SAFETY CAP: {cap}% max]" if cap is not None else ""
            print(f"\n=== Combined sweep: {label_a} + {label_b} together through {positions}"
                  f"{cap_note} (other 2 fingers held at {REFERENCE_POSITION}% / open) ===")
            for pos in positions:
                move_pair_safely(entry_a, entry_b, pos, args.speed)
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
