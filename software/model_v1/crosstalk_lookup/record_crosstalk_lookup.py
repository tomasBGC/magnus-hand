"""Step 2: records the true (directly-measured, no fitting) magnetic
baseline for every 4-finger position combination in combinations_manifest.json
(see sample_combinations.py) -- ALL 4 fingers moved to their sampled
positions SIMULTANEOUSLY, unlike model_v1/crosstalk/record_crosstalk.py's
one-finger-at-a-time sweeps or record_crosstalk_combined*.py's two-finger
validation batches.

Reuses record_crosstalk.py's connection/servo/capture logic directly (via
import -- that module is NOT modified): FINGER_DEFS, move_to_position,
FrameRecorder, frame_to_dict, wait_for_connection. Nothing existing is
touched; this is purely additive.

SAFETY (Index+Thumb collision) -- read before running
-------------------------------------------------------
Index and Thumb physically collide if both are closed past a threshold at
the SAME TIME (see crosstalk_lookup_common.SAFETY_CAPS -- 20% for this
broad-coverage session, tighter than record_crosstalk_combined_round2.py's
30% cap for its narrower, deliberately-away-from-that-corner validation
batches). This is enforced structurally, not just checked at the two
endpoints of a move:

  1. combinations_manifest.json is already pre-filtered (see
     sample_combinations.py) -- but this script re-checks every single
     combination's target positions again anyway, immediately before
     sending ANY servo command for it, via check_safety_or_abort(). A
     violation here means the manifest was hand-edited or corrupted after
     generation; the whole session aborts immediately with a clear error
     rather than skipping the one bad entry and continuing.
  2. Every combination's move is executed from a SAFE WAYPOINT: this
     script always returns ALL 4 fingers to the reference position (0%)
     immediately after capturing each combination, before commanding the
     next one. Since every accepted combination individually satisfies
     "Index<=20% OR Thumb<=20%", and each finger's servo motion is a
     direct, monotonic point-to-point move (no overshoot) from its current
     commanded position to its new target, routing every transition
     through the shared 0% waypoint guarantees that whichever of
     Index/Thumb has the SMALLER (<=20%) target for a given combination
     never exceeds 20% at ANY point during its own transit (it moves
     monotonically within [0, target] and target<=20%) -- so the two
     fingers can never BOTH be above the cap at the same instant, not just
     at the two move endpoints. This is what makes the safety property a
     structural guarantee of the recording protocol, not merely an
     endpoint check -- there is no servo position feedback on this
     hardware (TeensyServoLink is write-only, see AmazingHand/
     hand_controller.py), so a genuine continuous-time guarantee has to
     come from the motion protocol itself, not from monitoring it.

If you ever change this script to skip the return-to-reference step
between combinations (e.g. for speed), the above guarantee no longer
holds and the per-move check alone (endpoints only) is not sufficient.

Usage:
    python model_v1/crosstalk_lookup/record_crosstalk_lookup.py
        [--port COM7] [--baud 2000000] [--record-s 4.0] [--settle-s 1.5]
        [--manifest combinations_manifest.json] [--resume]

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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

_CROSSTALK_DIR = SCRIPT_DIR.parent / "crosstalk"
if str(_CROSSTALK_DIR) not in sys.path:
    sys.path.insert(0, str(_CROSSTALK_DIR))

from MagneticSensors.eflesh_controller import EfleshController
from AmazingHand import hand_controller as hc

# Reused, not reimplemented -- record_crosstalk.py is not modified by this
# script; only imported as a library (same pattern record_crosstalk_combined.py
# and record_crosstalk_combined_round2.py already use).
from record_crosstalk import (
    FINGER_DEFS,
    FrameRecorder,
    frame_to_dict,
    move_to_position,
    wait_for_connection,
)

import crosstalk_lookup_common as clc

RECORD_S_DEFAULT = 4.0
SETTLE_S_DEFAULT = 1.5

# Mirrors sensing_gui.py's CROSSTALK_POSITION_RAMP_S_PER_100PCT (same
# estimate: ~1s for a full 0-100% swing at hand_controller.DEFAULT_SPEED).
# Not imported from there -- this pipeline is deliberately standalone from
# the live GUI (see crosstalk_lookup_common.py's module docstring), so the
# value is duplicated with this comment as the explicit link, same as every
# other "same number, different file" case already in this codebase.
RAMP_S_PER_100PCT = 1.0

FINGER_BY_LABEL = {entry[0]: entry for entry in FINGER_DEFS}


def check_safety_or_abort(positions: dict[str, float], context: str) -> None:
    reason = clc.violates_safety_caps(positions)
    if reason is not None:
        raise SystemExit(
            f"SAFETY ABORT ({context}): refusing to move -- {reason}. "
            f"No servo command was sent for this combination. Stopping the session immediately."
        )


def move_all_simultaneously(positions: dict[str, float], speed: int) -> None:
    """Sends all 4 fingers' goal-position commands back-to-back (no
    artificial inter-finger delay -- same reasoning as
    record_crosstalk_combined.py's simultaneous 2-finger move: the whole
    point is genuinely concurrent motion, not staggered). Caller MUST have
    already checked check_safety_or_abort() for `positions`.

    2026-09-03: this docstring's "no artificial inter-finger delay" claim
    was previously false -- move_to_position() without sleep_after=False
    still ran Finger.close/open's default 0.01s sleep_after=True for every
    finger in turn, giving earlier fingers in FINGER_BY_LABEL order (Index,
    Middle, Ring, Thumb) a cumulative head start every call -- a real
    contributor to a visible staircase-instead-of-parallel close reported
    live. Now actually matches the docstring."""
    for label, entry in FINGER_BY_LABEL.items():
        move_to_position(entry, positions[label], speed, sleep_after=False)
    time.sleep(0.01)


def transit_settle_s(from_positions: dict[str, float], to_positions: dict[str, float],
                      settle_s: float) -> float:
    """Estimated wait (seconds) for a simultaneous move from
    `from_positions` to `to_positions` to mechanically settle: the
    transit-time estimate (proportional to whichever finger travels
    FARTHEST, since all 4 move at once and the slowest one gates
    settling -- see RAMP_S_PER_100PCT) plus a fixed extra settle margin
    (vibration/mechanical settling after arrival, same role as
    record_crosstalk.py's own --settle-s)."""
    max_distance = max(abs(to_positions[label] - from_positions[label]) for label in clc.FINGER_LABELS)
    transit_s = RAMP_S_PER_100PCT * max_distance / 100.0
    return transit_s + settle_s


def save_recording(out_dir: Path, combo_index: int, positions: dict[str, float],
                    frames: list, args: argparse.Namespace, is_reference: bool = False) -> Path:
    tag = "reference" if is_reference else f"combo{combo_index:03d}"
    filename = (f"crosstalk_lookup_{tag}_idx{positions['Index']:03.0f}"
                f"_mid{positions['Middle']:03.0f}_ring{positions['Ring']:03.0f}"
                f"_thumb{positions['Thumb']:03.0f}.json")
    path = out_dir / filename

    output = {
        "recording_info": {
            "tool": "record_crosstalk_lookup.py",
            "purpose": (
                "Directly-measured (not fitted) true magnetic baseline for one "
                "4-finger position combination, all 4 fingers moved SIMULTANEOUSLY. "
                "Idle only: no touching/pressing happened during this capture."
            ),
            "recorded_at_iso": datetime.now(timezone.utc).isoformat(),
            "combo_index": None if is_reference else combo_index,
            "is_reference": is_reference,
            "num_frames": len(frames),
            "requested_duration_s": args.record_s,
            "servo_speed": args.speed,
            "positions": positions,
        },
        "frames": [frame_to_dict(f) for f in frames],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    return path


def print_summary(label: str, positions: dict[str, float], frames: list) -> None:
    pos_str = " ".join(f"{k}={v:.0f}%" for k, v in positions.items())
    if not frames:
        print(f"  [WARNING] {label} ({pos_str}): 0 frames captured!")
        return
    print(f"  {label} ({pos_str}): {len(frames)} frames captured.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="eFlesh crosstalk-lookup data collection -- all 4 fingers moved "
                    "simultaneously to each sampled combination, true baseline captured directly")
    parser.add_argument("--port", help="Teensy COM port (auto-discovers if omitted)")
    parser.add_argument("--baud", type=int, default=2_000_000, help="Serial baud rate")
    parser.add_argument("--record-s", type=float, default=RECORD_S_DEFAULT,
                         help="Idle capture duration per combination (seconds)")
    parser.add_argument("--settle-s", type=float, default=SETTLE_S_DEFAULT,
                         help="Extra settle margin on top of the transit-time estimate (seconds)")
    parser.add_argument("--speed", type=int, default=hc.DEFAULT_SPEED, help="Servo speed (1-6)")
    parser.add_argument("--manifest", default=str(clc.MANIFEST_JSON), help="Path to combinations_manifest.json")
    parser.add_argument("--out-dir", default=str(clc.RAW_DIR), help="Output directory for JSON files")
    parser.add_argument("--resume", action="store_true",
                         help="Skip combinations whose output file already exists in --out-dir")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"No manifest at {manifest_path} -- run sample_combinations.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    combinations = manifest["combinations"]

    # Fail fast, before ANY hardware is touched, if the manifest was
    # tampered with after generation -- mirrors record_crosstalk_combined_
    # round2.py's up-front TESTS re-check, applied here to every one of the
    # (potentially 100+) manifest entries.
    print(f"Verifying all {len(combinations)} manifest combinations against the "
          f"{clc.INDEX_THUMB_SAFETY_CAP_PCT}% Index+Thumb safety cap before touching hardware...")
    for combo in combinations:
        check_safety_or_abort(
            {label: combo[label] for label in clc.FINGER_LABELS},
            context=f"pre-flight check, combo_index={combo['combo_index']}",
        )
    print("Pre-flight safety check passed for every combination.")

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

    reference_positions = dict(clc.REFERENCE_POSITIONS)
    check_safety_or_abort(reference_positions, context="initial homing move")

    print("Moving all fingers to reference position (0% = fully open)...")
    move_all_simultaneously(reference_positions, args.speed)
    time.sleep(SETTLE_S_DEFAULT)

    saved_paths: list[Path] = []
    skipped_existing: list[int] = []
    session_log: list[dict] = []
    start_time = time.time()

    try:
        # Reference capture (all 4 at 0%) -- every combination's "shift" is
        # measured against this single recording (see build_lookup_dataset.py).
        ref_path = out_dir / (f"crosstalk_lookup_reference_idx000_mid000_ring000_thumb000.json")
        if args.resume and ref_path.exists():
            print(f"\n[RESUME] Reference capture already exists -- skipping -> {ref_path.name}")
        else:
            print("\n=== Capturing reference pose (all fingers at 0%) ===")
            frames = recorder.capture(args.record_s)
            path = save_recording(out_dir, -1, reference_positions, frames, args, is_reference=True)
            print(f"  Saved -> {path.name}")
            print_summary("Reference", reference_positions, frames)

        current_positions = reference_positions
        for combo in combinations:
            idx = combo["combo_index"]
            target = {label: float(combo[label]) for label in clc.FINGER_LABELS}

            expected_name = (f"crosstalk_lookup_combo{idx:03d}_idx{target['Index']:03.0f}"
                              f"_mid{target['Middle']:03.0f}_ring{target['Ring']:03.0f}"
                              f"_thumb{target['Thumb']:03.0f}.json")
            if args.resume and (out_dir / expected_name).exists():
                print(f"[RESUME] combo {idx}: already captured -- skipping -> {expected_name}")
                skipped_existing.append(idx)
                current_positions = reference_positions  # session always leaves fingers homed between combos
                continue

            # Defensive per-move re-check, immediately before sending ANY
            # command for this combination -- same check as the pre-flight
            # pass above, run again here so a violation still halts the
            # session immediately even in the (should-be-impossible) case
            # the manifest changed on disk between the pre-flight pass and
            # this move.
            check_safety_or_abort(target, context=f"combo_index={idx}")

            print(f"\n=== Combo {idx + 1}/{len(combinations)}: "
                  f"Index={target['Index']:.0f}% Middle={target['Middle']:.0f}% "
                  f"Ring={target['Ring']:.0f}% Thumb={target['Thumb']:.0f}% ===")
            wait_s = transit_settle_s(current_positions, target, args.settle_s)
            move_all_simultaneously(target, args.speed)
            time.sleep(wait_s)

            frames = recorder.capture(args.record_s)
            path = save_recording(out_dir, idx, target, frames, args)
            saved_paths.append(path)
            print(f"  Saved -> {path.name}  (settled {wait_s:.2f}s)")
            print_summary(f"Combo {idx}", target, frames)
            session_log.append({
                "combo_index": idx, "positions": target,
                "n_frames": len(frames), "settle_wait_s": wait_s,
            })

            # Return to the safe waypoint (0%) before the NEXT combination --
            # see module docstring for why this is what makes the Index+Thumb
            # safety property hold continuously through transit, not just at
            # move endpoints.
            check_safety_or_abort(reference_positions, context=f"post-combo{idx} homing move")
            wait_s_home = transit_settle_s(target, reference_positions, args.settle_s)
            move_all_simultaneously(reference_positions, args.speed)
            time.sleep(wait_s_home)
            current_positions = reference_positions
    finally:
        print("\nReturning all fingers to reference position (0%) before exit...")
        check_safety_or_abort(reference_positions, context="final homing move")
        move_all_simultaneously(reference_positions, args.speed)
        time.sleep(SETTLE_S_DEFAULT)
        ctrl.stop()

    elapsed_min = (time.time() - start_time) / 60.0
    print(f"\nDone in {elapsed_min:.1f} min. {len(saved_paths)} new combinations captured "
          f"(+{len(skipped_existing)} skipped via --resume) of {len(combinations)} requested, "
          f"-> {out_dir}")

    session_log_path = out_dir / "session_log.json"
    session_log_path.write_text(json.dumps({
        "recorded_at_iso": datetime.now(timezone.utc).isoformat(),
        "n_requested": len(combinations),
        "n_captured_this_run": len(saved_paths),
        "n_skipped_existing": len(skipped_existing),
        "elapsed_minutes": elapsed_min,
        "captures": session_log,
    }, indent=2), encoding="utf-8")
    print(f"Session log written -> {session_log_path}")


if __name__ == "__main__":
    main()
