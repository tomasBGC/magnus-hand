"""Step 2: for every 4-finger position combination in
combinations_manifest.json (see sample_combinations.py), moves ALL 4
fingers to their sampled positions SIMULTANEOUSLY, waits for settling, then
captures the LIVE MODEL'S ACTUAL DISPLAYED OUTPUT (Force, confirmed
Contact, raw Force) for a few seconds -- with genuinely no touch anywhere.

This is deliberately NOT a raw-sensor-frame capture (unlike
model_v1/crosstalk_lookup/record_crosstalk_lookup.py) -- it drives the
EXACT same per-finger processing function the deployed GUI runs
(sensing_gui._process_one_finger, imported unmodified from the currently-
live model_v1/sensing_gui.py, itself the 2026-07-17 candidate promotion:
Schmitt trigger, spike guard, reverted crosstalk backends, and the
auto-rebaseline-on-return-to-open fix, all active), driven continuously
across the WHOLE 500-combination session (baseline tracking, periodic
recalibration, and the auto-rebaseline fix all carry state across combos
exactly as they would in a real continuous-use session) -- headlessly, no
Qt/GUI required, since _process_one_finger and FingerState are both plain,
Qt-independent Python (same reason model_v1/self_motion/ and
model_v1/production_review/verify_candidate_offline.py could already drive
this function directly).

SAFETY (Index+Thumb collision) -- read before running
-------------------------------------------------------
Same structural protocol as model_v1/crosstalk_lookup/record_crosstalk_lookup.py,
at a TIGHTER 15% cap (true_zero_common.SAFETY_CAPS) per explicit instruction
for this session:

  1. combinations_manifest.json is pre-filtered (see sample_combinations.py)
     -- this script re-verifies EVERY combination again before touching any
     hardware, and AGAIN immediately before sending any servo command for
     it, via check_safety_or_abort(). A violation aborts the whole session
     immediately.
  2. Every combination's move is executed from the safe 0% waypoint: this
     script always returns ALL 4 fingers to the reference position
     immediately after capturing each combination, before commanding the
     next one -- the same "route every transition through the shared safe
     waypoint" argument as record_crosstalk_lookup.py's own docstring (see
     that file for the full monotonic-transit argument for why this makes
     the safety property hold continuously through transit, not just at
     move endpoints).

Usage:
    python model_v1/true_zero_dataset/record_true_zero_dataset.py
        [--port COM7] [--baud 2000000] [--capture-s 3.0] [--settle-s 1.2]
        [--manifest combinations_manifest.json] [--resume]

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

_CROSSTALK_DIR = SCRIPT_DIR.parent / "crosstalk"
if str(_CROSSTALK_DIR) not in sys.path:
    sys.path.insert(0, str(_CROSSTALK_DIR))

_MODEL_V1_DIR = SCRIPT_DIR.parent
if str(_MODEL_V1_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_V1_DIR))

from MagneticSensors.eflesh_controller import EfleshController
from AmazingHand import hand_controller as hc

# Reused, not reimplemented, same pattern crosstalk_lookup uses.
from record_crosstalk import FINGER_DEFS, move_to_position, wait_for_connection

# The CURRENTLY-LIVE sensing_gui.py (model_v1/, promoted 2026-07-17) --
# imported as a library. Nothing in this file modifies it.
import sensing_gui as sg

import true_zero_common as tzc

CAPTURE_S_DEFAULT = 3.0
SETTLE_S_DEFAULT = 1.2
POLL_INTERVAL_S = sg.DISPLAY_INTERVAL_S  # match the live GUI's own poll cadence (~20Hz)

# Same duplicated-with-a-comment-link pattern crosstalk_lookup/
# record_crosstalk_lookup.py already uses for its own RAMP_S_PER_100PCT --
# this pipeline is deliberately standalone from the live GUI's own transit-
# ramp state (HandSensingEngine._finger_position_ramps), so the estimate is
# duplicated here rather than imported, with this comment as the explicit
# link. Matches sg.CROSSTALK_POSITION_RAMP_S_PER_100PCT's value.
RAMP_S_PER_100PCT = sg.CROSSTALK_POSITION_RAMP_S_PER_100PCT

FINGER_BY_LABEL = {entry[0]: entry for entry in FINGER_DEFS}


def check_safety_or_abort(positions: dict[str, float], context: str) -> None:
    reason = tzc.violates_safety_caps(positions)
    if reason is not None:
        raise SystemExit(
            f"SAFETY ABORT ({context}): refusing to move -- {reason}. "
            f"No servo command was sent for this combination. Stopping the session immediately."
        )


def move_all_simultaneously(positions: dict[str, float], speed: int) -> None:
    # Despite the name, this used to dispatch fingers one at a time with a
    # sleep after each (Finger.close/open's default sleep_after=True) --
    # giving earlier fingers in FINGER_BY_LABEL order (Index, Middle, Ring,
    # Thumb) a cumulative head start, a real contributor to a visible
    # staircase-instead-of-parallel close reported live 2026-09-03. Now
    # dispatches all 4 fingers' commands back-to-back before the one settle
    # sleep, actually matching this function's name and docstring intent.
    for label, entry in FINGER_BY_LABEL.items():
        move_to_position(entry, positions[label], speed, sleep_after=False)
    time.sleep(0.01)


def transit_duration_s(from_positions: dict[str, float], to_positions: dict[str, float]) -> float:
    max_distance = max(abs(to_positions[label] - from_positions[label]) for label in tzc.FINGER_LABELS)
    return RAMP_S_PER_100PCT * max_distance / 100.0


class LatestFrameHolder:
    """Same 'reader thread stashes, main loop always takes latest, drops
    older' pattern as sensing_gui.HandSensingEngine._on_raw_frame/poll() --
    keeps the serial reader thread cheap and this script's own polling loop
    faithful to the live GUI's actual ~20Hz processing cadence rather than
    processing every one of the sensor's much-faster raw frames (which
    would make the Contact hysteresis's frame-counted thresholds resolve
    unrealistically fast in wall-clock time).
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._last_seq_shown = None

    def on_frame(self, frame) -> None:
        with self._lock:
            self._frame = frame

    def get_latest_if_new(self):
        with self._lock:
            frame = self._frame
        if frame is None or frame.seq == self._last_seq_shown:
            return None
        self._last_seq_shown = frame.seq
        return frame


def run_calibration(ctrl_frames: LatestFrameHolder, predictor, duration_s: float, min_samples: int):
    """Mirrors HandSensingEngine.finish_calibration()'s exact logic:
    averages frame_to_raw_values() over `duration_s` seconds, per finger.
    Fingers must already be settled at the reference (0%) position."""
    samples: list[list[dict]] = [[] for _ in range(sg.NUM_FINGERS)]
    deadline = time.time() + duration_s
    while time.time() < deadline:
        frame = ctrl_frames.get_latest_if_new()
        if frame is not None:
            for i in range(sg.NUM_FINGERS):
                samples[i].append(sg.frame_to_raw_values(frame, predictor.raw_cols, i))
        time.sleep(0.01)
    baselines = []
    counts = []
    for i in range(sg.NUM_FINGERS):
        s = samples[i]
        counts.append(len(s))
        if s:
            baselines.append({col: sum(x[col] for x in s) / len(s) for col in predictor.raw_cols})
        else:
            baselines.append({col: 0.0 for col in predictor.raw_cols})
    return baselines, counts


def recompute_input_features(predictor, fstate, frame, crosstalk_models,
                              finger_positions: dict[str, float]) -> list[float]:
    """Read-only replication of the exact input-feature computation inside
    the LIVE sensing_gui._process_one_finger() (baseline subtraction, then
    crosstalk correction if enabled) -- duplicated here ONLY for logging
    purposes, since _process_one_finger's raw_delta is a local variable, not
    part of its returned FingerResult, and this file must not modify the
    live sensing_gui.py to expose it. Mirrors that function's own logic
    line-for-line (see model_v1/sensing_gui.py's own raw_delta/crosstalk
    block) so what's logged here is exactly what the model actually saw,
    needed for step 5's retraining (the recorded Force/Contact OUTPUT
    alone isn't enough to retrain a head against -- the corresponding
    INPUT feature vector is what a training row actually needs).
    """
    raw_absolute = sg.frame_to_raw_values(frame, predictor.raw_cols, fstate.index)
    raw_delta = {col: v - fstate.baseline[col] for col, v in raw_absolute.items()}
    my_label = sg.CROSSTALK_FINGER_LABELS[fstate.index]
    source = sg.CROSSTALK_CORRECTION_SOURCE.get(my_label, "pairwise_additive")
    model = crosstalk_models.get(source)
    if sg.CROSSTALK_CORRECTION_ENABLED and model is not None and finger_positions is not None:
        other_positions = {lbl: pos for lbl, pos in finger_positions.items() if lbl != my_label}
        predict_fn = sg.CROSSTALK_CORRECTION_BACKENDS[source]
        predicted_crosstalk = predict_fn(model, my_label, other_positions)
        raw_delta = {col: v - predicted_crosstalk.get(col, 0.0) for col, v in raw_delta.items()}
    return predictor._build_feature_row(raw_delta)[0].tolist()


def poll_and_process(ctrl_frames: LatestFrameHolder, predictor, fstates, crosstalk_models,
                      finger_positions: dict[str, float], transit_end_time: dict[str, float],
                      until_time: float, log_slice: "list | None" = None):
    """Runs the live per-finger processing loop (sg._process_one_finger,
    unmodified) from now until wall-clock `until_time`, at POLL_INTERVAL_S
    cadence, exactly mirroring HandSensingEngine.poll()'s own tick logic --
    including in_own_transit/any_finger_in_transit derived from
    transit_end_time, so CONTACT_SUPPRESS_DURING_OWN_TRANSIT, the baseline
    contamination gate, and AUTO_REBASELINE_ON_RETURN_TO_OPEN_ENABLED all
    behave exactly as they would live. If log_slice is a list, appends
    (now, {label: FingerResult}) for every tick -- used only during the
    actual capture window, not during transit/settle, to keep memory/output
    bounded.
    """
    next_tick = time.time()
    while True:
        now = time.time()
        if now >= until_time:
            break
        frame = ctrl_frames.get_latest_if_new()
        if frame is not None:
            in_own_transit = {label: now < transit_end_time.get(label, 0.0) for label in tzc.FINGER_LABELS}
            any_transit = any(in_own_transit.values())
            results = {}
            features_by_label = {}
            for fstate in fstates:
                label = sg.CROSSTALK_FINGER_LABELS[fstate.index]
                # MUST recompute features BEFORE calling _process_one_finger:
                # that call may mutate fstate.baseline (periodic tracking,
                # or the auto-rebaseline fix), and the feature vector logged
                # here has to reflect whatever baseline actually produced
                # this tick's fr.force/fr.raw_force, not a later one.
                if log_slice is not None:
                    features_by_label[label] = recompute_input_features(
                        predictor, fstate, frame, crosstalk_models, finger_positions)
                fr = sg._process_one_finger(
                    predictor, fstate, frame, now,
                    crosstalk_models=crosstalk_models,
                    finger_positions=finger_positions,
                    in_own_transit=in_own_transit[label],
                    any_finger_in_transit=any_transit,
                )
                results[label] = fr
                if fr.recal_message:
                    print(f"    [{fr.recal_message}]")
            if log_slice is not None:
                log_slice.append((now, {
                    label: {"force": r.force, "raw_force": r.raw_force, "contact": r.contact,
                            "glitched": r.glitched, "features": features_by_label[label]}
                    for label, r in results.items()
                }))
        sleep_for = max(0.0, next_tick + POLL_INTERVAL_S - time.time())
        time.sleep(sleep_for)
        next_tick += POLL_INTERVAL_S


def move_and_settle(ctrl_frames, predictor, fstates, crosstalk_models,
                     finger_positions: dict[str, float], target: dict[str, float],
                     speed: int, settle_s: float) -> None:
    """Commands all 4 fingers to `target` simultaneously and runs the live
    processing loop through the estimated transit + settle window,
    updating `finger_positions` in place to `target` immediately (this
    pipeline does not interpolate a ramp position during transit -- see
    module docstring's "assume target throughout transit" note; Contact
    evaluation during transit is frozen by CONTACT_SUPPRESS_DURING_OWN_
    TRANSIT regardless via in_own_transit, so the crosstalk-correction
    input during transit only affects diagnostics, not any captured data
    point in this dataset)."""
    duration = transit_duration_s(finger_positions, target)
    now = time.time()
    transit_end_time = {label: now + duration for label in tzc.FINGER_LABELS}
    finger_positions.clear()
    finger_positions.update(target)
    move_all_simultaneously(target, speed)
    poll_and_process(ctrl_frames, predictor, fstates, crosstalk_models,
                      finger_positions, transit_end_time, until_time=now + duration + settle_s)


def save_combo_capture(out_dir: Path, combo: dict, capture_log: list, args: argparse.Namespace) -> Path:
    idx = combo["combo_index"]
    filename = (f"true_zero_combo{idx:04d}_idx{combo['Index']:03.0f}"
                f"_mid{combo['Middle']:03.0f}_ring{combo['Ring']:03.0f}"
                f"_thumb{combo['Thumb']:03.0f}.json")
    path = out_dir / filename
    output = {
        "recording_info": {
            "tool": "record_true_zero_dataset.py",
            "purpose": "Live model's DISPLAYED Force/Contact/raw_Force at one simultaneous "
                       "4-finger position combination -- idle only, no touch.",
            "recorded_at_iso": datetime.now(timezone.utc).isoformat(),
            "combo_index": idx,
            "is_boundary_case": combo["is_boundary_case"],
            "positions": {label: combo[label] for label in tzc.FINGER_LABELS},
            "capture_s": args.capture_s,
            "n_ticks": len(capture_log),
        },
        "ticks": [{"t": t, **frs} for t, frs in capture_log],
    }
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return path


def summarize_combo(capture_log: list) -> dict:
    """Per-finger mean/max displayed Force, max raw_Force, confirmed-contact
    fraction, and glitch fraction over the whole capture window."""
    summary = {}
    for label in tzc.FINGER_LABELS:
        forces = [frs[label]["force"] for _, frs in capture_log]
        raw_forces = [frs[label]["raw_force"] for _, frs in capture_log]
        contacts = [frs[label]["contact"] for _, frs in capture_log]
        glitches = [frs[label]["glitched"] for _, frs in capture_log]
        n = max(1, len(forces))
        summary[label] = {
            "mean_force": sum(forces) / n,
            "max_force": max(forces) if forces else 0.0,
            "max_raw_force": max(raw_forces) if raw_forces else 0.0,
            "confirmed_contact_fraction": sum(contacts) / n,
            "glitch_fraction": sum(glitches) / n,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="true_zero_dataset collection -- all 4 fingers moved simultaneously to each "
                    "sampled combination, LIVE model's displayed Force/Contact captured directly")
    parser.add_argument("--port", help="Teensy COM port (auto-discovers if omitted)")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--capture-s", type=float, default=CAPTURE_S_DEFAULT,
                         help="Idle capture duration per combination (seconds)")
    parser.add_argument("--settle-s", type=float, default=SETTLE_S_DEFAULT,
                         help="Extra settle margin on top of the transit-time estimate (seconds)")
    parser.add_argument("--speed", type=int, default=hc.DEFAULT_SPEED)
    parser.add_argument("--manifest", default=str(tzc.MANIFEST_JSON))
    parser.add_argument("--out-dir", default=str(tzc.RAW_DIR))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                         help="Stop after this many NEWLY-captured combinations this run "
                              "(--resume-skipped ones don't count against the limit) -- for "
                              "running the session in smaller batches.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"No manifest at {manifest_path} -- run sample_combinations.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    combinations = manifest["combinations"]

    print(f"Verifying all {len(combinations)} manifest combinations against the "
          f"{tzc.INDEX_THUMB_SAFETY_CAP_PCT}% Index+Thumb safety cap before touching hardware...")
    for combo in combinations:
        check_safety_or_abort(
            {label: combo[label] for label in tzc.FINGER_LABELS},
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

    frame_holder = LatestFrameHolder()

    print("Connecting to eFlesh sensor array...")
    ctrl.on_frame = frame_holder.on_frame
    ctrl.start(port=args.port, baud=args.baud)
    if not wait_for_connection(connected_event):
        print("ERROR: could not connect to eFlesh sensor array.")
        sys.exit(1)

    hc.connect(ctrl)
    for _label, enable_fn, *_ in FINGER_DEFS:
        enable_fn()

    reference_positions = dict(tzc.REFERENCE_POSITIONS)
    check_safety_or_abort(reference_positions, context="initial homing move")

    print("Moving all fingers to reference position (0% = fully open)...")
    move_all_simultaneously(reference_positions, args.speed)
    time.sleep(2.0)

    print(f"Loading live model + predictor from {sg.MODEL_PATH.name}...")
    predictor = sg.LivePhiClassPredictor(model_path=str(sg.MODEL_PATH))
    fstates = [sg.FingerState(index=i, label=f"Finger {i + 1}") for i in range(sg.NUM_FINGERS)]

    crosstalk_models: dict = {"pairwise_additive": None, "nearest_neighbor": None,
                              "nearest_neighbor_smoothed": None, "ridge_quadratic": None}
    try:
        crosstalk_models["pairwise_additive"] = sg.load_crosstalk_model(sg.CROSSTALK_MODEL_PATH)
        print(f"  loaded {sg.CROSSTALK_MODEL_PATH.name} (pairwise_additive)")
    except Exception as exc:
        print(f"  FAILED to load pairwise_additive crosstalk model: {exc}")
    try:
        nn_model = sg.load_nn_lookup_model(sg.NN_LOOKUP_MODEL_PATH)
        crosstalk_models["nearest_neighbor"] = nn_model
        crosstalk_models["nearest_neighbor_smoothed"] = nn_model
        print(f"  loaded {sg.NN_LOOKUP_MODEL_PATH.name} (nearest_neighbor, nearest_neighbor_smoothed)")
    except Exception as exc:
        print(f"  FAILED to load nearest_neighbor crosstalk model: {exc}")
    try:
        crosstalk_models["ridge_quadratic"] = sg.load_ridge_crosstalk_model(sg.RIDGE_CROSSTALK_MODEL_PATH)
        print(f"  loaded {sg.RIDGE_CROSSTALK_MODEL_PATH.name} (ridge_quadratic)")
    except Exception as exc:
        print(f"  FAILED to load ridge_quadratic crosstalk model: {exc}")

    print(f"Calibrating baseline ({sg.BASELINE_DURATION_S:.0f}s)...")
    baselines, counts = run_calibration(frame_holder, predictor, sg.BASELINE_DURATION_S, sg.MIN_BASELINE_SAMPLES)
    for i, fstate in enumerate(fstates):
        fstate.baseline = baselines[i]
        fstate.last_recal_time = time.time()
        fstate.last_successful_recal_time = time.time()
        fstate.baseline_at_last_report = dict(baselines[i])
    print(f"  Calibration samples per finger: {counts}")

    finger_positions = dict(reference_positions)
    saved_paths: list[Path] = []
    skipped_existing: list[int] = []
    session_summaries: list[dict] = []
    start_time = time.time()

    if args.resume:
        already_done = sum(1 for c in combinations if (out_dir / (
            f"true_zero_combo{c['combo_index']:04d}_idx{c['Index']:03.0f}"
            f"_mid{c['Middle']:03.0f}_ring{c['Ring']:03.0f}_thumb{c['Thumb']:03.0f}.json")).exists())
        print(f"[RESUME] {already_done}/{len(combinations)} combinations already captured -- "
              f"will skip these without re-homing (fingers assumed still at reference).")

    try:
        for combo in combinations:
            idx = combo["combo_index"]
            target = {label: float(combo[label]) for label in tzc.FINGER_LABELS}

            expected_name = (f"true_zero_combo{idx:04d}_idx{target['Index']:03.0f}"
                              f"_mid{target['Middle']:03.0f}_ring{target['Ring']:03.0f}"
                              f"_thumb{target['Thumb']:03.0f}.json")
            if args.resume and (out_dir / expected_name).exists():
                skipped_existing.append(idx)
                # Every REAL capture (and the very first skip after startup
                # homing) already leaves fingers at the reference position --
                # only actually re-home if that's for some reason not
                # already true, instead of unconditionally re-issuing a
                # (redundant, but harmless if it ever WAS needed) move+settle
                # wait for every single skipped combo. With hundreds of
                # already-captured combos to skip past on a resumed run,
                # this avoids minutes of pure waste re-confirming a position
                # the fingers are already sitting at.
                if finger_positions != reference_positions:
                    move_and_settle(frame_holder, predictor, fstates, crosstalk_models,
                                    finger_positions, reference_positions, args.speed, args.settle_s)
                continue

            check_safety_or_abort(target, context=f"combo_index={idx}")

            tag = "BOUNDARY" if combo["is_boundary_case"] else "general"
            print(f"\n=== Combo {idx + 1}/{len(combinations)} [{tag}]: "
                  f"Index={target['Index']:.0f}% Middle={target['Middle']:.0f}% "
                  f"Ring={target['Ring']:.0f}% Thumb={target['Thumb']:.0f}% ===")

            move_and_settle(frame_holder, predictor, fstates, crosstalk_models,
                             finger_positions, target, args.speed, args.settle_s)

            capture_log: list = []
            poll_and_process(frame_holder, predictor, fstates, crosstalk_models,
                              finger_positions, transit_end_time={},
                              until_time=time.time() + args.capture_s, log_slice=capture_log)

            path = save_combo_capture(out_dir, combo, capture_log, args)
            saved_paths.append(path)
            summary = summarize_combo(capture_log)
            session_summaries.append({"combo_index": idx, "is_boundary_case": combo["is_boundary_case"],
                                      "positions": target, "summary": summary})
            flagged = [f"{l}={s['max_force']:.2f}N" for l, s in summary.items() if s["max_force"] > 0.01]
            print(f"  Saved -> {path.name}  ({len(capture_log)} ticks)")
            if flagged:
                print(f"  [NONZERO FORCE] {', '.join(flagged)}")
            else:
                print(f"  Clean: 0.00N on all 4 fingers")

            check_safety_or_abort(reference_positions, context=f"post-combo{idx} homing move")
            move_and_settle(frame_holder, predictor, fstates, crosstalk_models,
                             finger_positions, reference_positions, args.speed, args.settle_s)

            if args.limit is not None and len(saved_paths) >= args.limit:
                print(f"\n--limit {args.limit} reached this run -- stopping (fingers already "
                      f"homed to 0% above). Resume with --resume to continue with the rest.")
                break
    finally:
        print("\nReturning all fingers to reference position (0%) before exit...")
        check_safety_or_abort(reference_positions, context="final homing move")
        move_all_simultaneously(reference_positions, args.speed)
        time.sleep(2.0)
        ctrl.stop()

    elapsed_min = (time.time() - start_time) / 60.0
    print(f"\nDone in {elapsed_min:.1f} min. {len(saved_paths)} new combinations captured "
          f"(+{len(skipped_existing)} skipped via --resume) of {len(combinations)} requested, "
          f"-> {out_dir}")

    # Merge with any EXISTING session_log.json (from an earlier batch/run of
    # this same session) rather than overwrite it -- keyed by combo_index,
    # so running in --limit-ed batches never loses an earlier batch's
    # summaries. build_true_zero_dataset.py also independently re-derives
    # its combo list from the manifest + which raw/*.json files actually
    # exist on disk (not solely from this log), so this merge is a
    # convenience/redundancy, not the only source of truth.
    session_log_path = out_dir / "session_log.json"
    merged_summaries_by_idx: dict[int, dict] = {}
    if session_log_path.exists():
        try:
            prior = json.loads(session_log_path.read_text(encoding="utf-8"))
            for s in prior.get("summaries", []):
                merged_summaries_by_idx[s["combo_index"]] = s
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"  WARNING: could not parse prior session_log.json ({exc}) -- "
                  f"starting a fresh merged log from this run's summaries only.")
    for s in session_summaries:
        merged_summaries_by_idx[s["combo_index"]] = s
    merged_summaries = [merged_summaries_by_idx[i] for i in sorted(merged_summaries_by_idx)]

    session_log_path.write_text(json.dumps({
        "recorded_at_iso": datetime.now(timezone.utc).isoformat(),
        "model_path": str(sg.MODEL_PATH),
        "n_requested": len(combinations),
        "n_captured_this_run": len(saved_paths),
        "n_skipped_existing": len(skipped_existing),
        "n_total_merged": len(merged_summaries),
        "elapsed_minutes": elapsed_min,
        "summaries": merged_summaries,
    }, indent=2), encoding="utf-8")
    print(f"Session log written -> {session_log_path} ({len(merged_summaries)} combos total "
          f"across all runs merged in)")


if __name__ == "__main__":
    main()
