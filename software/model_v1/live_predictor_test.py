"""Standalone live Force/Theta/Phi prediction from the eFlesh sensor stream.

Connects to the Teensy over serial (reusing EfleshController's port
discovery/handshake logic from eflesh_controller.py), and for every
incoming sensor frame runs the 15 raw S1X..S5Z magnetometer values from
EACH of the 4 mux groups (Group 1-4, i.e. Finger 1-4) independently
through LivePhiClassPredictor -- same model bundle, applied per-finger --
printing the predicted Force (N) and Theta (deg) as regression outputs,
Phi as a classified direction (-90/0/+90 deg), and Contact as a dedicated
no-contact classification once per frame, per finger.

Each finger has its own baseline calibration, its own hysteresis state,
and its own glitch guard, so activity or noise on one finger never
affects the reported state of another.

This is a standalone test script, separate from unified_gui.py — no GUI,
no recording, no plotting. Just connect, read, predict, print.

Usage:
    python live_predictor_test.py [--port COM7] [--baud 2000000]

Press Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_V1_DIR = SCRIPT_DIR

_model_v1_str = str(MODEL_V1_DIR)
if _model_v1_str not in sys.path:
    sys.path.insert(0, _model_v1_str)

from MagneticSensors.eflesh_controller import EfleshController, FrameData, NUM_GROUPS
from live_phiclass_predictor import LivePhiClassPredictor  # noqa: E402

MODEL_PATH = MODEL_V1_DIR / "active_model.pkl"


# One mux group per finger (Group 1..4 <-> Finger 1..4 in frame.groups[],
# 0-indexed). All 4 groups are read and predicted on independently -- see
# the per-finger state lists in main() (baselines, finger_states) and the
# frame_to_raw_values(..., group_idx) parameter below.
NUM_FINGERS = NUM_GROUPS  # 4

# How long to sample raw readings at startup (fingertip assumed untouched)
# to compute a per-channel baseline offset. Each finger's baseline is
# computed independently, from that finger's own samples only.
BASELINE_DURATION_S = 2.0

# Baseline is a per-session average of whatever raw S1X..S5Z readings come in
# during BASELINE_DURATION_S -- it has no unit-specific constants baked in,
# so it's recomputed fresh (and is equally valid) for whichever fingertip is
# physically connected when the script starts, not just the unit MODEL_PATH
# was trained/fine-tuned on. Below this many samples, the average is on too
# few points to be a reliable baseline (e.g. a slow/glitchy serial link) --
# still used, but flagged so it's not mistaken for a solid calibration.
MIN_BASELINE_SAMPLES = 10

# Temporal smoothing on the Contact classifier's raw per-frame output, as a
# two-threshold hysteresis (Schmitt trigger) rather than a single instant-
# reset counter -- see the CONTACT_RELEASE_FRAMES comment below for why a
# single confirm-only threshold isn't enough.
#
# CONTACT_CONFIRM_FRAMES: the displayed state only switches from "no
# contact" to "contact" after this many CONSECUTIVE frames all predict
# Contact=1, so a brief isolated false-positive spike on otherwise-idle data
# (seen on a previously-untested fingertip unit) reads as noise and doesn't
# flash "contact" on screen. Raise this if idle false-positive streaks are
# observed to last longer than the current value (they have been seen
# lasting 3+ frames on some units, which this threshold alone won't catch --
# it can only filter streaks strictly shorter than itself).
CONTACT_CONFIRM_FRAMES = 3

# CONTACT_RELEASE_FRAMES: once contact is confirmed, this many CONSECUTIVE
# Contact=0 frames are required before the display drops back to "no
# contact". NOT zero/instant, on purpose: the Contact classifier isn't
# perfectly clean even while genuinely pressed (isolated single-frame
# misclassifications happen mid-press), and an instant, undebounced release
# meant a single misclassified frame during a real press would wipe out the
# confirmed state and force a full fresh CONTACT_CONFIRM_FRAMES
# re-accumulation before the display would show real numbers again --
# starving the display of updates for most of a noisy press and only
# catching up once a clean run happened (often right at/after release, since
# that's often the strongest, least ambiguous signal in a press-and-release
# motion). Kept smaller than CONTACT_CONFIRM_FRAMES so genuine releases are
# still fast, just not zero-tolerance for a single noisy frame.
CONTACT_RELEASE_FRAMES = 2

# When True, prints one compact trace line EVERY frame (not throttled like
# the once-a-second [DEBUG] block) showing the raw Contact prediction, the
# hysteresis counters/state, and whether the display is showing live numbers
# or the no-contact override. Meant to be turned on to capture a full
# press-and-release cycle for offline diagnosis, then off again once done --
# it's much noisier than the normal ~20 Hz display output.
CONTACT_TRACE_ENABLED = True

# Sanity guard on the raw baseline-corrected delta, applied BEFORE the frame
# is fed to the model at all. Real recordings (idle or genuine presses) keep
# every one of the 15 delta channels within roughly +/-200 of baseline -- see
# build_delta_dataset.py's idle-row analysis and the STARTUP VERIFY/[DEBUG]
# output during normal operation. A delta this far outside that range (seen
# in practice: one channel's raw reading sign-flipping and sitting around
# +150000 for several seconds on a previously-untested unit) isn't a real
# touch, it's the magnetometer glitching -- most likely a marginal connection
# that only faults under the mechanical stress of a press. Feeding that to
# the model produces meaningless extrapolated garbage (Force in the
# thousands of N, Theta in the tens of thousands of degrees were observed),
# and the Contact classifier's response to such wildly out-of-distribution
# input is arbitrary, not meaningful. Frames that trip this guard are
# discarded before predict() is ever called -- see the glitch check at the
# top of process_frame().
SANE_DELTA_MAX = 2000.0

# The display loop (see main()) polls for the newest frame at this cadence
# instead of processing/printing every single frame the MCU sends. This is
# fast enough that a human can't perceive the difference from true
# per-frame updates, but slow enough that console I/O can never become a
# bottleneck for the serial reader thread -- see the on_raw_frame /
# process_frame split in main() for why that matters.
DISPLAY_INTERVAL_S = 0.05  # ~20 Hz

# --- Post-release display stability ----------------------------------------
# Both constants below only shape the printed "Force:" number in the display
# line -- they never touch result['Force'] as used for the [DEBUG]/[TRACE]/
# [STARTUP VERIFY] diagnostic lines (all of which print the true raw model
# output), nor the Contact classifier's hysteresis state (which is driven
# entirely by result['Contact'], never by Force at all -- see the
# CONTACT_CONFIRM_FRAMES/CONTACT_RELEASE_FRAMES comments above).

# Once contact has ended (display_contact is False), a displayed Force below
# this is snapped to exactly 0.0 instead of showing small residual jitter
# left over while FORCE_SMOOTHING_WINDOW's rolling average is still draining
# out the last real (nonzero) readings from before release.
FORCE_DEADBAND_N = 0.2

# Rolling-average window (in frames) applied ONLY to the printed Force
# number, to smooth out per-frame regression noise. Kept small so it
# doesn't introduce noticeable display lag.
FORCE_SMOOTHING_WINDOW = 4


def frame_to_raw_values(frame: FrameData, raw_cols: list[str], group_idx: int) -> dict:
    """Extract the 15 raw S1X..S5Z values for one mux group (one finger).

    group_idx is 0-indexed (Group 1/Finger 1 -> 0, ... Group 4/Finger 4 -> 3),
    matching frame.groups[]. Every caller passes a single, specific group_idx,
    so values from different fingers are never mixed within one call.
    """
    raw_values = {}
    for col in raw_cols:
        sensor_idx = int(col[1]) - 1   # "S3Y" -> sensor 3 -> index 2
        axis = col[2].lower()          # "S3Y" -> "y"
        reading = frame.groups[group_idx][sensor_idx]
        raw_values[col] = getattr(reading, axis)
    return raw_values


def log_group_status(frame: FrameData) -> None:
    """Print each mux group's signal status, using 1-indexed Group/Finger names."""
    for group_idx, sensors in enumerate(frame.groups):
        has_signal = any(s.x or s.y or s.z for s in sensors)
        status = "reading data" if has_signal else "all-zero (check wiring/connection)"
        print(f"  Group {group_idx + 1} (Finger {group_idx + 1}): {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live eFlesh force/direction prediction")
    parser.add_argument("--port", help="Teensy COM port (auto-discovers if omitted)")
    parser.add_argument("--baud", type=int, default=2_000_000, help="Serial baud rate")
    args = parser.parse_args()

    print(f"Model in use: {MODEL_PATH.name}  (full path: {MODEL_PATH})")
    print("Model structure: 3 learners sharing 1 StandardScaler -- "
          "MLPRegressor(Force, Theta) + MLPClassifier(Phi in {-90, 0, +90}) "
          "+ MLPClassifier(Contact in {0, 1})  [Contact is a dedicated "
          "no-contact detector, trained with balanced class weighting -- "
          "see model_v1/train_model_contact.py]")

    predictor = LivePhiClassPredictor(model_path=str(MODEL_PATH))
    # Confirm the scaler in use is the exact one saved inside the bundle
    # (not a mismatched/freshly-constructed one): an unfitted StandardScaler
    # has no mean_/scale_ and would raise NotFittedError below, so reaching
    # this print at all is itself part of the proof. input_cols confirms the
    # 20-feature order (15 delta + 5 magnitude) this scaler was fit on.
    print(f"Scaler check: {len(predictor.scaler.mean_)} features, "
          f"input_cols={predictor.input_cols}")
    print(f"  scaler.mean_[:3]={predictor.scaler.mean_[:3].tolist()}  "
          f"scaler.scale_[:3]={predictor.scaler.scale_[:3].tolist()}")
    print(f"Contact classifier confirmed loaded: classes={predictor.contact_classes}  "
          f"(0=no contact, 1=contact; trained threshold was Force >= "
          f"{predictor.contact_force_threshold}N, but the no-contact decision below "
          f"is driven by this classifier's prediction, not by that threshold)")
    # Shared across all 4 fingers (just bookkeeping for the once-per-connection
    # group-status log and the once-a-second [DEBUG] print cadence).
    shared_state = {
        "logged_group_status": False,
        "last_debug_print": 0.0,
    }
    # One independent state dict per finger (Finger 1..4 <-> index 0..3) --
    # each finger's hysteresis (consecutive_*_frames, confirmed_contact),
    # startup-print counter, glitch counter, and last-shown display line are
    # tracked separately here so nothing about one finger's contact state can
    # leak into another's.
    finger_states = [
        {
            "startup_frames_shown": 0,
            "consecutive_contact_frames": 0,
            "consecutive_no_contact_frames": 0,
            "confirmed_contact": False,
            "last_display_line": f"Finger {i + 1}: Force: {0.0:7.3f} N   "
                                  f"Theta: {0.0:8.2f} deg   "
                                  f"Phi: {'N/A (no contact)':>18s}",
            "glitch_frame_count": 0,
            # Display-only rolling buffer for FORCE_SMOOTHING_WINDOW -- feeds
            # only the printed Force number, never read by the hysteresis
            # counters above (which are driven by result['Contact'], not by
            # Force at all).
            "display_force_window": deque(maxlen=FORCE_SMOOTHING_WINDOW),
        }
        for i in range(NUM_FINGERS)
    ]
    # One independent list of raw calibration samples per finger, so each
    # finger's baseline (computed after the loop below) is that finger's own
    # average and is never mixed with another finger's readings.
    baseline_samples: list[list[dict]] = [[] for _ in range(NUM_FINGERS)]

    # --- Reader-thread / display-loop split -------------------------------
    # EfleshController runs its serial reader on a background thread (see
    # _parse_frames in eflesh_controller.py) and calls on_frame() SYNCHRONOUSLY,
    # IN THAT THREAD, once per parsed frame. That reader loop itself is fine
    # -- it reads all currently-available bytes each pass and only sleeps
    # when the OS buffer is genuinely empty, so it never blocks or waits
    # unnecessarily on its own. The problem was on our side: the old
    # on_frame body ran two MLP forward passes AND an unthrottled print()
    # of Force/Theta/Phi for EVERY frame, all inside that same reader
    # thread. Printing is slow (console I/O, often several ms/line on
    # Windows) relative to how often frames arrive, so the reader thread
    # spent most of its time blocked inside print() instead of draining
    # the serial port. Bytes then piled up in the OS-level serial receive
    # buffer, and each subsequent frame was parsed later and later behind
    # when it actually arrived -- a backlog that grows the longer the
    # script runs, which is exactly the "lag that gets worse over time"
    # symptom being fixed here.
    #
    # Fix: on_raw_frame (below), which IS the callback that actually runs
    # on the reader thread, does the absolute minimum -- stash a reference
    # to the newest frame under a lock and return immediately. All the real
    # work (predict() + printing) happens in process_frame(), called only
    # from the main thread's display loop at DISPLAY_INTERVAL_S. That loop
    # always grabs whichever frame is newest; if it ever falls behind, it
    # intentionally skips straight to the latest frame rather than working
    # through a backlog, so the printed values are always as close to the
    # physically-current sensor state as possible instead of catching up
    # through stale ones.
    frame_lock = threading.Lock()
    latest_frame_holder: dict[str, FrameData | None] = {"frame": None}

    def on_calibration_frame(frame: FrameData) -> None:
        if not shared_state["logged_group_status"]:
            print("Mux group status (first frame received):")
            log_group_status(frame)
            shared_state["logged_group_status"] = True
        for i in range(NUM_FINGERS):
            baseline_samples[i].append(frame_to_raw_values(frame, predictor.raw_cols, i))

    def on_raw_frame(frame: FrameData) -> None:
        # Runs on the serial reader thread -- must stay this cheap.
        with frame_lock:
            latest_frame_holder["frame"] = frame

    def process_frame(frame: FrameData) -> None:
        # Runs on the main thread, at DISPLAY_INTERVAL_S cadence, never on
        # the serial reader thread.
        #
        # force_direction_model_v1_contact.pkl was trained on
        # baseline-subtracted delta features (dS1X..dS5Z + S1_mag..S5_mag
        # computed from those deltas) -- see model_v1/train_model_contact.py
        # -- so feeding raw_delta below (raw minus this session's baseline)
        # is the matched, correct input. Unlike the older v3 hysteresis
        # predictor, there's no ForceDirection feature and therefore no
        # bootstrap/two-pass step: every input is derivable from the current
        # frame alone.
        #
        # The [DEBUG] block's once-a-second cadence is shared across all 4
        # fingers (one gate, computed once per call) so the four fingers'
        # debug output for a given tick stays grouped together in the log
        # instead of drifting out of sync with each other.
        now = time.time()
        show_debug = now - shared_state["last_debug_print"] >= 1.0
        if show_debug:
            shared_state["last_debug_print"] = now
            # Direct, empirical lag measurement: frame.timestamp was set in
            # eflesh_controller.py the instant the reader thread finished
            # parsing this frame (shared across fingers -- it's a property of
            # the frame, not of any one group). If this stays small and
            # bounded (not growing run over run), the reader thread is
            # keeping up and there's no backlog; if it climbs steadily, bytes
            # are still backing up somewhere upstream of here.
            staleness_ms = (now - frame.timestamp) * 1000
            print(f"[DEBUG] frame seq={frame.seq}  staleness: {staleness_ms:.1f} ms   "
                  f"serial stats: frames={ctrl.frame_count} dropped={ctrl.dropped} "
                  f"sync_losses={ctrl.sync_losses} parse_errors={ctrl.parse_errors}")

        display_lines = []

        # Each finger (mux group) is processed as a fully independent pass:
        # its own baseline, its own glitch guard, its own predict() call, and
        # its own hysteresis state (finger_states[i]) -- nothing computed for
        # one finger is read by, or written into, another finger's state.
        for i in range(NUM_FINGERS):
            label = f"Finger {i + 1}"
            fstate = finger_states[i]
            baseline_i = baselines[i]

            raw_absolute = frame_to_raw_values(frame, predictor.raw_cols, i)
            raw_delta = {col: v - baseline_i[col] for col, v in raw_absolute.items()}

            # Sensor-glitch guard: if any channel is wildly outside the range
            # real (idle or pressed) data ever occupies, this finger's frame
            # is discarded before it ever reaches the model -- see the
            # SANE_DELTA_MAX comment for why. That finger's displayed state
            # is left exactly as it was (last known good line for THIS
            # finger), not reset or overridden to zero, since we have no
            # reason to believe its contact status actually changed -- we
            # just don't trust this frame's reading. Its hysteresis counters/
            # confirmed_contact are also left untouched, so a burst of
            # glitched frames can't itself trigger or cancel a confirm/
            # release transition. Applied independently per finger: a glitch
            # on one finger doesn't affect, skip, or discard the others --
            # the loop just moves on to the next finger below.
            glitched_cols = {
                col: val for col, val in raw_delta.items() if abs(val) > SANE_DELTA_MAX
            }
            if glitched_cols:
                fstate["glitch_frame_count"] += 1
                print(f"[GLITCH][{label}] frame seq={frame.seq} discarded (not fed to model): "
                      + ", ".join(f"{col}={val:.2f}" for col, val in glitched_cols.items())
                      + f"  (sane range is +/-{SANE_DELTA_MAX:.0f}; "
                      f"total glitched frames this finger this session: "
                      f"{fstate['glitch_frame_count']})")
                display_lines.append(fstate["last_display_line"])
                continue

            result = predictor.predict(raw_delta)

            # Startup verification: for the first few displayed frames after
            # baseline capture, print raw vs. baseline-corrected side by
            # side, plus the full three-part prediction breakdown (Force
            # regression, Theta regression, Phi classification incl.
            # per-class probabilities), so the new model structure is
            # visibly confirmed working, not just assumed. Tracked
            # per-finger (fstate["startup_frames_shown"]) since fingers can
            # start glitching/settling at different times.
            if fstate["startup_frames_shown"] < 5:
                fstate["startup_frames_shown"] += 1
                print(f"[STARTUP VERIFY][{label} {fstate['startup_frames_shown']}/5] "
                      f"raw vs. baseline-corrected (fed to model):")
                for col in predictor.raw_cols:
                    print(f"    {col}: raw={raw_absolute[col]:8.2f}   "
                          f"corrected={raw_delta[col]:8.2f}   "
                          f"(baseline={baseline_i[col]:8.2f})")
                # Same formula/order as _build_feature_row in live_phiclass_predictor.py
                # (sqrt of sum-of-squares of the delta triplet per sensor) so what's
                # printed here is exactly what's fed to predictor.scaler.transform().
                magnitudes = {
                    f"S{j+1}_mag": sum(raw_delta[c] ** 2 for c in triplet) ** 0.5
                    for j, triplet in enumerate(predictor.sensor_triplets)
                }
                print(f"    [{label}] computed magnitudes (fed to model): "
                      + ", ".join(f"{col}={val:.2f}" for col, val in magnitudes.items()))
                features = predictor._build_feature_row(raw_delta)
                features_scaled = predictor.scaler.transform(features)
                print(f"    [{label}] scaled feature vector (post-StandardScaler, actual model input): "
                      + ", ".join(f"{col}={val:.3f}"
                                  for col, val in zip(predictor.input_cols, features_scaled[0])))
                probs_str = ", ".join(f"{cls:+d}deg={p:.1%}" for cls, p in result["Phi_probs"].items())
                contact_probs_str = ", ".join(
                    f"{cls}={p:.1%}" for cls, p in result["Contact_probs"].items())
                print(f"    [{label}] Prediction breakdown: "
                      f"Force (regression)={result['Force']:.3f} N   "
                      f"Theta (regression)={result['Theta']:.2f} deg   "
                      f"Phi (classification)={result['Phi']:+d} deg  [{probs_str}]   "
                      f"Contact (classification)={result['Contact']} "
                      f"[{contact_probs_str}]")

            if show_debug:
                print(f"[DEBUG][{label}] raw (pre-baseline):  "
                      + ", ".join(f"{col}={raw_absolute[col]:.2f}" for col in predictor.raw_cols))
                print(f"[DEBUG][{label}] raw (post-baseline, fed to model): "
                      + ", ".join(f"{col}={raw_delta[col]:.2f}" for col in predictor.raw_cols))
                print(f"[DEBUG][{label}] Force prediction: {result['Force']:.3f} N   "
                      f"Contact prediction: {result['Contact']} "
                      f"(probs: {result['Contact_probs']})")

            # Temporal smoothing on the raw per-frame Contact prediction, as a
            # two-threshold hysteresis: "confirmed_contact" is a PERSISTENT state
            # (survives across frames) that only flips on a sustained run in
            # either direction, not an instantaneous streak-length test. This
            # matters because the raw classifier isn't perfectly clean even
            # while genuinely pressed -- an earlier version reset the confirm
            # counter to 0 on any single Contact=0 frame, so one misclassified
            # frame mid-press would force a full fresh CONTACT_CONFIRM_FRAMES
            # re-accumulation before the display showed real numbers again,
            # which could starve the display of updates for most of a press.
            # predictor.predict()'s raw result stays untouched (used as-is above
            # for the startup/debug prints); only this display decision is
            # smoothed. fstate is this finger's own dict, so this hysteresis
            # transition can never be triggered or suppressed by another
            # finger's Contact prediction.
            if result["Contact"] == 1:
                fstate["consecutive_contact_frames"] += 1
                fstate["consecutive_no_contact_frames"] = 0
            else:
                fstate["consecutive_no_contact_frames"] += 1
                fstate["consecutive_contact_frames"] = 0

            if not fstate["confirmed_contact"]:
                if fstate["consecutive_contact_frames"] >= CONTACT_CONFIRM_FRAMES:
                    fstate["confirmed_contact"] = True
            else:
                if fstate["consecutive_no_contact_frames"] >= CONTACT_RELEASE_FRAMES:
                    fstate["confirmed_contact"] = False

            display_contact = fstate["confirmed_contact"]

            if CONTACT_TRACE_ENABLED:
                # One line per frame per finger, everything needed to
                # reconstruct that finger's hysteresis state machine
                # behavior offline: the raw per-frame classifier output (and
                # its confidence), the two streak counters that feed it, the
                # persistent confirmed_contact state, and which branch the
                # display below actually takes. flush=True so the line
                # appears immediately even if stdout is being redirected/
                # piped while capturing a press-and-release cycle, instead of
                # sitting in a buffer until the process exits (which would
                # itself look exactly like "nothing updates until the end"
                # and confound this capture).
                contact_probs = result["Contact_probs"]
                probs_str = ", ".join(f"{cls}={p:.2f}" for cls, p in contact_probs.items())
                print(f"[TRACE][{label}] seq={frame.seq}  raw_Contact={result['Contact']} ({probs_str})  "
                      f"streak_contact={fstate['consecutive_contact_frames']}  "
                      f"streak_no_contact={fstate['consecutive_no_contact_frames']}  "
                      f"confirmed_contact={fstate['confirmed_contact']}  "
                      f"display={'LIVE' if display_contact else 'OVERRIDE'}  "
                      f"raw_Force={result['Force']:.3f}N  raw_Theta={result['Theta']:.2f}deg  "
                      f"raw_Phi={result['Phi']:+d}deg",
                      flush=True)

            # raw_force_now is the same Force-floored, contact-gated value
            # used everywhere else (STARTUP VERIFY/[DEBUG]/[TRACE] all print
            # result['Force'] directly, and hysteresis is driven only by
            # result['Contact'], not by Force) -- so nothing but the printed
            # "Force:" number below is affected by the smoothing/deadband
            # step that follows. Display-level floor only: Force is a
            # physical magnitude and can never legitimately be negative, but
            # the regressor is an unconstrained MLP and occasionally outputs
            # a small negative value (e.g. -0.05N) as ordinary residual
            # noise near zero force.
            raw_force_now = max(0.0, result["Force"]) if display_contact else 0.0

            # Display-only smoothing + settling deadband -- see
            # FORCE_SMOOTHING_WINDOW/FORCE_DEADBAND_N above. Neither touches
            # raw_force_now itself, only the number that gets printed.
            fstate["display_force_window"].append(raw_force_now)
            smoothed_force = (sum(fstate["display_force_window"])
                               / len(fstate["display_force_window"]))
            if not display_contact and smoothed_force < FORCE_DEADBAND_N:
                shown_force = 0.0
            else:
                shown_force = smoothed_force

            # No-contact override: gated on the smoothed Contact classifier
            # output, not on a Force threshold -- the Force regressor's own
            # low-force resolution is coarse and Theta is a leaky recording-level
            # label at idle, which is exactly why force_direction_model_v1_contact.pkl
            # has a separate classifier trained specifically to recognise "no
            # contact" (see model_v1/train_model_contact.py). When it says no
            # contact, Theta/Phi are not physically meaningful, so we override
            # the display instead of trusting the raw regression/classification
            # output.
            if not display_contact:
                display_line = (f"{label}: Force: {shown_force:7.3f} N   "
                                 f"Theta: {0.0:8.2f} deg   "
                                 f"Phi: {'N/A (no contact)':>18s}")
            else:
                phi_probs = result["Phi_probs"]
                probs_str = ", ".join(f"{cls:+d}deg={p:.0%}" for cls, p in phi_probs.items())
                display_line = (f"{label}: Force: {shown_force:7.3f} N   "
                                 f"Theta: {result['Theta']:8.2f} deg   "
                                 f"Phi: {result['Phi']:+4d} deg   "
                                 f"(probs: {probs_str})")
            # Cached so a subsequent glitched frame for THIS finger (see the
            # guard above) can keep showing this instead of running the
            # model on garbage input -- other fingers' cached lines are
            # untouched.
            fstate["last_display_line"] = display_line
            display_lines.append(display_line)

        # All 4 fingers' current status, one line each, printed together as
        # a single block so the display reads as one coherent snapshot per
        # frame tick rather than four lines interleaved with other output.
        for line in display_lines:
            print(line)

    ctrl = EfleshController()
    ctrl.on_status = print
    ctrl.on_error = print
    ctrl.on_connected = lambda port: print(f"Connected on {port}")
    ctrl.on_disconnected = lambda: print("Disconnected.")

    print("Connecting to eFlesh sensor array (Ctrl+C to stop)...")
    ctrl.on_frame = on_calibration_frame
    ctrl.start(port=args.port, baud=args.baud)

    try:
        print(f"Calibrating baseline ({BASELINE_DURATION_S:.0f}s, keep fingertip untouched)...")
        calibration_deadline = time.time() + BASELINE_DURATION_S
        while time.time() < calibration_deadline:
            time.sleep(0.05)

        # Each finger's baseline is computed independently, from that
        # finger's own baseline_samples[i] list only -- never averaged or
        # mixed across fingers -- so a weak/failed calibration on one finger
        # (e.g. that mux group unplugged) doesn't skew another finger's
        # baseline.
        baselines: list[dict] = []
        for i in range(NUM_FINGERS):
            label = f"Finger {i + 1}"
            samples = baseline_samples[i]
            if samples:
                baseline_i = {
                    col: sum(sample[col] for sample in samples) / len(samples)
                    for col in predictor.raw_cols
                }
            else:
                print(f"Warning: {label}: no frames received during calibration; "
                      f"using zero baseline.")
                baseline_i = {col: 0.0 for col in predictor.raw_cols}
            baselines.append(baseline_i)

            print(f"{label} baseline ({len(samples)} samples): "
                  + ", ".join(f"{col}={baseline_i[col]:.2f}" for col in predictor.raw_cols))

            if not samples:
                print(f"{label}: baseline calibration FAILED (0 samples) -- no-contact "
                      f"debug deltas will be meaningless until reconnected.")
            elif len(samples) < MIN_BASELINE_SAMPLES:
                print(f"{label}: baseline calibration WEAK: only {len(samples)} samples "
                      f"(< {MIN_BASELINE_SAMPLES}) -- consider restarting the script if "
                      f"readings look off.")
            else:
                print(f"{label}: baseline calibration confirmed: {len(samples)} samples "
                      f"captured fresh this run, over {BASELINE_DURATION_S:.0f}s, for "
                      f"whichever fingertip is connected right now -- valid regardless of "
                      f"which unit '{MODEL_PATH.name}' was originally trained/fine-tuned on.")
        print(f"Active model: {MODEL_PATH.name}  |  "
              + ", ".join(f"Finger {i + 1}: {len(baseline_samples[i])} samples"
                           for i in range(NUM_FINGERS)))

        # Switch the reader thread over to the cheap stash-only callback,
        # then drive prediction + printing from this (main) thread instead,
        # at a fixed display cadence -- see the comment above on_raw_frame /
        # process_frame for why this split exists.
        ctrl.on_frame = on_raw_frame
        print(f"Display loop running at ~{1 / DISPLAY_INTERVAL_S:.0f} Hz "
              f"(always shows the newest frame; older frames are dropped, "
              f"not queued, so lag can't accumulate over the session).")

        last_seq_shown = None
        while True:
            time.sleep(DISPLAY_INTERVAL_S)
            with frame_lock:
                frame = latest_frame_holder["frame"]
            if frame is None or frame.seq == last_seq_shown:
                continue
            last_seq_shown = frame.seq
            process_frame(frame)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ctrl.stop()


if __name__ == "__main__":
    main()
