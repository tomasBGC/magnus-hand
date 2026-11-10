"""Standalone live 4-finger tactile-monitoring GUI for the eFlesh hand.

Separate from unified_gui.py (characterisation/recording tool) -- this is a
lightweight demo/monitoring GUI only: no recording, no CSV/PNG export, no
theta/phi/face metadata inputs. It reuses the exact validated backend from
live_predictor_test.py:
  - frame_to_raw_values(frame, raw_cols, group_idx) for reading one finger's
    15 raw S1X..S5Z values out of a frame (see that function's docstring for
    why frame.groups[group_idx] can never mix data between fingers)
  - the same per-finger baseline calibration, delta-feature computation,
    force_direction_model_v1_contact.pkl prediction, hysteresis (confirm/
    release) contact smoothing, glitch guard (SANE_DELTA_MAX), and Force
    floor (max(0.0, Force)) -- reimplemented here as _process_one_finger()
    using the SAME imported constants/thresholds as the console tool, so
    both tools agree on what counts as a glitch, a confirmed contact, etc.

Each finger has its own FingerState (baseline, hysteresis counters, glitch
count, rolling force history, slip-hold timer) -- nothing here is shared
between the 4 finger_states[i] entries, matching live_predictor_test.py's
same requirement.

Serial reads never block the GUI thread: EfleshController's reader thread
only ever stashes the newest frame under a lock (HandSensingEngine.
_on_raw_frame), exactly like live_predictor_test.py's on_raw_frame. A
QTimer in the GUI thread polls at DISPLAY_INTERVAL_S and always processes
only the latest frame, dropping (not queueing) anything older, so a slow
GUI repaint can never build a backlog against the serial port.

Each finger's baseline is also periodically refreshed in the background
(RECALIBRATION_INTERVAL_S) against its own current resting readings, only
while that finger has been stably confirmed no-contact for a few seconds
(RECALIBRATION_MIN_STABLE_NO_CONTACT_S) -- see _process_one_finger. This
targets baseline drift over a long session (thermal, etc.), which live
[DIAG] toggle-rate data showed as the actual driver behind Force flicker
that worsened the longer a session ran, distinct from the display-only
smoothing/deadband and hysteresis-threshold tuning above it.

Usage:
    python sensing_gui.py [--port COM7] [--baud 2000000]
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime as _dt
from pathlib import Path

import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from PyQt6 import QtCore, QtGui, QtWidgets

from MagneticSensors.eflesh_controller import EfleshController, FrameData
from live_phiclass_predictor import LivePhiClassPredictor
from live_predictor_test import (
    BASELINE_DURATION_S,
    DISPLAY_INTERVAL_S,
    MIN_BASELINE_SAMPLES,
    MODEL_PATH,
    NUM_FINGERS,
    SANE_DELTA_MAX,
    frame_to_raw_values,
)

# Per-finger Force calibrators (2026-09-07): post-hoc isotonic correction of
# the Force regressor's raw output, fit against the load-cell grasping
# protocol's independent ground truth (grasping_dashboard/retrain_contact_
# with_load_cell_data.py's sibling force-calibration analysis). Only a
# finger with a real, validated monotonic relationship between raw Force and
# true force gets an entry here -- checked via a rep-grouped holdout, same
# rigor as every other model change in this project. Ring and Index showed
# essentially no correlation or an inverted one respectively (self-motion
# noise dominating their raw signal) -- a calibrator for either would
# launder noise into a falsely-confident number rather than fix anything,
# so they deliberately have no entry. Missing file / label -> no calibrator
# applied, byte-identical to pre-2026-09-07 behaviour for that finger.
FORCE_CALIBRATOR_PATHS: dict[str, Path] = {
    # DISABLED 2026-09-09 for the dissertation demo: this isotonic
    # calibrator's out_of_bounds='clip' behavior hard-caps Thumb's
    # CALIBRATED Force at 1.7681N (the highest true load-cell force it ever
    # saw fit -- confirmed directly: raw inputs from 1.5N up to 10N all
    # collapsed to the identical 1.7681N output), which made the demo's
    # 2.0N "firm" Grip-to-Target target unreachable for Thumb specifically
    # (it would report "safety max reached" instead of "target reached").
    # Removing this entry reverts Thumb to its pre-2026-09-07 UNCALIBRATED
    # Force output -- not just uncapped above 1.77N, but also without
    # whatever correction the calibrator was applying in the 0-1.77N range,
    # so Thumb's Force number is less trustworthy there than with the
    # calibrator active, not just less capped. Re-enable by uncommenting
    # once the calibrator is refit on data that reaches a higher true
    # Thumb force than the current 143-rep set did.
    # "Thumb": SCRIPT_DIR / "grasping_dashboard" / "thumb_force_calibrator_candidate.pkl",
}


def _load_force_calibrators() -> dict[str, object]:
    calibrators = {}
    for label, path in FORCE_CALIBRATOR_PATHS.items():
        if path.exists():
            calibrators[label] = joblib.load(path)
    return calibrators


FORCE_CALIBRATORS: dict[str, object] = _load_force_calibrators()

# model_v1/crosstalk/ is a separate, additive pipeline (see its own
# README.md) and not a package -- added to sys.path here, the same way
# SCRIPT_DIR itself is added above, so its modules import cleanly without
# copying or modifying anything in that folder. crosstalk_model.py's
# load_model()/predict_shift() are used completely unmodified below (see
# CROSSTALK_CORRECTION_ENABLED and HandSensingEngine.__init__).
_CROSSTALK_DIR = SCRIPT_DIR / "crosstalk"
if str(_CROSSTALK_DIR) not in sys.path:
    sys.path.insert(0, str(_CROSSTALK_DIR))
from crosstalk_model import load_model as load_crosstalk_model  # noqa: E402
from crosstalk_model import predict_shift as crosstalk_predict_shift  # noqa: E402

# model_v1/crosstalk_lookup/ (see its own README.md) -- directly-measured
# nearest-neighbour crosstalk correction, an alternative to crosstalk_model
# above. Same "separate, additive pipeline, imported unmodified" pattern as
# _CROSSTALK_DIR. nn_lookup_model.predict_shift() has the exact same
# (model, affected_label, source_positions) -> {channel: shift} signature
# as crosstalk_model.predict_shift() by design, so both can sit behind the
# same call site -- see CROSSTALK_CORRECTION_SOURCE/CROSSTALK_CORRECTION_
# BACKENDS below.
_CROSSTALK_LOOKUP_DIR = SCRIPT_DIR / "crosstalk_lookup"
if str(_CROSSTALK_LOOKUP_DIR) not in sys.path:
    sys.path.insert(0, str(_CROSSTALK_LOOKUP_DIR))
from nn_lookup_model import load_model as load_nn_lookup_model  # noqa: E402
from nn_lookup_model import predict_shift as nn_lookup_predict_shift  # noqa: E402
from nn_lookup_model import predict_shift_smoothed as nn_lookup_predict_shift_smoothed  # noqa: E402

# model_v1/crosstalk_lookup/ridge_crosstalk_model.py (2026-07-17, production
# review) -- origin-anchored quadratic-ridge backend fitted on the 100
# directly-measured simultaneous-4-finger combinations (self-effect-corrected
# target), see model_v1/production_review/07_crosstalk_ridge_report.md. Same
# "separate, additive pipeline, imported unmodified" pattern and the same
# predict_shift(model, affected_label, source_positions) -> {channel: shift}
# contract as the two backends above (_CROSSTALK_LOOKUP_DIR is already on
# sys.path at this point).
from ridge_crosstalk_model import load_model as load_ridge_crosstalk_model  # noqa: E402
from ridge_crosstalk_model import predict_shift as ridge_crosstalk_predict_shift  # noqa: E402

# model_v1/crosstalk_lookup/magnetfree_index_model.py (2026-08-30) -- fifth
# backend, Index-only. Fitted from a MAGNET-FREE reference fingertip
# physically mounted in the Index position (no self-effect possible, unlike
# every backend above, which was fitted/measured with a normal magnet-
# equipped Index) -- see magnetfree_index_disturbance/report.md. Offline
# LOOCV found it beats the currently-deployed nearest_neighbor_smoothed by
# 1.5-2.9x lower MAE per finger (see magnetfree_index_disturbance/
# model_validation_report.md) and models the confirmed SUB-additive
# combination behaviour via a fitted alpha scalar rather than assuming
# simple summation. Same predict_shift(model, affected_label,
# source_positions) -> {channel: shift} contract, EXCEPT it raises if
# affected_label != "Index" (this model has no Middle/Ring/Thumb data --
# see that module's own docstring). Registered as an AVAILABLE backend only
# -- NOT CROSSTALK_CORRECTION_SOURCE's deployed default for Index. The live
# counterbalanced A/B this required (live_ab_grasp_test.py, real Index+Thumb
# and Index+Middle grasps, 5 reps/config/scenario, see
# magnetfree_index_disturbance/live_ab_grasp_report.md) did NOT show a clear
# live win despite the offline LOOCV improvement above -- same documented
# offline-MAE-mispredicts-live-behaviour risk this codebase already hit once
# with plain "nearest_neighbor" (better offline MAE, worse live false-contact
# rate, see that backend's own comment below). Kept registered/available,
# same posture as "ridge_quadratic", pending either more live reps or a live
# metric that isolates crosstalk quality from the separate Contact-classifier
# and SANE_DELTA_MAX-glitch confounds that report identifies.
from magnetfree_index_model import load_model as load_magnetfree_index_model  # noqa: E402
from magnetfree_index_model import predict_shift as magnetfree_index_predict_shift  # noqa: E402

# ============================================================================
# GUI-specific constants (all easy to retune independently of the console
# tool's constants, which are imported above and left untouched)
# ============================================================================

# Declared as their own GUI-local constants (originally seeded from the
# console tool's CONTACT_CONFIRM_FRAMES/CONTACT_RELEASE_FRAMES, 3/2) rather
# than importing those directly, so they can be retuned for whatever noise/
# raw-Contact-toggle-rate this GUI actually observes (see
# contact_toggle_rate_hz in FingerResult/_process_one_finger below)
# independently of live_predictor_test.py's own tuning.
#
# Raised from that original 3/2 (150ms/100ms @ 20Hz) after a live
# 4-finger [DIAG] session (~25s, untouched hand) showed the raw Contact
# classifier throwing isolated 1-2 frame misfires on EVERY finger -- 1 to 6
# "streak-collapse" events each (a long-confirmed streak like 2026 frames
# dropping straight to single digits from one stray raw_Contact=0 reading).
# With CONTACT_RELEASE_FRAMES=2, two such misfires in a row was enough to
# actually flip confirmed_contact off and zero the displayed Force, which is
# what produced the visible flicker. 5/5 (250ms each way) absorbs that
# isolated-blip pattern while staying fast enough not to meaningfully delay
# a genuine quick tap.
#
# This is a mitigation for that specific mechanism, not a fix for the
# session's other finding: Finger 1/2 were perfectly stable (toggle rate
# 0.00/s) for roughly the first two-thirds of that session and only started
# toggling in the final third, while Finger 4 was noisiest mid-session and
# calmer by the end -- a time-varying pattern across previously-independent
# channels that points at a systemic/drifting noise source (thermal, power,
# or a stale baseline), not per-channel classifier noise alone. Loosening
# these thresholds further would just mask that instead of fixing it -- if
# flicker persists after this change, look at power/thermal stability or a
# mid-session baseline recalibration before raising these any further.
#
# 2026-07-20 (evening): CONTACT_CONFIRM_FRAMES only lowered 5 -> 4 (see
# true_zero_dataset/select_confirm_frames_report.md). CONTACT_RELEASE_
# FRAMES deliberately left at 5, untouched -- the flicker mechanism above
# is specifically about a short RELEASE count letting isolated misfires
# flip an already-confirmed streak off, not about the CONFIRM count, so
# shortening CONFIRM alone doesn't reopen it. Motivation: self_motion_
# report.md's finding that Ring/Thumb show elevated Force noise WHILE
# MOVING can break a genuine touch's streak mid-accumulation (a noisy dip
# resets consecutive_contact_frames before it reaches the requirement),
# reading as reduced sensitivity specifically during a grip motion, even
# with CONTACT_SUPPRESS_DURING_OWN_TRANSIT disabled. Swept CONFIRM_FRAMES
# in {5,4,3,2} against ~75 pooled no-touch finger-minutes (both overall
# and continuous_movement_stream.json's real-transit data in isolation)
# plus real touch data for latency: found a genuine CLIFF at 2 (false-
# accepts jump ~10x overall, ~13x during movement specifically -- exactly
# where the noise this change targets lives, confirming the risk is real)
# with a corrupted-looking touch-confirm-rate (85.7%, latency WORSENING
# again vs 3, i.e. noise contamination, not real sensitivity). 4 is the
# well-supported small step: modest risk (10->14 false episodes, 2->3
# during movement, still tiny in absolute terms) for a genuine, clean
# latency improvement (1.42s->1.26s, confirm rate 66.7%->68.2%) without
# touching the cliff. 3 was available but rejected -- bigger risk jump
# (33 episodes, 3x baseline) for only marginally better latency (1.11s)
# than 4 already gives. Revert via sensing_gui_BACKUP_pre_confirm_
# frames_4_*.py if needed.
CONTACT_CONFIRM_FRAMES = 4
CONTACT_RELEASE_FRAMES = 5

# --- Contact-probability Schmitt trigger (2026-07-17, production review) ----
# Replaces the hard predict()==1 signal feeding the confirm/release streaks
# with two thresholds on the Contact classifier's own probability
# (result["Contact_probs"][1]): a frame counts toward the CONFIRM streak at
# P >= CONTACT_P_CONFIRM (plus the existing MIN_CONTACT_FORCE_N gate below,
# unchanged), counts toward the RELEASE streak at P <= CONTACT_P_RELEASE,
# and in the dead zone between the two thresholds leaves BOTH streak
# counters untouched -- holding state there is the point of a Schmitt
# trigger. Validated offline in model_v1/production_review/
# 04_calibration_report.md: on held-out Units-1+2 test rows this cuts idle
# frames feeding the confirm streak by 28% (0.0633 -> 0.0454) AND mid-press
# release pressure by 29% (0.1277 -> 0.0908) simultaneously; the cost is a
# 2-6pp confirm-sensitivity loss in the already-weak <1N light-touch band
# (see that report's addendum table -- a stricter P_hi=0.8 "calm mode"
# exists but costs 13-17pp there, so it is documented, not default).
# raw_contact / toggle-rate diagnostics stay based on the classifier's hard
# output, ungated, exactly as before. Flip the flag to False to restore the
# exact pre-review streak behaviour without deleting code (same
# "revert without deleting code" pattern as CROSSTALK_CORRECTION_ENABLED).
# RE-ENABLED (2026-07-17) after a live diagnosis session: temporarily set to
# False to test whether the dead zone itself was letting Index's
# stuck-confirmed issue (see candidate_v2/README.md's "Live test findings")
# persist. RULED OUT -- disabling it entirely (byte-identical to the old
# single-threshold hysteresis) did not change Index's behaviour at all. Root
# cause traced instead to a pre-existing (not introduced by this review)
# architectural gap: BOTH baseline-tracking modes (this one and
# "stable_window") only update a finger's baseline while it is NOT
# confirmed-contact, so a brief transient false-confirm (observed: a 2-tick,
# 4-5Hz raw_Contact toggle burst timed to a multi-finger servo move, which
# happened to satisfy the confirm streak) permanently freezes that finger's
# baseline the instant it locks in -- there is no existing mechanism in
# either mode to self-correct afterward, only a fresh startup calibration
# (restarting the app) clears it. This is a genuine, separate finding, not
# something Steps 1-5 caused or fix -- see README.md for the full writeup.
CONTACT_PROB_SCHMITT_ENABLED = True
# 2026-07-20: lowered from 0.6/0.3 -- a small, conservative sensitivity bump
# for the true-zero classifier's light-touch confirm rate (see
# true_zero_dataset/select_thresholds_consolidated_report.md). Chosen as the
# SMALLEST step down from a 5-candidate sweep (0.60 down to 0.40) run
# against ~75 pooled finger-minutes of genuinely-no-touch data from 5
# separate recordings (idle, a static high-crosstalk pose, transit/movement,
# and two real freeform-touching sessions' untouched fingers) -- risk grows
# monotonically below 0.60/0.30 in that broader pool (unlike the narrower
# original idle-only sweep, which read as flat zero risk all the way to
# 0.15/0.05 -- the value candidate_v4 tried and had to revert after an
# unexplained live regression, see true-zero-threshold-investigation
# memory), so this step was deliberately kept small rather than jumping
# further down that same sweep. At 0.55/0.28: 6 false-confirm episodes
# (4.9s total, ~0.11% of the pooled no-touch time) vs 0 at 0.60/0.30, in
# exchange for a measured +1.7 to +2.3 percentage-point live confirm-rate
# improvement across the [0.3,0.5)/[0.5,0.7)/[0.7,1.0)N light-touch bands
# (real touch data, not the offline jig set). Revert to 0.6/0.3 by
# restoring model_v1/true_zero_dataset/sensing_gui_BACKUP_pre_0.55_0.28_
# threshold_*.py if this proves too sensitive live.
#
# 2026-07-20 (same evening): nudged again, 0.55/0.28 -> 0.53/0.27, after
# live use at 0.55/0.28 confirmed the first bump was a genuine improvement
# and the user asked for "a tiny bit more". Same consolidated-sweep method,
# re-run anchored at 0.55/0.28 as the new baseline: 0.53/0.27 is the
# smallest next step -- false-confirm exposure barely moves (7 episodes,
# 6.2s, ~0.14% of the same ~75 pooled no-touch finger-minutes, vs 6
# episodes/4.9s/~0.11% at 0.55/0.28), for a further +0.4 to +0.5 percentage
# -point live confirm-rate gain across the same 3 light-touch bands. Revert
# to 0.55/0.28 via sensing_gui_BACKUP_pre_0.53_0.27_threshold_*.py, or all
# the way to 0.6/0.3 via the _pre_0.55_0.28_ backup above, if needed.
#
# 2026-07-20 (same evening): third nudge, 0.53/0.27 -> 0.52/0.265. NOTE:
# this step shows clearly DIMINISHING RETURNS vs the previous two -- re-run
# consolidated sweep anchored at 0.53/0.27 found false-confirm exposure
# grows faster here (7->9 episodes, 6.2s->9.2s, ~0.14%->~0.20% of the same
# ~75 pooled no-touch finger-minutes) for a much smaller live confirm-rate
# gain (+0.2 to +0.5pp, vs +0.4-0.5pp for the PREVIOUS, smaller-risk step)
# -- going one step further to 0.51/0.26 was checked and found to add risk
# (12.4s vs 9.2s) for ZERO additional recall benefit (identical confirm
# rates to 0.52/0.265 in all 3 bands), so that was rejected in favour of
# this one. If more sensitivity is requested again, re-run the sweep fresh
# rather than continuing to step down by the same increment -- the
# risk/benefit ratio is worsening, this may be close to a practical floor
# for threshold-only tuning. Revert to 0.53/0.27 via sensing_gui_BACKUP_
# pre_0.52_0.265_threshold_*.py if needed.
#
# 2026-07-20 (same evening): fourth nudge, 0.52/0.265 -> 0.50/0.25.
# CORRECTION to the "practical floor" note above: that was based on too
# fine a step (0.52->0.51) where the pooled sample is small enough for
# single-episode noise to look like a plateau. A WIDER re-sweep (0.52 down
# to 0.35) shows recall keeps climbing steadily, not flat -- 0.272/0.545/
# 0.783 @0.52/0.265 -> 0.275/0.551/0.788 @0.50/0.25 -> 0.292/0.563/0.812
# @0.45/0.22 -- so there IS real remaining headroom, just at a
# decelerating rate per unit of threshold change. Picked 0.50/0.25 as a
# genuinely modest next step (false-confirm exposure 9.2s->14.4s, ~0.20%->
# ~0.32% of the same ~75 pooled no-touch finger-minutes -- still small in
# absolute terms). If asked for more again, re-run select_thresholds_
# consolidated.py anchored here with a WIDE candidate spread (not tiny
# 0.01 steps, which read as noisier plateau than the underlying trend
# actually is) before concluding there's no room left. Revert to
# 0.52/0.265 via sensing_gui_BACKUP_pre_0.50_0.25_threshold_*.py if needed.
#
# 2026-07-20 (same evening): fifth nudge, 0.50/0.25 -> 0.40/0.20 -- a
# bigger step than the previous few, per explicit request (no longer just
# "a tiny bit"). Re-swept (recall recomputed at the NOW-current
# CONTACT_CONFIRM_FRAMES=4, not the old 5, since that also affects the
# baseline recall figures): 0.318/0.596/0.838 @0.40/0.20 vs 0.280/0.560/
# 0.805 @0.50/0.25 across the 3 light-touch bands -- a real, clean jump.
# False-confirm exposure 14.4s->35.6s, ~0.32%->~0.79% of the same ~75
# pooled no-touch finger-minutes -- still under 1%, and comfortably clear
# of the emerging cliff that starts around 0.30/0.15 (episodes nearly
# double vs the step before it there). Revert to 0.50/0.25 via
# sensing_gui_BACKUP_pre_0.40_0.20_threshold_*.py if needed.
CONTACT_P_CONFIRM = 0.30
CONTACT_P_RELEASE = 0.15
# 2026-07-21, still later same session: bumped once more (0.35/0.18 ->
# 0.30/0.15) at explicit user request for maximum sensitivity ahead of a
# demo where a box touching all 4 fingers still wasn't registering
# reliably at 0.35/0.18. This step has a real, small, already-measured
# cost (~1.07% of no-touch time / 8 episodes across ~10 pooled no-touch
# finger-minutes -- see true_zero_dataset/select_index_only_thresholds_
# report.md's Index-isolated numbers, the closest on-file measurement for
# this exact step). Deliberately NOT pushed further: the next step
# (0.25/0.12) showed ~26% false-confirm time in that same sweep -- the
# system would show false contact roughly a quarter of the time with
# NOTHING touching, which would make a demo look broken in the opposite
# direction. If sensitivity is still insufficient at this level, the
# threshold is very likely not the actual bottleneck anymore (see the
# still-open crosstalk-from-Thumb / genuine-raw-signal questions in
# index-finger-sensitivity-override memory) -- further gains should come
# from investigating why raw signal itself is weak/absent, not from
# continuing to lower these two constants.

# 2026-07-21: raised sensitivity for ALL 4 fingers, not just Index -- a
# live demo with a box touching all 4 fingers found Middle/Ring/Thumb also
# failing to reliably register force, the same underlying symptom Index
# hit earlier the same day (see true-zero-threshold-investigation memory).
# This step (0.40/0.20 -> 0.35/0.18) was ALREADY validated free of cost
# for all 4 fingers pooled together (0 false-accepts across every no-touch
# recording on file, see the CONTACT_P_CONFIRM/RELEASE history further up
# this file and true_zero_dataset/select_thresholds_consolidated_report.md)
# -- previously only applied to Index specifically via a per-finger
# override (see git history / index-finger-sensitivity-override memory)
# after an Index-only re-sweep (true_zero_dataset/select_index_only_
# thresholds.py) confirmed Index had no LESS margin than the rest at this
# same step; now simplified to a single global value since all 4 fingers
# need it and none of them differ from each other at this level. The next
# step down (0.30/0.15) showed a real, small cost even pooled across all
# fingers (~1.07% of no-touch time isolated to Index specifically) -- not
# applied here; revisit the reports above if more sensitivity is wanted
# and that cost is judged acceptable.
CONTACT_P_CONFIRM_PER_FINGER: dict[str, float] = {
    # 2026-09-07: lowered from the 0.30 global default after the retrained
    # (load-cell-augmented) contact_classifier's holdout probabilities
    # showed precision stays a PERFECT 1.000 all the way down to 0.02 for
    # Thumb specifically (783 held-out ticks, 6 reps never used in
    # training) -- the missed touches at 0.30 were real, low-force
    # contacts (0.6-0.8N) the classifier already assigns a real but
    # moderate probability to, not ambiguous/noisy cases. Recall at 0.30
    # was 0.815; at 0.10 it's 0.974, with the SAME zero false-positive
    # rate. Kept the same 2:1 confirm:release ratio as the global default
    # (0.30/0.15) rather than pushing to the more aggressive levels tested
    # (down to 0.02), leaving some margin since this is a single holdout,
    # not exhaustive live testing across every object/condition. See
    # retrain_contact_with_load_cell_data.py's sibling threshold-sweep
    # analysis for the full precision/recall table.
    #
    # CAVEAT (2026-09-08, INVESTIGATED -- see thumb_recall_discrepancy_
    # investigation.md for the full writeup): re-running this exact
    # analysis with a proper, checked-in script (grasping_dashboard/
    # sweep_finger_thresholds.py) hit the IDENTICAL 6 held-out Thumb reps
    # (confirming the seed-42 rep-grouped holdout split reproduces exactly)
    # but could NOT reproduce these recall numbers -- 0.569 at 0.30 and
    # 0.713 at 0.10 measured, not 0.815/0.974. Ruled out as causes: (a) a
    # different model file -- active_model.pkl's contact_classifier/scaler/
    # regressor are bit-identical to the candidate that was actually
    # evaluated; (b) randomness/seed-dependence -- the holdout split and
    # MLPClassifier.predict_proba are both fully deterministic. Tested and
    # NOT sufficient as an explanation: simulating the full live Schmitt-
    # trigger + frame-hysteresis + MIN_CONTACT_FORCE_N state machine
    # (byte-for-byte port of _process_one_finger) instead of a naive
    # per-tick threshold -- moves 0.10's number partway (0.713 -> 0.782,
    # still 19pp short of 0.974) but makes 0.30's number WORSE, not better
    # (0.569 -> 0.488), the opposite of what would explain 0.815. Also
    # found: Thumb's holdout has ZERO true negatives (every held-out tick
    # is a genuine touch), so "perfect precision" here is a trivial
    # property of this holdout, not real evidence against false positives
    # -- live testing is the only way to actually assess Thumb's FP rate.
    # VERDICT: do not cite 0.815/0.974 anywhere -- treat as unverified.
    # Cite 0.569 (@0.30) / 0.713 (@0.10) instead, from sweep_finger_
    # thresholds.py. The 0.10 override is left in place below since it's
    # still a genuine recall improvement either way -- just a smaller one
    # than originally recorded, and with precision untestworthy for Thumb
    # specifically from this data.
    "Thumb": 0.10,
    # 2026-09-08: added after grasping_dashboard/sweep_finger_thresholds.py
    # (per_finger_threshold_sweep_report.md) found Middle's holdout
    # precision PERFECT (1.000) at every threshold tested from 0.30 down to
    # 0.02, with recall climbing steadily as the threshold drops (0.910 @
    # 0.30 -> 0.928 @ 0.10 -> 0.948 @ 0.02) -- a clean, zero-cost win, same
    # shape as Thumb's case. 0.10 chosen (not the more aggressive 0.02) for
    # the same reason as Thumb: keep the 2:1 confirm:release ratio and
    # leave margin on a single holdout. Index and Ring were swept too but
    # NOT given an override -- both already show recall=1.000 at every
    # threshold tested including the current 0.30 default, so lowering
    # their threshold would only cost precision (Index 0.983->0.947, Ring
    # 0.946->0.825 from 0.30->0.02) for zero recall benefit. See the report
    # for the full per-finger tables.
    "Middle": 0.10,
}
CONTACT_P_RELEASE_PER_FINGER: dict[str, float] = {
    "Thumb": 0.05,  # same 2:1 ratio to CONTACT_P_CONFIRM_PER_FINGER["Thumb"] as the global default
    "Middle": 0.05,  # same 2:1 ratio, see CONTACT_P_CONFIRM_PER_FINGER["Middle"]'s comment
}

# --- Transient spike guard (2026-07-17, production review) ------------------
# The only input guard before this was |delta| > SANE_DELTA_MAX (2000):
# single-channel spikes BELOW that pass straight into the model, and
# model_v1/production_review/05_robustness_report.md measured what they do
# -- a 300-raw-unit spike on one channel of a genuinely idle frame drives
# the deployed model to ~22% false confirm-feed frames at ~7.6N phantom
# force. Genuine per-tick channel jumps at the live 20Hz cadence never
# exceeded 19.4 raw units on Units 1+2 held-out data, INCLUDING during
# presses (measured in production_review/00_REVIEW_SUMMARY.md), so a
# 300-unit/tick jump guard has ~15x margin over real dynamics. One-tick-hold
# semantics (see FingerState.spike_hold_active / _process_one_finger): a
# suspicious jump is suppressed ONCE -- the frame is treated exactly like
# the existing SANE_DELTA_MAX glitch path -- and if the same jump persists
# on the next tick it is accepted as a genuine fast change. A real
# ultra-fast press is therefore delayed at most one ~50ms tick, while a
# single-frame electrical/mux spike never reaches the model at all.
SPIKE_JUMP_GUARD_ENABLED = True
SPIKE_JUMP_MAX_PER_TICK = 300.0

# --- Phi display gating (2026-07-17, production review) ---------------------
# Azimuth (Phi) is physically undefined for a near-normal press: at low
# Theta the stimulus is the same whatever Phi the classifier guesses (the
# pre-review model's Phi accuracy at theta=0 was 15% -- BELOW the 3-class
# chance floor, i.e. the degenerate labels actively misled it). With the
# review's retrained Phi head (trained without those rows), gating the
# DISPLAY at |Theta| >= PHI_DISPLAY_MIN_THETA_DEG and max class probability
# >= PHI_DISPLAY_MIN_CONF yields 98.8% displayed-Phi accuracy at ~71%
# coverage on held-out Units-1+2 theta>=30 rows
# (model_v1/production_review/08_phi_head_report.md). Display-only: nothing
# upstream (contact hysteresis, slip detection, recalibration) reads Phi.
PHI_DISPLAY_MIN_THETA_DEG = 25.0
PHI_DISPLAY_MIN_CONF = 0.7

# In addition to the classifier's raw_Contact==1 output, a frame must also
# have a raw Force regression output above this (small) threshold to count
# toward the consecutive_contact_frames streak below -- see
# _process_one_finger. Added because baseline drift (see
# RECALIBRATION_INTERVAL_S) was observed pushing the Contact classifier to a
# sustained false Contact=1 -- and therefore a jittery, nonzero displayed
# Force -- with nothing touching the finger at all, without necessarily
# moving the separately-computed Force regression output by a meaningful
# amount. Gating on both together stops drift ALONE from ever confirming
# contact, since a genuine touch produces a real force magnitude while
# drift alone typically does not. Deliberately small so a genuine light
# touch is not delayed/missed by this gate. Does NOT affect raw_contact
# (the diagnostic toggle-rate reading below still reflects the classifier's
# true, ungated output).
MIN_CONTACT_FORCE_N = 0.1

# --- Contact suppression during a finger's OWN active servo transit --------
# (2026-07-15) model_v1/self_motion/'s investigation (self_motion_report.md)
# found 3 of 4 fingers show elevated noise WHILE moving under their own
# commanded servo motion (moving-up/moving-down phases), separate from and
# unexplained by cross-finger crosstalk (which is structurally ~0 there,
# since no other finger moves in that test). This flag, when True, freezes
# a finger's Contact hysteresis (consecutive_contact_frames/consecutive_
# no_contact_frames/confirmed_contact) for the duration of THAT finger's own
# transit ramp (see HandSensingEngine._finger_position_ramps / the transit-
# timing comment on CROSSTALK_POSITION_RAMP_S_PER_100PCT) -- resuming normal
# evaluation the instant the ramp completes (same duration estimate the
# crosstalk correction's transit ramp already uses, not a new/separate
# timing mechanism). raw_contact/toggle_rate diagnostics are NOT frozen
# (same "gate only affects the confirm streak, diagnostics still show the
# classifier's true output" pattern MIN_CONTACT_FORCE_N's gate above
# already uses) -- only the discrete confirmed/not-confirmed decision is
# held at whatever it was the instant this finger's own transit began.
#
# IMPORTANT SCOPE NOTE (see self_motion_report.md's own "Interpretation"
# section): this only suppresses evaluation WHILE moving. The self-motion
# investigation's dominant finding was a STABLE false positive while
# SETTLED at/near full closure (Middle/Ring/Thumb sitting at 100% confirmed
# contact for a sustained 3s+ window, not a transit-timed transient) -- this
# flag does not and cannot fix that case, since by the time the transit
# ramp completes and suppression lifts, the finger is sitting exactly at
# the position that misfires. This targets the transit-window noise
# specifically (elevated toggle rate/Force std while moving, e.g. Index's
# ~40Hz raw oscillation), not the separate steady-state extrapolation issue
# -- see that report for why a full fix needs either a self-position
# correction term or retrained/expanded training data, neither of which
# this flag attempts.
#
# Single named switch, same "easy to toggle without deleting code" pattern
# as CROSSTALK_CORRECTION_ENABLED -- flip to False to compare false-contact
# behaviour with suppression on vs off.
#
# 2026-07-20 (evening): disabled. Live grip-testing found a genuine
# STRUCTURAL deadlock, not just a duration problem: a closed-loop
# controller that keeps commanding another small step BECAUSE it hasn't
# yet observed target force can never observe that force if literally any
# nonzero suppression window re-arms faster than the loop's own force-check
# cadence -- the finger keeps "trying" to reach a force reading that this
# flag guarantees can never arrive while it's still trying. Progressively
# shrinking CONTACT_TRANSIT_SUPPRESS_S (0.4s -> 0.15s, see that constant)
# reduced but did not eliminate this, since the failure mode depends on
# re-trigger cadence, not window size, on any usage pattern where a new
# move can be commanded faster than the window clears. Disabling
# eliminates the deadlock completely. Trade-off knowingly accepted: this
# reintroduces the FULL self_motion_report.md risk (3 of 4 fingers show
# elevated noise for their entire physical motion, e.g. Index's ~40Hz raw
# oscillation) with NO mitigation at all, not even the reduced protection
# CONTACT_TRANSIT_SUPPRESS_S was providing -- watch for false contact
# specifically WHILE a finger is actively moving (not after it settles;
# AUTO_REBASELINE_ON_RETURN_TO_OPEN_ENABLED, separately, still handles the
# settled-false-positive case this flag was never responsible for anyway).
# Revert by setting this back to True (CONTACT_TRANSIT_SUPPRESS_S's cap
# logic is still in place and would immediately apply again), or restore
# sensing_gui_BACKUP_pre_suppress_disabled_*.py for the exact prior state.
CONTACT_SUPPRESS_DURING_OWN_TRANSIT = False

# --- Cap on the contact-suppression window (2026-07-20) ---------------------
# Live-use finding: CONTACT_SUPPRESS_DURING_OWN_TRANSIT above suppresses for
# the SAME duration the crosstalk-correction position ramp uses
# (CROSSTALK_POSITION_RAMP_S_PER_100PCT, up to 1.0s for a full 0-100% move)
# -- that duration is proportional to how far the finger was COMMANDED to
# travel, not how far it actually got. During a manual grip test (Index/
# Thumb commanded toward/through a target position, but stopped early by an
# actual object), the finger physically stops in a fraction of that time,
# yet Contact stays completely blind for the REST of the original estimate
# -- felt live as "no force reading" for up to ~1s after a large move,
# specifically on the two fingers doing the gripping. This constant caps
# the CONTACT-suppression window specifically (see its use in poll() below)
# at a fixed ceiling, independent of how far the move was commanded --
# SHORT moves (e.g. Grip-to-Target's small per-round steps, whose own ramp
# duration is already well under this cap) are completely unaffected either
# way; only LARGE moves get capped.
#
# Deliberately does NOT touch CROSSTALK_POSITION_RAMP_S_PER_100PCT or the
# crosstalk-correction position interpolation at all -- finger_positions
# (crosstalk correction) and in_own_transit (this suppression) are already
# independent parameters into _process_one_finger (see poll()), so capping
# one has no effect on the other's own, separately-validated duration.
#
# Trade-off, accepted deliberately: self_motion_report.md's finding behind
# CONTACT_SUPPRESS_DURING_OWN_TRANSIT was elevated noise for the finger's
# ENTIRE physical motion (not just an initial backlash blip -- e.g. Index's
# ~40Hz raw oscillation while moving), so this cap reintroduces a modest
# amount of exactly that risk for the TAIL of any large, UNOBSTRUCTED move
# (one that isn't stopped early by an object) -- specifically the window
# between this cap and the move's full original duration, where the finger
# may still genuinely be moving/noisy but Contact evaluation has already
# resumed. Original value: 0.4s, matching hand_control_gui.py's own
# existing SERVO_SETTLE_DELAY_MS=400ms figure for "how long to trust a
# settling servo" (already used/validated elsewhere in this project for
# the same kind of judgement), rather than an arbitrary new number.
#
# 2026-07-20 (same evening): loosened 0.4s -> 0.15s -- live use at 0.4s
# still showed noticeable grip-test interference on Index/Thumb (readings
# still felt blocked for a real, if shorter, window on larger moves), so
# the user asked for a more aggressive cut. 0.15s is close to the fastest
# realistic servo backlash-settling time rather than the more conservative
# SERVO_SETTLE_DELAY_MS figure the original 0.4s borrowed -- accepted
# knowingly trading MORE of the original self-motion-noise protection
# (see the trade-off paragraph above) for faster real force readings
# during a grip. Revert to 0.4s via sensing_gui_BACKUP_pre_0.15_transit_
# cap_*.py, or to fully uncapped behaviour by setting this to float("inf").
CONTACT_TRANSIT_SUPPRESS_S = 0.15

# --- Diagnostics -------------------------------------------------------------
# Rolling window (seconds) over which each finger's raw Contact-classifier
# toggle rate (0<->1 transitions/sec, BEFORE hysteresis smoothing) is
# computed -- see contact_toggle_times/contact_toggle_rate_hz below. Used to
# empirically judge whether CONTACT_CONFIRM_FRAMES/CONTACT_RELEASE_FRAMES are
# tight enough for the model's real observed noise in this GUI, or whether
# the classifier is oscillating right at the confirm/release boundary.
TOGGLE_RATE_WINDOW_S = 1.0

# How often the per-finger diagnostic block (toggle rate, hysteresis streak
# counters, glitch count) is printed to the console -- throttled the same
# way live_predictor_test.py throttles its [DEBUG] block, so this can be run
# for a while and produce a readable log instead of 20 lines/sec.
DIAG_PRINT_INTERVAL_S = 1.0

# When True, prints one [FTRACE] line EVERY frame for EVERY finger -- not
# throttled like the [DIAG] block above -- covering exactly what's needed to
# diagnose the post-release residual-Force question (raw regression output,
# the Force-floor/smoothing/deadband chain that turns it into what's
# displayed, the Contact hysteresis state driving that chain, and
# recalibration timing): raw_Force (result["Force"], unclamped/un-overridden,
# same value the [DIAG] block and EVENT log never show), raw_force_now (the
# Force-floored, contact-gated value BEFORE smoothing), smoothed_force (post
# FORCE_SMOOTHING_WINDOW rolling average), the deadband decision itself
# (whether smoothed_force actually got snapped to 0.0 this tick and why not,
# if not), confirmed_contact plus both hysteresis streak counters, and
# recalibration bookkeeping (no_contact_since age, seconds since
# last_recal_time, whether a recalibration committed this very tick). Meant
# to be turned on only to capture one specific press-and-release cycle for
# offline analysis (same convention as live_predictor_test.py's
# CONTACT_TRACE_ENABLED), then off again -- at ~20Hz x 4 fingers this is far
# noisier than the throttled [DIAG] block above.
# Attempted TEMPORARILY ENABLING this (2026-07-17) to diagnose Index's
# reproducible stuck-confirmed issue (see candidate_v2/README.md's "Live
# test findings") -- reverted back to False after 3 consecutive identical
# crashes (exit 127, right after "Servo control link established", before
# calibration even started) with it set to True, vs. zero crashes across
# every earlier launch with it False. Testing whether this flag is actually
# implicated or whether the crashes are an unrelated rapid-relaunch/serial-
# port-recovery issue.
FORCE_TRACE_ENABLED = False

# --- Periodic baseline recalibration ----------------------------------------
# The [DIAG] session that motivated the CONTACT_CONFIRM_FRAMES/
# CONTACT_RELEASE_FRAMES bump above also showed the actual root cause: Finger
# 1/2 were perfectly stable (toggle rate 0.00/s) for roughly two-thirds of
# the session and only started destabilizing near the end, while Finger 4
# was noisiest mid-session -- a pattern that drifts over time rather than a
# fixed per-channel fault. That's consistent with the baseline captured once
# at connect time slowly going stale (thermal drift, etc.), which makes
# every later delta feature (and therefore the Contact classifier's input)
# progressively noisier the longer a session runs. Periodically re-anchoring
# each finger's baseline against its own CURRENT resting readings is the
# targeted fix for that specific mechanism, independently of any GUI-display
# smoothing/hysteresis tuning above.
#
# How often (wall-clock seconds) each finger is considered for a background
# baseline refresh. Per-finger, not global -- see HandSensingEngine's
# last_recal_time bookkeeping -- so a finger currently in contact never
# blocks another finger's recalibration and never blocks its own once it
# does settle. Lowered from the original 120s: at-rest Force jitter was
# still showing up well within a 120s window, i.e. drift was accumulating
# faster than that interval corrected for it. 45s still keeps recalibration
# an infrequent background event (and MIN_CONTACT_FORCE_N above now stops
# the drift itself from blocking its own correction by falsely confirming
# contact), while tracking drift meaningfully faster than before.
RECALIBRATION_INTERVAL_S = 45.0

# A finger must have been CONTINUOUSLY confirmed "no contact" for at least
# this long before a recalibration attempt is allowed to commit, so a
# refresh can never fire mid-press or fire against residual settling right
# after a release (which would bake a not-quite-resting reading into the new
# baseline). Reset the instant contact is confirmed again -- see
# no_contact_since in FingerState.
RECALIBRATION_MIN_STABLE_NO_CONTACT_S = 3.0

# --- Stale-baseline cascade mitigation --------------------------------------
# Diagnostic finding (2026-07): a finger whose Contact classifier keeps
# false-positiving (e.g. residual crosstalk noise, see
# CROSSTALK_CORRECTION_ENABLED below) can get its recalibration blocked
# indefinitely -- not because the code is broken, but because it's working
# exactly as designed: every time confirmed_contact flips True,
# no_contact_since resets to None (see _process_one_finger), so a finger
# whose false-positives recur more often than every
# RECALIBRATION_MIN_STABLE_NO_CONTACT_S (3.0s) can never accumulate a full
# clean window to recommit, leaving it stuck on an increasingly stale
# baseline. Observed live: one finger stayed stuck for 13.5+ seconds during
# a busy multi-finger test session.
#
# Mitigation (deliberately NOT a hard override -- see RECALIBRATION_STUCK_
# MIN_STABLE_NO_CONTACT_S below): once a finger has gone this long without a
# SUCCESSFUL recalibration commit, _process_one_finger temporarily accepts a
# much shorter stable-no-contact window instead of the normal 3.0s -- still
# requires a genuine gap where confirmed_contact is False, just a shorter
# one, so recovery is faster the next time a false-positive burst clears
# even briefly. This can NEVER commit while confirmed_contact is True,
# regardless of how long the finger has been stuck -- that gate is
# unconditional in _process_one_finger and this mitigation does not touch
# it. A hard time-based override that ignores confirmed_contact entirely
# was considered and rejected: if it ever fired during a genuine sustained
# press, it would bake the press into the new "resting" baseline and the
# system would then silently read zero force under real, continued load --
# a worse failure than a temporarily stale baseline.
RECALIBRATION_STUCK_THRESHOLD_S = 15.0

# The relaxed stable-no-contact requirement used only once
# RECALIBRATION_STUCK_THRESHOLD_S has been exceeded (see above) -- still a
# real requirement, not zero/instant, so a momentary clean blip still can't
# recalibrate against not-yet-settled data; just short enough that the next
# genuine gap between false-positive bursts is enough to recover, rather
# than needing a 3-second window no burst pattern has been leaving room for.
RECALIBRATION_STUCK_MIN_STABLE_NO_CONTACT_S = 1.0

# Minimum number of raw samples required before a recalibration attempt
# commits (mirrors MIN_BASELINE_SAMPLES's role for the one-time startup
# calibration) -- below this, the attempt is skipped and retried on the next
# RECALIBRATION_INTERVAL_S tick rather than committing a baseline averaged
# over too few points.
RECALIBRATION_MIN_SAMPLES = 10

# How many of the most recent no-contact raw samples are retained per finger
# while waiting to accumulate enough for a recalibration -- just needs to
# comfortably exceed RECALIBRATION_MIN_SAMPLES; bounds memory in case a
# finger sits untouched (and therefore never triggers the
# RECALIBRATION_MIN_STABLE_NO_CONTACT_S->commit path just yet) for a very
# long stretch of a long session.
RECAL_SAMPLE_BUFFER_MAXLEN = 200

# --- Baseline tracking: mode + contamination gate (2026-07-15) --------------
# Two problems, confirmed (not just theorized) by model_v1/crosstalk_lookup/
# diagnose_recalibration_cadence.py's simulation, which drives this file's
# own real recalibration logic against synthetic touch/release timelines
# over a virtual clock:
#
#   1. STARVATION -- everything above this comment (RECALIBRATION_MIN_
#      STABLE_NO_CONTACT_S=3.0s, and even the STUCK-recovery relaxed 1.0s)
#      requires a CONTINUOUS no-contact window before a commit can happen at
#      all. Under any repeated touch/release cadence with gaps under ~2s
#      (an entirely ordinary interaction pattern -- press for a second,
#      let go for a second, repeat), the simulation showed literally ZERO
#      commits across a full 5-minute session -- not delayed, completely
#      starved the whole time.
#   2. CONTAMINATION -- _process_one_finger's own long-standing comment
#      already flagged that a commit landing while a NEIGHBOUR is mid-
#      motion bakes that neighbour's then-current crosstalk into the new
#      baseline, but this was never actually prevented in code. The same
#      simulation confirmed it's not just a theoretical risk: in the
#      cadences where a commit DID eventually squeeze through (via the
#      STUCK-recovery path's narrower 1.0s window), most of those commits
#      landed exactly during simulated neighbour motion -- the escape valve
#      recalibration gets is disproportionately likely to bake in bad data
#      right when it fires.
#
# BASELINE_TRACKING_MODE picks which mechanism is active:
#   "low_pass" (default) -- see _update_baseline_low_pass(). Fixes #1: a
#       continuously-updating exponential low-pass average, applied on
#       EVERY eligible tick (no contact, no transit anywhere) instead of
#       requiring one long uninterrupted window -- there is no "stable
#       window" left to starve.
#   "stable_window" -- the ORIGINAL mechanism (RECALIBRATION_MIN_STABLE_
#       NO_CONTACT_S / RECALIBRATION_STUCK_* / RECALIBRATION_MIN_SAMPLES /
#       RECAL_SAMPLE_BUFFER_MAXLEN above), kept fully intact and selectable
#       as an explicit fallback -- byte-identical to its pre-2026-07-15
#       behaviour except for the contamination gate below, which applies to
#       BOTH modes (see _update_baseline_stable_window()). Single named
#       switch, same "revert without deleting code" pattern as
#       CROSSTALK_CORRECTION_ENABLED/CONTACT_SUPPRESS_DURING_OWN_TRANSIT.
BASELINE_TRACKING_MODE = "low_pass"

# Per-eligible-sample EWMA smoothing factor for "low_pass" mode:
# baseline[col] += BASELINE_LOW_PASS_ALPHA * (raw_absolute[col] - baseline[col]).
# Chosen via alpha = 2/(N+1) (the standard EMA "span" formula) with N=200,
# matching RECAL_SAMPLE_BUFFER_MAXLEN's own choice of window size for the
# OLD mechanism's averaging batch -- deliberately the same rough "how many
# samples of memory does this filter have" scale as before, just spread
# continuously across eligible ticks instead of one discrete batch. Applied
# per ELIGIBLE sample, not per wall-clock tick or wall-clock second -- a
# busier session (fewer eligible ticks) naturally adapts more slowly in
# real time, which is the correct behaviour (less clean data to learn from),
# but it NEVER fully stops the way the stable-window mechanism's starvation
# does, since even a single eligible tick between touches contributes a
# (small) update.
BASELINE_LOW_PASS_ALPHA = 0.01

# How often (wall-clock seconds) "low_pass" mode emits a status message
# (the low-pass equivalent of stable_window's "baseline recalibrated"
# message) summarising drift accumulated since the last one -- purely a
# GUI-log throttle, does not gate or affect the underlying continuous
# update in any way. Reuses RECALIBRATION_INTERVAL_S's own value/spirit
# (the original "how often should the user hear about this" cadence) for
# continuity, not because low_pass mode's actual update rate is tied to it.
BASELINE_LOW_PASS_REPORT_INTERVAL_S = RECALIBRATION_INTERVAL_S

# 2026-07-20: minimum wall-clock time after a confirmed_contact release
# (True->False transition, tracked in FingerState.release_time) before
# low_pass eligibility is allowed to resume -- see _update_baseline_low_pass.
# Fixes a self-reinforcing baseline-capture loop found via
# record_press_while_traveling.py: a single spurious release (e.g. one brief
# signal dip mid-transit, even while the finger is genuinely still being
# pressed) opened eligibility with zero delay; the very next eligible tick
# nudged baseline toward the still-elevated raw signal, shrinking raw_delta,
# which dropped p_contact, which kept eligibility open -- baseline then
# chased the touch to convergence within ~1s (baseline_l1_drift climbed
# unbounded, observed reaching 1000+ over ten seconds) and the touch stayed
# permanently masked (p_new pinned at 0.000) until raw force rose enough to
# outrun the ~1%/tick drift. A short cooldown gives the classifier a clean,
# unmoved baseline for one full CONTACT_CONFIRM_FRAMES-worth of ticks right
# after any release, so a genuinely still-present touch reconfirms (ccf ramps
# back up, which independently keeps eligibility closed via the
# consecutive_contact_frames==0 check already in place) before baseline ever
# gets a chance to move. 1.0s is several multiples of CONTACT_CONFIRM_FRAMES
# (4 ticks, ~0.2s) at the ~20Hz poll rate, so a real re-touch reconfirms well
# inside the window; a GENUINE release (finger actually lifted) simply waits
# an extra second before low_pass resumes tracking it, which is a one-time
# cost per release, not a recurring one.
BASELINE_LOW_PASS_POST_RELEASE_COOLDOWN_S = 1.0

# --- Auto-rebaseline on return to the open reference position (2026-07-17) -
# Live-testing finding (see candidate_v2/README.md's "Live test findings"):
# joint backlash/play means a commanded "0% open" is not a perfectly
# reproducible TRUE physical rest position -- a finger can settle a small
# but real amount away from wherever it was when its baseline was last
# captured, purely from mechanical slop, with nothing touching it. If that
# small settling difference is enough to push raw_delta over the Contact
# classifier's decision boundary, it gets falsely confirmed -- and because
# EVERY existing recalibration path above (both BASELINE_TRACKING_MODEs)
# refuses to update a baseline while confirmed_contact is True (see
# RECALIBRATION_STUCK_THRESHOLD_S's comment for why that rule normally
# holds -- baking a genuine ongoing press into a "resting" baseline would be
# worse), that false confirm was PERMANENTLY unrecoverable without
# restarting the app. Reproduced live 4/4 times, independent of crosstalk
# backend and independent of CONTACT_PROB_SCHMITT_ENABLED, so it is neither
# of those -- it is specifically the open-position backlash + baseline-
# freeze-while-confirmed combination.
#
# This flag adds ONE deliberate, narrow exception to that "never recalibrate
# while confirmed" rule: the instant a finger's own commanded-position
# transit completes AND it has settled at the fully-open reference position
# (0%, the SAME position the one-time startup calibration itself used), a
# short burst of fresh raw samples is collected and forcibly becomes that
# finger's new baseline -- regardless of confirmed_contact. This does not
# force confirmed_contact itself back to False; it only replaces the stale
# baseline, so the classifier naturally sees a near-zero delta again on the
# next tick and the EXISTING release-streak/Schmitt-release-probability
# logic un-confirms it through its normal path (typically within a quarter
# second), rather than this flag bypassing that safety net directly.
#
# Trade-off, accepted deliberately (same one RECALIBRATION_STUCK_THRESHOLD_S's
# own comment already flagged and rejected for the GENERAL any-position
# case): if a finger is genuinely being touched at the exact instant its
# transit settles at the open reference, that touch gets baked into the new
# baseline. Re-accepted here only because this trigger is much narrower --
# the fully-open position is this hand's normal AT-REST position, not an
# active-grip position, so a genuine touch landing in this specific
# ~REBASELINE_OPEN_SETTLE_S window is far less likely than at an arbitrary
# already-confirmed position. Flip to False to remove this exception
# entirely (same "revert without deleting code" pattern used throughout this
# file) if that trade-off ever proves wrong in practice.
AUTO_REBASELINE_ON_RETURN_TO_OPEN_ENABLED = True

# How close to the fully-open reference (0%) a finger's settled position
# must be to count as "returned to open" -- a small tolerance for float
# accumulation in the position-ramp interpolation, not a meaningfully wide
# window (Close/Open All commands settle exactly at 0/100, so this rarely
# needs to matter in practice).
REBASELINE_OPEN_POSITION_TOLERANCE_PCT = 1.0

# How long (seconds) to collect fresh raw samples after arriving at the
# open reference before committing them as the new baseline -- gives the
# mechanical settling itself (the backlash effect this whole feature exists
# to correct for) time to actually finish, same spirit as SERVO_SETTLE_
# DELAY_MS in hand_control_gui.py but independently tunable here. Shorter
# than the one-time startup calibration's BASELINE_DURATION_S (2.0s) since
# this is a lightweight refresh of an already-close baseline, not a
# from-scratch one.
REBASELINE_OPEN_SETTLE_S = 0.5

# Minimum samples required before a rebaseline commits (mirrors
# RECALIBRATION_MIN_SAMPLES's role for the periodic mechanisms above) --
# below this, more ticks are collected rather than committing an average
# over too few points.
REBASELINE_OPEN_MIN_SAMPLES = 8

# Bounds the sample buffer -- comfortably above REBASELINE_OPEN_MIN_SAMPLES
# at the live ~20Hz poll rate over REBASELINE_OPEN_SETTLE_S, with headroom
# in case commit is briefly delayed.
REBASELINE_OPEN_MAX_SAMPLES = 40

# --- Crosstalk correction (model_v1/crosstalk/, see its own README.md) -----
# Additive, on top of everything above: subtracts each finger's PREDICTED
# magnetic crosstalk shift (from the other 3 fingers' CURRENT commanded
# servo positions, via crosstalk_model.predict_shift()) from raw_delta,
# right after the static/periodic baseline subtraction above and before the
# glitch guard / model prediction -- see _process_one_finger. Does not
# replace, gate, or alter the static startup baseline, periodic
# recalibration, deadband, smoothing, glitch guard, or Contact classifier
# in any way; if this flag is False (or the model fails to load), every
# finger's crosstalk correction is simply 0, i.e. behaviour is byte-for-byte
# identical to before this feature existed.
#
# Single named switch, per the integration request -- flip to False to
# compare noise/false-contact behaviour with the correction on vs. off
# without deleting any code.
CROSSTALK_CORRECTION_ENABLED = False  # DISABLED AGAIN 2026-09-09, explicit user request, same demo session -- was briefly re-enabled earlier the same day (see git history for that comment) then turned back off. Originally set False 2026-09-07 as a TEMPORARY diagnostic testing whether the retrained (load-cell) contact_classifier -- trained on UNCORRECTED deltas -- performs worse when fed corrected deltas it never saw in training during real multi-finger grasps; that diagnostic's own result/conclusion was never recorded anywhere. Net effect of today's back-and-forth: back to the same False state this file already had before tonight, no different than if it had never been touched.

# model_v1/crosstalk/train_crosstalk_model.py's fitted output -- loaded once
# at HandSensingEngine construction (see __init__), same "load once at
# startup" pattern as MODEL_PATH/LivePhiClassPredictor above.
CROSSTALK_MODEL_PATH = SCRIPT_DIR / "crosstalk" / "crosstalk_model_v1.json"

# model_v1/crosstalk_lookup/nn_lookup_model.py's directly-measured nearest-
# neighbour alternative (see that folder's README.md) -- loaded alongside
# CROSSTALK_MODEL_PATH above, same "load once at startup" pattern.
NN_LOOKUP_MODEL_PATH = SCRIPT_DIR / "crosstalk_lookup" / "nn_lookup_model_v1.json"

# model_v1/crosstalk_lookup/ridge_crosstalk_model.py's origin-anchored
# quadratic-ridge backend (2026-07-17, production review -- see
# model_v1/production_review/07_crosstalk_ridge_report.md), same "load once
# at startup" pattern as the two paths above.
RIDGE_CROSSTALK_MODEL_PATH = SCRIPT_DIR / "crosstalk_lookup" / "crosstalk_ridge_v1.json"

# (2026-08-30) fifth backend, Index-only -- see magnetfree_index_predict_shift's
# import comment above.
MAGNETFREE_INDEX_MODEL_PATH = SCRIPT_DIR / "crosstalk_lookup" / "magnetfree_index_model_v1.json"

# Engine finger index (0..3, i.e. mux Group 1..4) -> crosstalk-model finger
# label ("Index"/"Middle"/"Ring"/"Thumb"). Confirmed mapping (same one
# hand_control_gui.py's FINGER_CONTROLS and record_crosstalk.py's
# FINGER_DEFS already use): Index=Finger 1 (index 0), Middle=Finger 2
# (index 1), Ring=Finger 3 (index 2), Thumb=Finger 4 (index 3).
CROSSTALK_FINGER_LABELS = ["Index", "Middle", "Ring", "Thumb"]

# --- Crosstalk correction SOURCE, per finger (2026-07-15, revised same day) -
# Three interchangeable correction backends now exist -- crosstalk_model.py's
# original fitted pairwise-additive polynomials ("pairwise_additive"),
# model_v1/crosstalk_lookup/nn_lookup_model.py's directly-measured nearest-
# neighbour lookup ("nearest_neighbor", built from 100 real simultaneous-
# 4-finger recordings, self-effect-corrected -- see that folder's README.md
# and self_effect.py's module docstring), and that same data with an origin
# anchor plus distance-weighted blending ("nearest_neighbor_smoothed" -- see
# below for why). All three implement the exact same predict_shift(model,
# affected_label, source_positions) -> {channel: shift} contract (see
# CROSSTALK_CORRECTION_BACKENDS below), so swapping which one a given finger
# uses is just changing its entry here -- no other code path changes.
#
# Per-finger, not a single global switch, because model_v1/crosstalk_lookup/
# validate_corrected_nn.py's held-out evaluation (scored against the actual
# deployment target, crosstalk_from_others -- NOT the raw delta_from_
# reference model_v1/crosstalk_lookup/comparison_report.md scored, which
# conflated each finger's own self-effect into the target and inflated
# every approach's error roughly alike) found the backends do NOT agree on
# which one is better for every finger:
#
#   Finger   pairwise_additive MAE   nearest_neighbor MAE   nn_smoothed MAE
#   Index         35.8                    19.8                   19.4  <- best
#   Middle        43.4                    50.1                   47.6
#   Ring          35.0                    47.2                   38.8
#   Thumb          3.2                     4.3                    3.3
#
# (see model_v1/crosstalk_lookup/comparison_report_corrected.md for the full
# per-channel table). A nearest-neighbour variant only wins for Index --
# which is also the finger the original false-contact/noise investigation
# this whole pipeline grew out of was about -- while pairwise_additive
# remains the better OFFLINE choice for Middle/Ring/Thumb. A blanket global
# swap would have fixed Index at the cost of regressing the other three, so
# this stays a per-finger decision rather than one flag, confirmed with the
# user rather than assumed.
#
# Index: nearest_neighbor_smoothed, not plain nearest_neighbor -- live
# regression found and fixed (2026-07-15, same day). realistic_session_
# diagnostic.py's COUNTERBALANCED A/B (order alternated across repetitions
# specifically to rule out session-drift artifacts) found plain
# nearest_neighbor produced MORE Index false-contact episodes than
# pairwise_additive, not fewer: pooled across 10 reps, 5.00+/-0.00
# episodes/session vs pairwise_additive's 3.60+/-1.45 (15 reps pooled) --
# despite nearest_neighbor's better MAE above. Root cause (confirmed
# directly, see nn_lookup_model.py's build_model() docstring):
# pairwise_additive's polynomials are fitted with no intercept, forced
# through the origin BY CONSTRUCTION, so at the reference position (all
# other fingers at 0%) it always predicts EXACTLY 0.0 -- plain
# nearest_neighbor has no such guarantee (its nearest stored point can sit a
# genuine distance away with a real nonzero measured shift; one measured
# case: -37 raw units on S3Z where the correct answer is exactly 0), and 4
# of the 5 flagged false-contact sequence steps queried exactly that
# reference position. nearest_neighbor_smoothed fixes this (explicit
# zero-shift anchor at the reference position, plus distance-weighted
# blending of the nearest few points instead of a hard single-nearest jump
# at Voronoi cell boundaries) without giving up the accuracy win (MAE 19.4,
# even slightly better than plain nearest_neighbor's 19.8). Re-validated
# live with the SAME counterbalanced methodology: pooled Index gap dropped
# from 1.40 episodes/session (5.00 vs 3.60) to 0.20 (3.80+/-0.40 over 5 reps
# vs the same 3.60+/-1.45 pairwise_additive baseline) -- effectively closed.
# Plain "nearest_neighbor" (unsmoothed) is kept available as a backend for
# comparison/regression-testing, just no longer any finger's default.
#
# Toggle any finger (or try a new backend) by editing its entry here; if
# nn_lookup_model_v1.json fails to load, every finger configured for
# "nearest_neighbor"/"nearest_neighbor_smoothed" transparently gets no
# correction rather than falling back to pairwise_additive (see
# HandSensingEngine.__init__) -- silently swapping which physical
# correction a finger receives on a load failure would be more surprising
# than simply disabling it, and __init__'s print makes the degraded state
# visible either way.
# (2026-07-17, production review) Index and Middle were TRIED on
# "ridge_quadratic" -- model_v1/production_review/07_crosstalk_ridge_report.md's
# 5x repeated 5-fold CV over all 100 directly-measured 4-finger combinations
# (self-effect-corrected target) had given Middle 46.8 -> 41.7 MAE vs
# pairwise_additive (-11%) and Index 18.2 -> 17.7 vs nearest_neighbor_smoothed,
# with the same structural exact-zero-at-the-origin guarantee that made
# nearest_neighbor_smoothed safe to deploy, plus continuity everywhere and no
# additive-combination assumption. REVERTED THE SAME DAY after a live
# candidate_v2 test: closing all fingers then opening through ~70% back to
# 100% (fully open) left Index and Middle STUCK at confirmed_contact=True
# indefinitely (170+ and 172+ consecutive confirmed=True DIAG ticks with no
# release, vs Ring/Thumb -- both still on pairwise_additive -- cleanly
# returning to confirmed=False in the same window). This is exactly the
# offline-MAE-mispredicts-live-behaviour failure mode plain nearest_neighbor
# already produced once before in this exact spot (see the 2026-07-15 note
# above) -- the ridge backend likely has residual correction error near the
# fully-open end of its fitted range for these two fingers, keeping the
# Contact classifier's probability parked in the Schmitt trigger's dead zone
# (see CONTACT_P_CONFIRM/CONTACT_P_RELEASE) rather than dropping low enough
# to release. "ridge_quadratic" is kept registered as an available backend
# (see CROSSTALK_CORRECTION_BACKENDS) for future comparison/regression-
# testing, same as plain "nearest_neighbor" above, just not deployed to any
# finger pending a fix (repeat calibration data nearer 100% open, most
# likely) and re-validation with this same live test.
CROSSTALK_CORRECTION_SOURCE: dict[str, str] = {
    "Index": "nearest_neighbor_smoothed",
    "Middle": "pairwise_additive",
    "Ring": "pairwise_additive",
    "Thumb": "pairwise_additive",
}

# Third backend (2026-07-15): model_v1/crosstalk_lookup/nn_lookup_model.py's
# predict_shift_smoothed() -- same directly-measured data as "nearest_
# neighbor" above, but (a) with an explicit zero-shift anchor point added at
# the reference position (0%,0%,0%) and (b) a distance-weighted blend of the
# nearest few points instead of a hard single-nearest jump. Added after
# realistic_session_diagnostic.py's counterbalanced live A/B found "nearest_
# neighbor" (Index's deployed default at the time) produced MORE false-
# contact episodes than pairwise_additive, not fewer -- 5.00+/-0.00 vs
# 3.80+/-1.33 episodes/session pooled across 10 reps each -- despite scoring
# BETTER on offline MAE. Root cause, confirmed directly (see nn_lookup_
# model.py's build_model() docstring): "nearest_neighbor" has no equivalent
# of pairwise_additive's structural "forced through the origin" guarantee,
# so at/near the reference position its nearest stored point could be a
# genuine distance away with a non-trivial measured shift of its own (one
# measured case: -37 raw units on S3Z where the correct answer is exactly
# 0) -- and 4 of the 5 flagged false-contact sequence steps queried exactly
# that reference position. "nearest_neighbor_smoothed" fixes both the
# origin gap and the general hard-jump discontinuity at Voronoi cell
# boundaries; model_v1/crosstalk_lookup/validate_corrected_nn.py's offline
# re-check confirmed it does not cost the steady-state accuracy advantage
# (Index MAE 19.4 vs plain nearest_neighbor's 19.8, both far ahead of
# pairwise_additive's 35.8) -- see CROSSTALK_CORRECTION_SOURCE's own
# comment for the live counterbalanced re-validation result that decided
# whether this became Index's actual deployed default.
#
# All three backends share the exact same call signature (verified:
# `(model, affected_label, source_positions) -> {channel: shift}`), so
# dispatch is a plain lookup, not a branch -- a fourth backend later only
# means one more entry here plus a matching CROSSTALK_CORRECTION_SOURCE
# value, no call-site changes.
CROSSTALK_CORRECTION_BACKENDS = {
    "pairwise_additive": crosstalk_predict_shift,
    "nearest_neighbor": nn_lookup_predict_shift,
    "nearest_neighbor_smoothed": nn_lookup_predict_shift_smoothed,
    # (2026-07-17, production review) fourth backend -- same call signature
    # as the three above; see RIDGE_CROSSTALK_MODEL_PATH and the switch
    # rationale on CROSSTALK_CORRECTION_SOURCE.
    "ridge_quadratic": ridge_crosstalk_predict_shift,
    # (2026-08-30) fifth backend, Index-only -- see magnetfree_index_predict_shift's
    # own import comment above for what this is and why it's not deployed yet.
    "magnetfree_index": magnetfree_index_predict_shift,
}

# Diagnostic finding (2026-07): hand_control_gui.py's _update_position_label
# calls set_finger_position() SYNCHRONOUSLY the instant a servo command is
# sent -- before the physical servo has actually finished travelling there.
# Treating the position as an instantaneous step function meant the
# crosstalk correction assumed a neighbour's FINAL position immediately,
# while the real magnetic field was still ramping as that finger physically
# moved -- guaranteed-wrong during the transit window regardless of how
# well-fit the underlying crosstalk model is. This showed up as transient
# (self-resolving within ~2s), timing-correlated false "Contact" episodes
# on Index specifically (onset 0.2-1.7s after a neighbour's move, i.e.
# overlapping the transit window), compounding Index<-Middle's own already-
# elevated crosstalk-fit residual error (see model_v1/crosstalk/
# crosstalk_model_report.md).
#
# Fix: set_finger_position() now records a RAMP (see HandSensingEngine.
# _current_finger_position) from the finger's previous estimated position to
# the new commanded target, linear over an estimated transit duration
# proportional to how far it's travelling, instead of jumping instantly.
# poll() re-samples this ramp fresh every tick, so the crosstalk correction
# tracks an estimate of where the finger actually is during the move, not
# just its eventual destination.
#
# CROSSTALK_POSITION_RAMP_S_PER_100PCT is the estimated time (seconds) for a
# FULL 0-100% swing at the default servo speed (hand_controller.DEFAULT_SPEED
# == 3) -- scaled linearly for shorter moves (see set_finger_position). Not
# a per-move measurement: derived from (a) hand_control_gui.py's own
# SERVO_SETTLE_DELAY_MS=400ms figure for "the servo's own motion" on smaller,
# more typical moves, (b) the observed 0.2-2.0s span of false-contact episode
# onset+duration for full-range moves in the live diagnostic session, and
# (c) a back-of-envelope angular-speed estimate (~120 degrees total 0-100%
# travel range at DEFAULT_SPEED's ~170 deg/s -> ~0.7s). A single fixed
# estimate, not scaled by whatever Speed dropdown value a given move
# actually used -- see hand_control_gui.py's Speed dropdown -- so this ramp
# duration is most accurate at the default speed and becomes a rougher
# approximation if a different speed is selected.
CROSSTALK_POSITION_RAMP_S_PER_100PCT = 1.0

# REVERTED (2026-07-15): a position deadband (CROSSTALK_POSITION_DEADBAND_PCT,
# quantizing each finger's position estimate to the nearest 5% before it's
# used by the crosstalk correction) briefly lived here. It was added to
# fight tick-to-tick correction JITTER from mechanical backlash (see
# model_v1/crosstalk/position_sensitivity_report.md's original analysis),
# and did measurably reduce Index's false-contact episode count. But that
# same report's own "Deadband widening effectiveness" section quantifies
# the cost: adding the 5% deadband roughly DOUBLED Index<-Middle's
# worst-case crosstalk-correction error at every backlash magnitude tested
# (e.g. +/-2% backlash: 6.10 raw units with no deadband vs. 12.00 with the
# 5% deadband; +/-5% backlash: 14.88 vs. 20.50) and raised its MEAN error
# across the full position range too (e.g. 0.82 -> 1.28 at +/-2%). Index<-
# Middle is Index's dominant crosstalk source, so that extra, systematic
# over/under-correction on Index's own raw_delta was large enough to bias
# its predicted Force low enough to be felt as "weak"/underestimated
# readings during real presses -- a live-feel regression that matters more
# than the false-contact noise the deadband was suppressing. Removed the
# quantization step entirely (poll() now feeds _current_finger_position()'s
# raw, un-quantized estimate straight to the crosstalk correction, same as
# before this constant existed) -- Index's false-contact noise is expected
# to return to its pre-deadband level as a result; see the transit-timing
# ramp above, which is NOT part of this revert and stays in place.

WINDOW_TITLE = "eFlesh Hand Sensing Monitor"
MIN_WINDOW_W = 1180
MIN_WINDOW_H = 820

# Vertical force gauge full-scale. 15N comfortably covers the force range
# seen in prior characterisation recordings (unified_gui.py's own sane-data
# clip, MAX_VALID_FORCE_N, is 20N) while still giving useful resolution for
# the light fingertip-contact forces this rig actually produces.
FORCE_GAUGE_MAX_N = 15.0

# Rolling force-history chart window, in seconds. 5-10s is enough to see a
# slip (a sudden dip) in context without the chart scrolling by too fast to
# read.
HISTORY_WINDOW_S = 8.0

# --- Slip detection -- a deliberately simple, easily-retuned rule ---------
# A finger is flagged as "slipping" when, while contact was already
# confirmed, its displayed Force drops by at least SLIP_DROP_FRACTION from
# its own peak within the trailing SLIP_WINDOW_S seconds. This is a coarse
# heuristic for a demo/monitoring tool, not a trained slip classifier --
# raise SLIP_DROP_FRACTION or shrink SLIP_WINDOW_S if ordinary noise/soft
# releases are triggering false positives; lower/widen them if genuine
# slips are being missed.
SLIP_DROP_FRACTION = 0.30
SLIP_WINDOW_S = 0.5

# How long a finger's panel stays flagged red after a slip is detected,
# independent of whether the force has since recovered -- purely so the
# flag is visible to a human glancing at the screen instead of flashing for
# a single ~50ms tick.
SLIP_DISPLAY_HOLD_S = 1.0

# --- Post-release display stability ----------------------------------------
# Both constants below only shape what's shown to the user (the numeric
# Force readout + gauge bar via FingerResult.force) -- neither ever touches
# the raw per-frame Force used for the Contact classifier's hysteresis state
# or for slip detection (both keep reacting to the unsmoothed, un-deadbanded
# value -- see raw_force_now in _process_one_finger), so slip detection stays
# exactly as fast/responsive as before.

# Once contact has ended (confirmed_contact is False), a DISPLAYED Force
# below this is snapped to exactly 0.0 instead of showing small residual
# jitter left over while FORCE_SMOOTHING_WINDOW's rolling average is still
# draining out the last real (nonzero) readings from before release.
#
# Widened from 0.2 -- offline analysis of idle-Force noise (no genuine
# touch, true Force<0.05N) found this floor was already letting roughly
# half of resting readings through even before accounting for further
# session-noise accumulation (idle predicted-Force p50~0.17N, p90~0.57N).
# Safe to raise further than that might sound: this deadband ONLY ever
# applies while confirmed_contact is False, so it can never suppress or
# delay a genuine touch -- that's gated entirely by the separate Contact
# classifier + MIN_CONTACT_FORCE_N + CONTACT_CONFIRM_FRAMES hysteresis
# above, none of which this constant touches. 0.4 is a deliberately
# conservative first step (comfortably below the ~0.6-1N a genuine light
# touch typically reads once confirmed) meant to visibly reduce resting-
# state jitter without masking anything real -- raise further if
# accumulated session noise still shows through it.
FORCE_DEADBAND_N = 0.4

# Rolling-average window (in frames) applied ONLY to the displayed Force
# number/gauge, to smooth out per-frame regression noise. Kept small so it
# doesn't introduce noticeable display lag -- deliberately NOT widened
# alongside FORCE_DEADBAND_N above, since a longer window directly adds
# display lag (each extra frame is another ~50ms at the ~20Hz poll rate),
# and lag was a previously-investigated, separately-tracked concern in its
# own right. FORCE_DEADBAND_N alone -- a threshold, not a window -- gets
# the requested noise reduction without that cost; only widen this too if
# the deadband increase alone turns out not to be enough.
#
# Re-tuning attempted and REVERTED (2026-07-15, model_v1/self_motion/
# tune_force_smoothing_window.py / force_smoothing_window_report.md): first
# pass measured noise reduction against self_motion_dataset.csv's
# arrived-100 phase, which was a STABLE FALSE-POSITIVE PLATEAU at the time
# (raw_force_now nonzero throughout, since confirmed_contact was wrongly
# True) -- widening to 6 looked like a real, if modest, win on that data.
# That dataset turned out to be STALE: it predated the Contact classifier
# self-closure fix's deployment (contact_classifier_fix_deployed_at_iso =
# 2026-07-13T23:02:43Z; the dataset was built at 22:10, before it). Re-run
# against the CURRENTLY deployed (already-fixed) model, arrived-100 no
# longer confirms contact at all (0.0% across all 4 fingers -- see
# self_motion_report.md) -- raw_force_now is now constantly 0.0 throughout
# that phase for every finger, so there is literally nothing left to
# measure a noise REDUCTION against with the data currently available (any
# window size trivially smooths a constant zero to the same constant
# zero). Since raw_force_now is 0.0 whenever confirmed_contact is False by
# construction (see _process_one_finger), this constant's noise-reduction
# benefit can only ever be tested against genuine or false CONFIRMED
# contact data -- with the false-positive case now fixed and no real-touch
# capture available (would need a human physically pressing the sensor),
# there is currently no valid dataset to re-derive a window size from.
# Reverted to 4 rather than keep an unsupported change on record -- the
# analytical lag numbers in force_smoothing_window_report.md remain valid
# (pure arithmetic, independent of any dataset) if this is revisited once
# real touch data becomes available.
FORCE_SMOOTHING_WINDOW = 4

# ============================================================================
# Backend: per-finger state + processing (Qt-independent, reuses the
# console tool's constants/frame-reading helper -- see module docstring)
# ============================================================================

@dataclass
class FingerState:
    """All mutable state for ONE finger. Every finger owns its own instance
    (see HandSensingEngine.finger_states) -- nothing here is ever read from
    or written into another finger's FingerState, so one finger's contact/
    slip/glitch state can never leak into another's (same invariant as
    live_predictor_test.py's finger_states list).
    """
    index: int
    label: str
    baseline: dict[str, float] = field(default_factory=dict)
    consecutive_contact_frames: int = 0
    consecutive_no_contact_frames: int = 0
    confirmed_contact: bool = False
    contact_start_time: float | None = None
    glitch_frame_count: int = 0
    # --- Transient spike guard (see SPIKE_JUMP_GUARD_ENABLED) --------------
    # last_accepted_delta: the most recent post-crosstalk raw_delta that was
    # actually accepted (passed BOTH the spike guard and the SANE_DELTA_MAX
    # glitch guard) and fed to the model -- None until the first accepted
    # frame, so the guard can never fire on the very first tick. Updated
    # only on accepted frames, never on discarded ones, so a discarded
    # glitch frame can never become the jump reference.
    # spike_hold_active: True only while the immediately-previous tick was
    # suppressed by the spike guard -- a jump that PERSISTS into the next
    # tick is then accepted as a genuine fast change rather than suppressed
    # again (the "one-tick hold" in that constant's comment).
    last_accepted_delta: dict[str, float] | None = None
    spike_hold_active: bool = False
    last_slip_time: float = float("-inf")
    # (timestamp, raw_force_now) pairs, trimmed to HISTORY_WINDOW_S. Also
    # doubles as the source for the slip-detection peak lookup, so there is
    # exactly one rolling-force buffer per finger, not two. Deliberately
    # holds the RAW (unsmoothed) force -- see _process_one_finger -- so the
    # history chart still shows real, fast drops and slip detection isn't
    # delayed by display smoothing.
    force_history: deque[tuple[float, float]] = field(default_factory=deque)
    # Separate, display-only rolling buffer for FORCE_SMOOTHING_WINDOW --
    # feeds only FingerResult.force, never read by slip detection/hysteresis.
    display_force_window: deque[float] = field(
        default_factory=lambda: deque(maxlen=FORCE_SMOOTHING_WINDOW))
    # --- Diagnostics: raw (pre-hysteresis) Contact-classifier toggle rate --
    # last_raw_contact is the previous tick's raw result["Contact"] (0/1);
    # contact_toggle_times holds a timestamp for every tick where raw
    # Contact differed from the previous tick, trimmed to
    # TOGGLE_RATE_WINDOW_S -- see _process_one_finger.
    last_raw_contact: int | None = None
    contact_toggle_times: deque[float] = field(default_factory=deque)
    # --- Periodic baseline recalibration (see RECALIBRATION_INTERVAL_S) ----
    # no_contact_since: wall-clock time this finger's confirmed_contact most
    # recently became False; reset to None the instant contact is confirmed
    # again, so a recalibration can never straddle a press.
    no_contact_since: float | None = None
    # Raw (pre-baseline-subtraction) samples collected while confirmed_contact
    # is False, cleared the instant contact is confirmed -- so a commit can
    # only ever average samples from the current, uninterrupted no-contact
    # stretch, never samples left over from an earlier one.
    recal_samples: deque[dict[str, float]] = field(
        default_factory=lambda: deque(maxlen=RECAL_SAMPLE_BUFFER_MAXLEN))
    # Wall-clock time of this finger's last recalibration (successful or the
    # engine's initial startup calibration) -- gates RECALIBRATION_INTERVAL_S.
    # Set to the real startup-calibration time by
    # HandSensingEngine.finish_calibration(), not left at 0.0, so the first
    # periodic check is timed from when this finger's baseline actually
    # became valid.
    last_recal_time: float = 0.0
    # Wall-clock time this finger's baseline was ACTUALLY last refreshed
    # (startup calibration, a successful periodic/requested commit, or a
    # manual reset) -- ALWAYS a real timestamp, never the 0.0 sentinel
    # HandSensingEngine.request_recalibration() writes into last_recal_time
    # above to clear the RECALIBRATION_INTERVAL_S gate. That sentinel is
    # exactly why last_recal_time can't ALSO be reused to detect "has this
    # finger been stuck an unusually long time" (see
    # RECALIBRATION_STUCK_THRESHOLD_S): `now - 0.0` against a real
    # (epoch-scale) `now` is astronomically large immediately after ANY
    # ordinary post-move request, not just after a genuine multi-second
    # cascade of blocked commits. last_successful_recal_time instead
    # directly measures how stale the CURRENT baseline actually is,
    # updated in the same places last_recal_time is set to a real
    # timestamp (finish_calibration, a successful _process_one_finger
    # commit, start_manual_recalibration) but left untouched by
    # request_recalibration()'s gate-clearing sentinel write.
    last_successful_recal_time: float = 0.0
    last_result: "FingerResult | None" = None
    # --- "low_pass" BASELINE_TRACKING_MODE only (see that constant) --------
    # last_recal_time/last_successful_recal_time above are REUSED for this
    # mode too, not duplicated: last_recal_time becomes "last time the
    # throttled status message was emitted" (still gates BASELINE_LOW_PASS_
    # REPORT_INTERVAL_S the same way it gated RECALIBRATION_INTERVAL_S
    # before), last_successful_recal_time becomes "last time the baseline
    # was actually updated" (now essentially every eligible tick, so it
    # stays fresh far more often than under stable_window -- the direct,
    # intended effect of fixing starvation). baseline_update_count/
    # baseline_at_last_report are genuinely NEW, low_pass-specific state:
    baseline_update_count: int = 0
    # Snapshot of `baseline` at the last status report -- lets the report
    # show drift ACCUMULATED SINCE THAT REPORT (max per-channel shift),
    # mirroring stable_window's own "max channel shift" figure, without
    # needing a discrete before/after commit pair the way that mode has.
    # None until the first report (see _update_baseline_low_pass()).
    baseline_at_last_report: "dict[str, float] | None" = None
    # Wall-clock time of this finger's most recent confirmed_contact
    # True->False transition (set in _process_one_finger), regardless of
    # BASELINE_TRACKING_MODE; None until the first release. Gates
    # BASELINE_LOW_PASS_POST_RELEASE_COOLDOWN_S in _update_baseline_low_pass.
    release_time: float | None = None
    # --- AUTO_REBASELINE_ON_RETURN_TO_OPEN_ENABLED only (see that constant) -
    # Previous tick's in_own_transit value -- lets _update_open_position_
    # rebaseline() detect the exact tick a transit completes (edge, not
    # level) without needing HandSensingEngine.poll() to pass anything new.
    was_in_own_transit: bool = False
    # Wall-clock time this finger arrived at the open reference and started
    # collecting fresh samples; None while no rebaseline is pending.
    open_rebaseline_pending_since: float | None = None
    # Raw (pre-baseline-subtraction) samples collected since arriving,
    # regardless of confirmed_contact -- the deliberate exception this
    # feature exists for.
    open_rebaseline_samples: "deque[dict[str, float]]" = field(
        default_factory=lambda: deque(maxlen=REBASELINE_OPEN_MAX_SAMPLES))


@dataclass
class FingerResult:
    """One finger's outcome for a single displayed frame -- everything a
    FingerPanel needs to redraw itself, with no reference back into the
    engine's mutable FingerState (history is a snapshot list, not the live
    deque), so the GUI layer can never accidentally mutate backend state.
    """
    index: int
    label: str
    glitched: bool
    force: float          # displayed Force (N): 0 when not in contact,
                           # floored at 0 otherwise -- see console's Force
                           # floor comment in live_predictor_test.py
    raw_force: float       # true regression output, unclamped/un-overridden
    theta: float
    phi: int
    # Max Phi class probability for this frame (the classifier's own
    # confidence in `phi`), regardless of contact state -- feeds the
    # display-only Phi gate (see PHI_DISPLAY_MIN_CONF); carried over from
    # the last good frame on glitched ticks like the other reused fields.
    phi_conf: float
    contact: bool           # smoothed/confirmed contact state
    contact_duration_s: float | None
    slip_active: bool        # True while the panel should show red
    slip_event: bool         # True only on the tick a NEW slip was flagged
    slip_message: str | None  # log line for a new slip event, else None
    history: list[tuple[float, float]]
    # --- Diagnostics (see TOGGLE_RATE_WINDOW_S/DIAG_PRINT_INTERVAL_S) ------
    raw_contact: int | None       # last known raw (pre-hysteresis) Contact
                                   # classifier output; None if never seen
    contact_toggle_rate_hz: float  # raw Contact 0<->1 transitions/sec over
                                    # the trailing TOGGLE_RATE_WINDOW_S
    consecutive_contact_frames: int
    consecutive_no_contact_frames: int
    glitch_frame_count: int
    # log line for a newly-committed baseline recalibration on this tick,
    # else None -- see RECALIBRATION_INTERVAL_S.
    recal_message: str | None


def _update_toggle_rate(fstate: FingerState, now: float) -> float:
    """Ages contact_toggle_times out to TOGGLE_RATE_WINDOW_S and returns this
    finger's current raw-Contact toggle rate (transitions/sec). Called on
    every tick -- including glitched ones -- so old transitions age out by
    wall-clock time even when no fresh raw Contact value was observed this
    tick, keeping the rate reading accurate rather than frozen mid-window.
    """
    while fstate.contact_toggle_times and now - fstate.contact_toggle_times[0] > TOGGLE_RATE_WINDOW_S:
        fstate.contact_toggle_times.popleft()
    return len(fstate.contact_toggle_times) / TOGGLE_RATE_WINDOW_S


def _update_baseline_stable_window(fstate: FingerState, raw_absolute: dict[str, float], now: float,
                                    no_finger_in_transit: bool, raw_cols: list[str]) -> str | None:
    """BASELINE_TRACKING_MODE="stable_window": the ORIGINAL recalibration
    mechanism, byte-identical to its pre-2026-07-15 behaviour EXCEPT for
    the contamination gate (`no_finger_in_transit`) -- see that constant's
    module comment for why this mode is kept at all (fallback) and what it
    still doesn't fix (starvation under fast touch/release cadences).

    Contamination gate placement: applied to SAMPLE COLLECTION (a tick
    where any finger anywhere is mid-transit is simply not appended to
    recal_samples), not to the no_contact_since STABLE-WINDOW TIMER, which
    keeps running off confirmed_contact alone exactly as before -- a brief
    neighbour transit during an otherwise-clean window delays the commit
    only by however many samples it caused to be skipped (naturally via
    RECALIBRATION_MIN_SAMPLES still needing to be met by clean samples),
    not by resetting the whole window the way losing confirmed_contact
    does. The commit condition itself ALSO re-checks no_finger_in_transit
    as a final guard, so a commit can never land on the exact tick a
    transit starts either.
    """
    if fstate.confirmed_contact:
        fstate.no_contact_since = None
        fstate.recal_samples.clear()
    else:
        if fstate.no_contact_since is None:
            fstate.no_contact_since = now
        if no_finger_in_transit:
            fstate.recal_samples.append(raw_absolute)

    is_recal_stuck = now - fstate.last_successful_recal_time >= RECALIBRATION_STUCK_THRESHOLD_S
    required_stable_no_contact_s = (
        RECALIBRATION_STUCK_MIN_STABLE_NO_CONTACT_S if is_recal_stuck
        else RECALIBRATION_MIN_STABLE_NO_CONTACT_S
    )

    if (not fstate.confirmed_contact
            and fstate.no_contact_since is not None
            and now - fstate.no_contact_since >= required_stable_no_contact_s
            and now - fstate.last_recal_time >= RECALIBRATION_INTERVAL_S
            and len(fstate.recal_samples) >= RECALIBRATION_MIN_SAMPLES
            and no_finger_in_transit):
        samples = list(fstate.recal_samples)
        new_baseline = {col: sum(s[col] for s in samples) / len(samples) for col in raw_cols}
        max_shift = max(abs(new_baseline[c] - fstate.baseline[c]) for c in raw_cols)
        stuck_note = (
            f", STUCK-RECOVERY: {now - fstate.last_successful_recal_time:.1f}s since last "
            f"successful recalibration, used relaxed {required_stable_no_contact_s:.1f}s "
            f"stable-window threshold instead of the normal "
            f"{RECALIBRATION_MIN_STABLE_NO_CONTACT_S:.1f}s"
        ) if is_recal_stuck else ""
        fstate.baseline = new_baseline
        fstate.last_recal_time = now
        fstate.last_successful_recal_time = now
        return (
            f"{fstate.label}: baseline recalibrated ({len(samples)} samples over "
            f"{now - fstate.no_contact_since:.1f}s stable no-contact, "
            f"max channel shift {max_shift:.2f}{stuck_note})"
        )
    return None


def _update_baseline_low_pass(fstate: FingerState, raw_absolute: dict[str, float], now: float,
                               no_finger_in_transit: bool, in_own_transit: bool,
                               raw_cols: list[str]) -> str | None:
    """BASELINE_TRACKING_MODE="low_pass" (default): continuously updates
    `fstate.baseline` via an exponential moving average (BASELINE_LOW_PASS_
    ALPHA) on every ELIGIBLE tick, instead of requiring one long
    uninterrupted stable-no-contact window before a discrete commit -- see
    that constant's module comment for why (starvation, confirmed by
    simulation). Eligible = this finger not confirmed-in-contact, not in
    its own transit, AND no finger anywhere else is mid-transit either
    (`no_finger_in_transit` already implies `not in_own_transit` since
    "any" includes this finger -- `in_own_transit` is still checked
    directly too, both for defensive robustness and so this function's own
    eligibility condition reads completely self-contained without relying
    on a caller-side invariant holding).

    No stable-window timer, no sample buffer, no discrete commit -- every
    eligible tick nudges the baseline a little rather than averaging a
    batch collected over time. Returns a throttled (BASELINE_LOW_PASS_
    REPORT_INTERVAL_S) status message summarising drift since the last
    report, or None on ticks that don't report (whether or not an update
    happened) -- purely a GUI-log courtesy, never gates the update itself.

    2026-07-20: eligibility ALSO requires consecutive_contact_frames == 0,
    not just confirmed_contact == False. Without this, a finger actively
    accumulating a confirm streak (genuinely deforming, partway toward
    CONTACT_CONFIRM_FRAMES but not there yet) still counted as "not
    confirmed" and so stayed eligible -- meaning every tick of that
    build-up nudged the baseline toward the very deformation trying to
    confirm, working against its own confirmation. Live evidence this is
    real, not theoretical: the segmentation-free regression in
    post_release_sensitivity_report.md already found baseline_l1_drift has
    an independent NEGATIVE effect on P(contact) (R^2=0.199) -- this closes
    the specific gap that lets drift accumulate mid-touch, not just
    post-release. A truly idle finger has consecutive_contact_frames == 0
    almost always, so this shouldn't meaningfully reduce update frequency
    during genuine rest -- only during an active, ambiguous rise. Revert by
    restoring sensing_gui_BACKUP_pre_partial_streak_baseline_fix_*.py.

    2026-07-20 (same day, second fix): eligibility ALSO requires
    BASELINE_LOW_PASS_POST_RELEASE_COOLDOWN_S to have elapsed since
    fstate.release_time (this finger's most recent confirmed_contact
    True->False transition). The partial-streak fix above closes the gap
    while a touch is actively confirming, but does nothing once a release
    has already fired -- including a SPURIOUS one-tick release triggered by
    a brief signal dip during transit while the finger is still genuinely
    pressed. Without this cooldown, eligibility reopens on the very same
    tick as the release with zero delay: the first eligible tick nudges
    baseline toward the still-elevated raw signal, shrinking raw_delta,
    which drops p_contact, which keeps eligibility open -- a self-
    reinforcing loop with no circuit breaker. Confirmed live via
    record_press_while_traveling.py (continuous unreleased press on Ring
    while stepping 0->100%): baseline_l1_drift climbed unbounded (8.8 -> 1007
    over ~10s) and p_new stayed pinned at 0.000 the whole time, only
    recovering once raw force rose enough (mean 4.14N at 100%) to outrun the
    ~1%/tick drift. The cooldown gives the classifier one clean, unmoved
    baseline for several CONTACT_CONFIRM_FRAMES-worth of ticks right after
    any release, so a genuinely still-present touch reconfirms (which then
    keeps eligibility closed via consecutive_contact_frames==0 above) before
    baseline ever gets a chance to move. Revert by restoring
    sensing_gui_BACKUP_pre_baseline_runaway_fix_*.py.
    """
    cooldown_elapsed = (fstate.release_time is None
                         or now - fstate.release_time >= BASELINE_LOW_PASS_POST_RELEASE_COOLDOWN_S)
    eligible = ((not fstate.confirmed_contact) and fstate.consecutive_contact_frames == 0
                and no_finger_in_transit and not in_own_transit and cooldown_elapsed)
    if eligible:
        for col in raw_cols:
            fstate.baseline[col] += BASELINE_LOW_PASS_ALPHA * (raw_absolute[col] - fstate.baseline[col])
        fstate.baseline_update_count += 1
        fstate.last_successful_recal_time = now

    if fstate.baseline_at_last_report is None:
        fstate.baseline_at_last_report = dict(fstate.baseline)
        fstate.last_recal_time = now
        return None

    if now - fstate.last_recal_time >= BASELINE_LOW_PASS_REPORT_INTERVAL_S:
        max_shift = max(abs(fstate.baseline[c] - fstate.baseline_at_last_report[c]) for c in raw_cols)
        message = (f"{fstate.label}: baseline low-pass tracking -- {fstate.baseline_update_count} "
                   f"updates since start, max channel shift {max_shift:.2f} over the last "
                   f"{now - fstate.last_recal_time:.1f}s")
        fstate.baseline_at_last_report = dict(fstate.baseline)
        fstate.last_recal_time = now
        return message
    return None


def _update_open_position_rebaseline(fstate: FingerState, raw_absolute: dict[str, float], now: float,
                                      my_position_pct: "float | None", in_own_transit: bool,
                                      raw_cols: list[str]) -> "str | None":
    """AUTO_REBASELINE_ON_RETURN_TO_OPEN_ENABLED (see that constant's module
    comment for the full rationale/trade-off): the instant this finger's own
    transit completes AT the fully-open reference position, collects a short
    burst of fresh raw samples and forcibly replaces `fstate.baseline` with
    their average -- regardless of confirmed_contact, the one deliberate
    exception to every other recalibration path in this file. Does NOT
    touch confirmed_contact/consecutive_*_frames directly; replacing the
    baseline just means the NEXT tick's raw_delta (computed against the
    fresh baseline back in _process_one_finger) should read near-zero again,
    so the existing release-streak/Schmitt-release-probability logic
    un-confirms it through its own normal path.

    my_position_pct: this finger's current estimated position (0=open..
    100=closed), or None if the caller hasn't wired position tracking yet
    (same fail-safe-before-wiring posture as crosstalk_models/
    finger_positions elsewhere in this file) -- returns None immediately in
    that case after still updating was_in_own_transit, so this is always
    safe to call.

    Returns a log message the instant a rebaseline commits, else None.
    """
    if my_position_pct is None:
        fstate.was_in_own_transit = in_own_transit
        return None

    just_arrived_at_open = (
        fstate.was_in_own_transit and not in_own_transit
        and my_position_pct <= REBASELINE_OPEN_POSITION_TOLERANCE_PCT
    )
    fstate.was_in_own_transit = in_own_transit

    if just_arrived_at_open:
        fstate.open_rebaseline_pending_since = now
        fstate.open_rebaseline_samples.clear()

    if fstate.open_rebaseline_pending_since is None:
        return None

    # Abort if a new transit starts before this commits -- partial-settle
    # data would otherwise get baked in. A fresh transit re-arms this itself
    # once IT completes at the open reference, if applicable.
    if in_own_transit:
        fstate.open_rebaseline_pending_since = None
        fstate.open_rebaseline_samples.clear()
        return None

    fstate.open_rebaseline_samples.append(raw_absolute)

    if (now - fstate.open_rebaseline_pending_since >= REBASELINE_OPEN_SETTLE_S
            and len(fstate.open_rebaseline_samples) >= REBASELINE_OPEN_MIN_SAMPLES):
        samples = list(fstate.open_rebaseline_samples)
        new_baseline = {col: sum(s[col] for s in samples) / len(samples) for col in raw_cols}
        max_shift = max(abs(new_baseline[c] - fstate.baseline[c]) for c in raw_cols)
        was_stuck_confirmed = fstate.confirmed_contact
        fstate.baseline = new_baseline
        fstate.last_recal_time = now
        fstate.last_successful_recal_time = now
        fstate.open_rebaseline_pending_since = None
        fstate.open_rebaseline_samples.clear()
        stuck_note = (" (was confirmed_contact=True -- should clear within the normal "
                      "release streak on the next few ticks)" if was_stuck_confirmed else "")
        return (
            f"{fstate.label}: auto-rebaselined at open position "
            f"({len(samples)} samples, max channel shift {max_shift:.2f}){stuck_note}"
        )
    return None


def _process_one_finger(predictor: LivePhiClassPredictor, fstate: FingerState,
                         frame: FrameData, now: float,
                         crosstalk_models: "dict[str, dict | None] | None" = None,
                         finger_positions: "dict[str, float] | None" = None,
                         settled_finger_positions: "dict[str, float] | None" = None,
                         in_own_transit: bool = False,
                         any_finger_in_transit: bool = False) -> FingerResult:
    """Baseline delta -> crosstalk correction -> glitch guard -> predict ->
    hysteresis -> slip check, for exactly one finger. Mirrors
    live_predictor_test.py's process_frame loop body finger-for-finger (same
    constants, same formulas), just returning a result object instead of
    printing.

    in_own_transit: True while THIS finger's own commanded-position ramp is
    still in progress (see CONTACT_SUPPRESS_DURING_OWN_TRANSIT's comment) --
    default False so this stays safe to call before a caller wires up
    HandSensingEngine's transit tracking, same fail-safe posture as
    crosstalk_models/finger_positions below.

    any_finger_in_transit: True while ANY of the 4 fingers (this one
    included) has an active transit ramp -- feeds the baseline-tracking
    contamination gate (see BASELINE_TRACKING_MODE's module comment /
    _update_baseline_stable_window() / _update_baseline_low_pass()).
    Default False for the same fail-safe-before-wiring reason as
    in_own_transit above.

    crosstalk_models/finger_positions are optional (default None -- no
    correction applied, identical to this function's behaviour before the
    crosstalk feature existed) so this stays safe to call even before a
    caller wires up HandSensingEngine.crosstalk_models/set_finger_position.
    crosstalk_models is keyed by backend name ("pairwise_additive"/
    "nearest_neighbor", see CROSSTALK_CORRECTION_BACKENDS) rather than being
    a single model, since which backend applies is now a per-finger choice
    (see CROSSTALK_CORRECTION_SOURCE) -- a finger whose configured backend's
    model is unavailable (missing key, or that entry is None) gets no
    correction for that tick rather than raising, same fail-safe posture the
    single-model version already had.

    settled_finger_positions (2026-07-20): if given, used INSTEAD of
    finger_positions when looking up OTHER fingers' positions for crosstalk
    correction specifically (this finger's OWN entry -- e.g. the auto-
    rebaseline-on-return-to-open position check -- still reads from
    finger_positions, unaffected). Optional, defaults to None -- falls back
    to finger_positions, byte-identical to this function's behaviour before
    this parameter existed, so every existing caller that doesn't wire it
    up (the true_zero_dataset scripts, candidate_v2-v5, etc.) is unaffected.
    See HandSensingEngine.poll()'s own settled_positions comment for why
    this exists: crosstalk models were only ever fit on static, settled
    combinations, never validated against a moving neighbour's live,
    guessed intermediate position.
    """
    raw_absolute = frame_to_raw_values(frame, predictor.raw_cols, fstate.index)
    raw_delta = {col: v - fstate.baseline[col] for col, v in raw_absolute.items()}

    # --- Crosstalk correction (additive, on top of the static/periodic
    # baseline above -- see CROSSTALK_CORRECTION_ENABLED) ---------------------
    # Both backends' fitted/measured functions represent delta values
    # measured relative to ALL 4 fingers sitting at the 0% (open) reference
    # position (see model_v1/crosstalk/README.md's "Dataset" section, and
    # model_v1/crosstalk_lookup/self_effect.py's module docstring for why
    # the nearest-neighbor backend's target is specifically corrected to
    # match this same "shift from others only" semantics) -- the exact same
    # reference fstate.baseline was itself captured against (the startup
    # calibration, and ordinarily each periodic recalibration, both run
    # with the hand at rest). So subtracting the selected backend's
    # predicted shift from raw_delta here estimates "what raw_delta would
    # be if the other fingers were still at 0%", isolating the contact-
    # driven component from ordinary hand-shape crosstalk. One documented
    # edge case: a periodic recalibration that happens to commit while a
    # neighbour is mid-motion would bake that neighbour's THEN-current
    # crosstalk into the new baseline -- not corrected for here, since doing
    # so would mean touching the recalibration logic itself, which this
    # integration is required not to alter or weaken.
    if CROSSTALK_CORRECTION_ENABLED and crosstalk_models is not None and finger_positions is not None:
        my_label = CROSSTALK_FINGER_LABELS[fstate.index]
        source = CROSSTALK_CORRECTION_SOURCE.get(my_label, "pairwise_additive")
        model = crosstalk_models.get(source)
        if model is not None:
            # Prefer settled_finger_positions for the OTHER fingers' values
            # (see this function's own docstring + HandSensingEngine.poll()'s
            # settled_positions comment) -- falls back to finger_positions
            # if not provided, byte-identical to pre-2026-07-20 behaviour.
            neighbor_source = settled_finger_positions if settled_finger_positions is not None else finger_positions
            other_positions = {lbl: pos for lbl, pos in neighbor_source.items() if lbl != my_label}
            predict_fn = CROSSTALK_CORRECTION_BACKENDS[source]
            predicted_crosstalk = predict_fn(model, my_label, other_positions)
            raw_delta = {col: v - predicted_crosstalk.get(col, 0.0) for col, v in raw_delta.items()}

    # --- Transient spike guard (see SPIKE_JUMP_GUARD_ENABLED) ---------------
    # Applied to the FINAL raw_delta (post-baseline, post-crosstalk), i.e.
    # exactly what the model would otherwise be fed. A suspicious jump is
    # routed through the same discard-and-reuse-last-result path as the
    # SANE_DELTA_MAX glitch guard below; spike_hold_active makes that
    # suppression one-tick-only (a persisting jump is accepted next tick as
    # a genuine fast change -- see the constant's comment).
    spike_glitch = False
    if (SPIKE_JUMP_GUARD_ENABLED
            and fstate.last_accepted_delta is not None
            and not fstate.spike_hold_active):
        max_jump = max(abs(raw_delta[col] - fstate.last_accepted_delta[col])
                       for col in raw_delta)
        if max_jump > SPIKE_JUMP_MAX_PER_TICK:
            spike_glitch = True
            fstate.spike_hold_active = True

    slip_visual = (now - fstate.last_slip_time) < SLIP_DISPLAY_HOLD_S

    # Sensor-glitch guard, applied to this finger only -- a glitch on one
    # finger never discards or alters another finger's frame (each finger
    # is processed independently by HandSensingEngine.poll()'s loop). This
    # finger's hysteresis/contact/slip-hold state is left untouched, and the
    # last known good result is reused for display, exactly like the
    # console tool's glitch guard.
    glitched_cols = {col: val for col, val in raw_delta.items() if abs(val) > SANE_DELTA_MAX}
    if glitched_cols or spike_glitch:
        # spike_glitch (see the spike guard above) reuses this exact
        # discard path -- last result redisplayed, hysteresis/contact/slip
        # state untouched, glitch_frame_count incremented.
        fstate.glitch_frame_count += 1
        toggle_rate = _update_toggle_rate(fstate, now)
        last = fstate.last_result
        result = FingerResult(
            index=fstate.index, label=fstate.label, glitched=True,
            force=last.force if last else 0.0,
            raw_force=last.raw_force if last else 0.0,
            theta=last.theta if last else 0.0,
            phi=last.phi if last else 0,
            phi_conf=last.phi_conf if last else 0.0,
            contact=fstate.confirmed_contact,
            contact_duration_s=(now - fstate.contact_start_time)
            if fstate.contact_start_time is not None else None,
            slip_active=slip_visual, slip_event=False, slip_message=None,
            history=list(fstate.force_history),
            raw_contact=fstate.last_raw_contact,
            contact_toggle_rate_hz=toggle_rate,
            consecutive_contact_frames=fstate.consecutive_contact_frames,
            consecutive_no_contact_frames=fstate.consecutive_no_contact_frames,
            glitch_frame_count=fstate.glitch_frame_count,
            recal_message=None,
        )
        fstate.last_result = result
        if FORCE_TRACE_ENABLED:
            print(f"[FTRACE][{fstate.label}] GLITCHED -- frame discarded, "
                  f"reusing last displayed Force={result.force:.3f}N  "
                  f"confirmed_contact={fstate.confirmed_contact}  "
                  f"streak(contact/no-contact)={fstate.consecutive_contact_frames}/"
                  f"{fstate.consecutive_no_contact_frames}", flush=True)
        return result

    # Frame accepted by BOTH guards (spike + SANE_DELTA_MAX) -- it becomes
    # the spike guard's new jump reference. Deliberately only updated here,
    # never on a discarded frame, so a glitch can never poison the
    # reference and make the NEXT genuine frame look like a jump.
    fstate.spike_hold_active = False
    fstate.last_accepted_delta = dict(raw_delta)

    result = predictor.predict(raw_delta)

    # Per-finger Force calibration (see FORCE_CALIBRATORS' own comment) --
    # applied here, immediately after the raw prediction and before
    # anything downstream (contact gating, raw_force_now, fr.raw_force,
    # fr.force) consumes result["Force"], so every consumer sees the same
    # corrected value consistently rather than calibrating in several
    # places. A no-op for any finger without a calibrator entry.
    calibrator = FORCE_CALIBRATORS.get(CROSSTALK_FINGER_LABELS[fstate.index])
    if calibrator is not None:
        result["Force"] = float(calibrator.predict([result["Force"]])[0])

    was_confirmed = fstate.confirmed_contact

    # --- Diagnostics: raw (pre-hysteresis) Contact toggle rate --------------
    # Recorded BEFORE the hysteresis update below touches anything, so this
    # reflects the classifier's true per-frame output, not the smoothed
    # confirmed_contact state -- this is what answers "is the raw classifier
    # oscillating at the boundary" independently of whether
    # CONTACT_CONFIRM_FRAMES/CONTACT_RELEASE_FRAMES are currently masking it.
    raw_contact = result["Contact"]
    if fstate.last_raw_contact is not None and raw_contact != fstate.last_raw_contact:
        fstate.contact_toggle_times.append(now)
    fstate.last_raw_contact = raw_contact
    toggle_rate = _update_toggle_rate(fstate, now)

    # Streak-counter hysteresis, fed by either the probability Schmitt
    # trigger (CONTACT_PROB_SCHMITT_ENABLED=True, the 2026-07-17 production-
    # review default -- see that constant's comment) or the original hard
    # predict()==1 signal (flag False -- byte-identical to
    # live_predictor_test.py's process_frame logic), applied to this
    # finger's own counters only. Either way, the MIN_CONTACT_FORCE_N gate
    # only affects which frames count toward the confirm streak --
    # raw_contact/toggle_rate above stay based on the classifier's true,
    # ungated output, so the diagnostics still show what the classifier
    # alone is doing.
    #
    # CONTACT_SUPPRESS_DURING_OWN_TRANSIT: while this finger's own transit
    # ramp is active, the streak counters and confirmed_contact are frozen
    # at whatever they were the instant transit began -- neither advances
    # nor resets, so a partial pre-transit streak resumes exactly where it
    # left off once the ramp completes rather than being discarded. Applied
    # AFTER the raw_contact/toggle_rate diagnostics above (unaffected) and
    # BEFORE contact_start_time/recalibration-sample-collection below (which
    # both key off confirmed_contact and therefore inherit the freeze
    # automatically, with no separate special-casing needed).
    suppress_this_tick = CONTACT_SUPPRESS_DURING_OWN_TRANSIT and in_own_transit
    if not suppress_this_tick:
        # Two-threshold probability Schmitt trigger (see
        # CONTACT_PROB_SCHMITT_ENABLED's comment for the validation
        # numbers). With the flag off, release_signal is the exact
        # complement of confirm_signal, so the dead-zone hold below is
        # unreachable and the streak behaviour is byte-identical to the
        # pre-review single-signal logic.
        if CONTACT_PROB_SCHMITT_ENABLED:
            # Per-finger override (see CONTACT_P_CONFIRM_PER_FINGER's own
            # comment) -- falls back to the global CONTACT_P_CONFIRM/
            # CONTACT_P_RELEASE for any finger not listed, so this is a
            # no-op for Middle/Ring/Thumb and byte-identical to the
            # pre-override behaviour for them.
            my_label = CROSSTALK_FINGER_LABELS[fstate.index]
            confirm_thr = CONTACT_P_CONFIRM_PER_FINGER.get(my_label, CONTACT_P_CONFIRM)
            release_thr = CONTACT_P_RELEASE_PER_FINGER.get(my_label, CONTACT_P_RELEASE)
            p_contact = result["Contact_probs"][1]
            confirm_signal = (p_contact >= confirm_thr
                              and result["Force"] > MIN_CONTACT_FORCE_N)
            release_signal = p_contact <= release_thr
        else:
            confirm_signal = result["Contact"] == 1 and result["Force"] > MIN_CONTACT_FORCE_N
            release_signal = not confirm_signal
        if confirm_signal:
            fstate.consecutive_contact_frames += 1
            fstate.consecutive_no_contact_frames = 0
        elif release_signal:
            fstate.consecutive_no_contact_frames += 1
            fstate.consecutive_contact_frames = 0
        # else: dead zone (CONTACT_P_RELEASE < P < CONTACT_P_CONFIRM, or
        # P >= CONTACT_P_CONFIRM with Force under the MIN_CONTACT_FORCE_N
        # gate) -- BOTH streak counters hold; that hold is the point of a
        # Schmitt trigger.

        if not fstate.confirmed_contact:
            if fstate.consecutive_contact_frames >= CONTACT_CONFIRM_FRAMES:
                fstate.confirmed_contact = True
        else:
            if fstate.consecutive_no_contact_frames >= CONTACT_RELEASE_FRAMES:
                fstate.confirmed_contact = False
                fstate.release_time = now

    # Contact-duration timer: starts the instant confirmed_contact flips on,
    # resets to "not running" the instant it flips off, restarted fresh on
    # the next contact -- per the GUI spec.
    if fstate.confirmed_contact and fstate.contact_start_time is None:
        fstate.contact_start_time = now
    elif not fstate.confirmed_contact:
        fstate.contact_start_time = None

    # --- Baseline tracking (see BASELINE_TRACKING_MODE) ----------------------
    # Contamination gate, applies to BOTH modes below: a finger's baseline
    # should never absorb samples captured while ANY finger anywhere on the
    # hand is mid-transit -- even a perfectly stationary updating finger is
    # picking up genuine geometric crosstalk from a moving neighbour at that
    # instant, not a clean "at rest" reading.
    no_finger_in_transit = not any_finger_in_transit
    if BASELINE_TRACKING_MODE == "low_pass":
        recal_message = _update_baseline_low_pass(
            fstate, raw_absolute, now, no_finger_in_transit, in_own_transit, predictor.raw_cols)
    else:
        recal_message = _update_baseline_stable_window(
            fstate, raw_absolute, now, no_finger_in_transit, predictor.raw_cols)

    # --- Auto-rebaseline on return to open (see AUTO_REBASELINE_ON_RETURN_
    # TO_OPEN_ENABLED) -- independent of, and evaluated in addition to,
    # BASELINE_TRACKING_MODE above; this is the one deliberate exception to
    # the "never recalibrate while confirmed" rule both of those modes
    # otherwise enforce. Runs unconditionally (own internal flag check)
    # rather than being folded into either function above, since it targets
    # a specific POSITION event (arriving at the open reference), not a
    # general periodic/continuous update policy.
    if AUTO_REBASELINE_ON_RETURN_TO_OPEN_ENABLED:
        my_label = CROSSTALK_FINGER_LABELS[fstate.index]
        my_position_pct = finger_positions.get(my_label) if finger_positions is not None else None
        rebaseline_message = _update_open_position_rebaseline(
            fstate, raw_absolute, now, my_position_pct, in_own_transit, predictor.raw_cols)
        if rebaseline_message is not None:
            recal_message = rebaseline_message

    # raw_force_now is the same Force-floored, contact-gated value the
    # console tool displays -- used below for slip detection and the
    # history chart. It is intentionally NEVER smoothed or deadbanded, so
    # slip detection keeps reacting to a real, fast drop with no added lag
    # (see FORCE_DEADBAND_N/FORCE_SMOOTHING_WINDOW, applied further down,
    # for the separate display-only value).
    if fstate.confirmed_contact:
        raw_force_now = max(0.0, result["Force"])  # Force floor
        theta = result["Theta"]
        phi = result["Phi"]
    else:
        raw_force_now = 0.0
        theta = 0.0
        phi = 0

    # Slip detection: a simple, named-constant-driven rule -- see the
    # SLIP_DROP_FRACTION/SLIP_WINDOW_S comments above. peak is computed from
    # history BEFORE this tick's sample is appended, so a finger can't
    # "slip relative to itself" on the very first reading.
    #
    # A NEW slip is only logged/flagged if at least SLIP_DISPLAY_HOLD_S has
    # elapsed since this finger's last one (reusing the same constant that
    # already controls how long the panel stays red) -- NOT a bare
    # single-tick edge trigger on slip_now flipping False->True. A noisy or
    # lightly-wobbling touch can dip below and briefly recover back above
    # the SLIP_DROP_FRACTION threshold several times within well under a
    # second (observed live: one finger crossed it 3 times within ~360ms,
    # every time measured against the SAME recent peak, none landing on
    # 0.00N -- i.e. one continuous wobbly touch, not 3 separate physical
    # slips). An edge trigger logs each of those crossings as its own
    # "slip," which is noise amplification; the cooldown collapses a burst
    # like that into a single logged event per touch episode.
    slip_event = False
    slip_message = None
    recent = [f for t, f in fstate.force_history if now - t <= SLIP_WINDOW_S]
    peak = max(recent) if recent else raw_force_now
    slip_now = was_confirmed and peak > 0 and (peak - raw_force_now) / peak >= SLIP_DROP_FRACTION
    if slip_now and (now - fstate.last_slip_time) >= SLIP_DISPLAY_HOLD_S:
        fstate.last_slip_time = now
        slip_event = True
        slip_message = (
            f"{fstate.label}: Slip detected -- Force {peak:.2f} N -> {raw_force_now:.2f} N "
            f"({(peak - raw_force_now) / peak:.0%} drop within {SLIP_WINDOW_S:.1f}s)"
        )
    slip_visual = slip_visual or slip_event

    fstate.force_history.append((now, raw_force_now))
    while fstate.force_history and now - fstate.force_history[0][0] > HISTORY_WINDOW_S:
        fstate.force_history.popleft()

    # --- Display-only smoothing + settling deadband -------------------------
    # Feeds only FingerResult.force (the numeric readout + gauge bar) --
    # everything above (hysteresis, contact_start_time, slip detection,
    # force_history/the history chart) already used raw_force_now and is
    # untouched by either of these.
    fstate.display_force_window.append(raw_force_now)
    smoothed_force = sum(fstate.display_force_window) / len(fstate.display_force_window)
    deadband_applied = not fstate.confirmed_contact and smoothed_force < FORCE_DEADBAND_N
    displayed_force = 0.0 if deadband_applied else smoothed_force

    fr = FingerResult(
        index=fstate.index, label=fstate.label, glitched=False,
        force=displayed_force, raw_force=result["Force"], theta=theta, phi=phi,
        phi_conf=max(result["Phi_probs"].values()),
        contact=fstate.confirmed_contact,
        contact_duration_s=(now - fstate.contact_start_time)
        if fstate.contact_start_time is not None else None,
        slip_active=slip_visual, slip_event=slip_event, slip_message=slip_message,
        history=list(fstate.force_history),
        raw_contact=raw_contact,
        contact_toggle_rate_hz=toggle_rate,
        consecutive_contact_frames=fstate.consecutive_contact_frames,
        consecutive_no_contact_frames=fstate.consecutive_no_contact_frames,
        glitch_frame_count=fstate.glitch_frame_count,
        recal_message=recal_message,
    )
    fstate.last_result = fr

    if FORCE_TRACE_ENABLED:
        no_contact_age = (f"{now - fstate.no_contact_since:.2f}s"
                           if fstate.no_contact_since is not None else "n/a")
        since_recal = now - fstate.last_recal_time
        # Distinct from since_recal above: last_recal_time can be the 0.0
        # gate-clear sentinel (see FingerState.last_recal_time), which makes
        # since_recal astronomically large right after ANY ordinary post-
        # move request -- not a meaningful "how stale is the baseline"
        # reading on its own. since_successful_recal uses
        # last_successful_recal_time instead (always a real timestamp), so
        # it's the honest answer to that question, and is also exactly what
        # RECALIBRATION_STUCK_THRESHOLD_S/is_recal_stuck above are computed
        # from.
        since_successful_recal = now - fstate.last_successful_recal_time
        print(f"[FTRACE][{fstate.label}] "
              f"raw_Force={result['Force']:+7.3f}N  "
              f"raw_force_now(floored+gated)={raw_force_now:+6.3f}N  "
              f"smoothed_force={smoothed_force:+6.3f}N  "
              f"deadband_applied={deadband_applied}  "
              f"displayed_force={displayed_force:+6.3f}N  |  "
              f"confirmed_contact={fstate.confirmed_contact}  "
              f"streak(contact/no-contact)={fstate.consecutive_contact_frames}/"
              f"{fstate.consecutive_no_contact_frames}  "
              f"contact_eval_suppressed(own_transit)={suppress_this_tick}  |  "
              f"no_contact_since_age={no_contact_age}  "
              f"since_last_recal={since_recal:.1f}s  "
              f"since_successful_recal={since_successful_recal:.1f}s  "
              f"recal_stuck={is_recal_stuck}  "
              f"recal_committed_this_tick={recal_message is not None}",
              flush=True)

    return fr


class HandSensingEngine(QtCore.QObject):
    """Owns the EfleshController connection, per-finger baseline
    calibration, and per-finger prediction state. Qt-signal boundary for
    connect/status/error notifications only -- frame data itself never goes
    through a signal (see poll()) so a slow GUI thread can't cause a serial
    read backlog.
    """

    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    connected = QtCore.pyqtSignal(str)
    disconnected = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictor = LivePhiClassPredictor(model_path=str(MODEL_PATH))
        self.finger_states = [FingerState(index=i, label=f"Finger {i + 1}")
                               for i in range(NUM_FINGERS)]
        self.ctrl = EfleshController()
        self.calibrating = True

        self._frame_lock = threading.Lock()
        self._latest_frame: FrameData | None = None
        self._last_seq_shown: int | None = None
        self._baseline_samples: list[list[dict]] = [[] for _ in range(NUM_FINGERS)]

        # --- Crosstalk correction (see CROSSTALK_CORRECTION_ENABLED) --------
        # Both backends loaded once here, same "load once at startup"
        # pattern as self.predictor above -- which one actually applies to
        # a given finger is decided per-finger by CROSSTALK_CORRECTION_
        # SOURCE at the point of use (_process_one_finger), not here.
        # Failure to load either one independently degrades that entry to
        # None rather than raising -- _process_one_finger already treats a
        # missing/None model for a finger's configured backend as "no
        # correction for this tick", so a missing/broken model (of either
        # kind) can never crash or block the rest of the sensing pipeline
        # from starting. If nn_lookup_model_v1.json specifically fails to
        # load, any finger configured for "nearest_neighbor" transparently
        # gets no correction rather than falling back to pairwise_additive
        # -- silently swapping which physical correction a finger receives
        # on a load failure would be more surprising than simply disabling
        # it, and the fallback print below makes the degraded state visible
        # either way. Reported via print(), not self.status.emit() -- this
        # runs inside __init__, before MainWindow has had a chance to
        # connect self.status to anything (see MainWindow.__init__), so an
        # emit() here would be a signal with no listener yet and would
        # silently never be seen.
        self.crosstalk_models: dict[str, dict | None] = {
            "pairwise_additive": None, "nearest_neighbor": None, "nearest_neighbor_smoothed": None,
            "ridge_quadratic": None, "magnetfree_index": None,
        }
        if CROSSTALK_CORRECTION_ENABLED:
            try:
                self.crosstalk_models["pairwise_additive"] = load_crosstalk_model(CROSSTALK_MODEL_PATH)
                print(f"Crosstalk correction: loaded {CROSSTALK_MODEL_PATH.name} (pairwise_additive)")
            except Exception as exc:
                print(f"Crosstalk correction: FAILED to load {CROSSTALK_MODEL_PATH.name} "
                      f"({exc}) -- fingers configured for pairwise_additive get no correction.")
            try:
                nn_model = load_nn_lookup_model(NN_LOOKUP_MODEL_PATH)
                # "nearest_neighbor" and "nearest_neighbor_smoothed" are two
                # different PREDICT FUNCTIONS (see CROSSTALK_CORRECTION_
                # BACKENDS) over the exact same underlying measured data --
                # one load, both keys point at the same model object.
                self.crosstalk_models["nearest_neighbor"] = nn_model
                self.crosstalk_models["nearest_neighbor_smoothed"] = nn_model
                print(f"Crosstalk correction: loaded {NN_LOOKUP_MODEL_PATH.name} "
                      f"(nearest_neighbor, nearest_neighbor_smoothed)")
            except Exception as exc:
                print(f"Crosstalk correction: FAILED to load {NN_LOOKUP_MODEL_PATH.name} "
                      f"({exc}) -- fingers configured for nearest_neighbor/nearest_neighbor_smoothed "
                      f"get no correction.")
            try:
                self.crosstalk_models["ridge_quadratic"] = load_ridge_crosstalk_model(
                    RIDGE_CROSSTALK_MODEL_PATH)
                print(f"Crosstalk correction: loaded {RIDGE_CROSSTALK_MODEL_PATH.name} "
                      f"(ridge_quadratic)")
            except Exception as exc:
                print(f"Crosstalk correction: FAILED to load {RIDGE_CROSSTALK_MODEL_PATH.name} "
                      f"({exc}) -- fingers configured for ridge_quadratic get no correction.")
            try:
                self.crosstalk_models["magnetfree_index"] = load_magnetfree_index_model(
                    MAGNETFREE_INDEX_MODEL_PATH)
                print(f"Crosstalk correction: loaded {MAGNETFREE_INDEX_MODEL_PATH.name} "
                      f"(magnetfree_index, Index-only)")
            except Exception as exc:
                print(f"Crosstalk correction: FAILED to load {MAGNETFREE_INDEX_MODEL_PATH.name} "
                      f"({exc}) -- fingers configured for magnetfree_index get no correction.")
            for label, source in CROSSTALK_CORRECTION_SOURCE.items():
                if self.crosstalk_models.get(source) is None:
                    print(f"Crosstalk correction: {label} is configured for '{source}' but that "
                          f"backend failed to load -- {label} will get NO crosstalk correction "
                          f"until this is fixed.")
        # label ("Index"/"Middle"/"Ring"/"Thumb") -> last-known commanded
        # position (0=open..100=closed, same scale predict_shift() was
        # fitted on). Defaults to 0 (the reference/open position) for every
        # finger -- the same position the crosstalk model treats as
        # contributing exactly zero shift, so an unknown/not-yet-moved
        # finger is assumed to contribute no crosstalk rather than an
        # arbitrary guess. Set via set_finger_position(), called by
        # hand_control_gui.py's UnifiedMainWindow whenever it updates its
        # own per-finger "current position %" label (see that module's
        # _update_position_label) -- this dict is simply hand_control_gui.
        # py's existing position tracking, mirrored here, not a second,
        # independent source of truth.
        self._finger_positions: dict[str, float] = {label: 0.0 for label in CROSSTALK_FINGER_LABELS}
        # label -> (start_position, start_time, target_position, duration_s),
        # or None if that finger isn't mid-transit -- see set_finger_position()
        # / _current_finger_position() / CROSSTALK_POSITION_RAMP_S_PER_100PCT.
        self._finger_position_ramps: dict[str, tuple[float, float, float, float] | None] = {
            label: None for label in CROSSTALK_FINGER_LABELS
        }

    def start(self, port: str | None, baud: int) -> None:
        """Connects (and, if port is None, auto-discovers). NOTE: this can
        block for several seconds during discovery -- callers must invoke it
        from a background thread, not the GUI thread (see MainWindow.
        _on_connect_clicked)."""
        ctrl = self.ctrl
        ctrl.on_status = lambda m: self.status.emit(m)
        ctrl.on_error = lambda m: self.error.emit(m)
        ctrl.on_connected = lambda p: self.connected.emit(p)
        ctrl.on_disconnected = lambda: self.disconnected.emit()
        ctrl.on_frame = self._on_calibration_frame
        self.calibrating = True
        self._last_seq_shown = None
        ctrl.start(port=port, baud=baud)

    def _on_calibration_frame(self, frame: FrameData) -> None:
        # Runs on the serial reader thread. Each finger's raw samples go
        # into its own list, so each finger's baseline (computed in
        # finish_calibration) is that finger's own average only.
        with self._frame_lock:
            for i in range(NUM_FINGERS):
                self._baseline_samples[i].append(
                    frame_to_raw_values(frame, self.predictor.raw_cols, i))

    def _on_raw_frame(self, frame: FrameData) -> None:
        # Runs on the serial reader thread -- must stay this cheap. Same
        # split as live_predictor_test.py's on_raw_frame/process_frame:
        # only the newest frame is ever kept, and all real work happens in
        # poll(), driven by a GUI-thread QTimer instead of this callback.
        with self._frame_lock:
            self._latest_frame = frame

    def finish_calibration(self) -> list[int]:
        """Computes each finger's baseline from its own samples only, then
        switches the reader thread to the cheap stash-only callback. Returns
        each finger's sample count, for status/UI reporting."""
        with self._frame_lock:
            samples = [list(s) for s in self._baseline_samples]
        now = time.time()
        counts = []
        for i, fstate in enumerate(self.finger_states):
            s = samples[i]
            counts.append(len(s))
            if s:
                fstate.baseline = {
                    col: sum(x[col] for x in s) / len(s)
                    for col in self.predictor.raw_cols
                }
            else:
                fstate.baseline = {col: 0.0 for col in self.predictor.raw_cols}
            # Anchor the periodic-recalibration clock (RECALIBRATION_INTERVAL_S)
            # to this moment -- when this finger's baseline actually became
            # valid -- not to whenever the FingerState object happened to be
            # constructed. Also clear any leftover no-contact-window state
            # from a previous connection, so a reconnect starts fresh rather
            # than carrying over stale samples/timestamps.
            fstate.last_recal_time = now
            fstate.last_successful_recal_time = now
            fstate.no_contact_since = None
            fstate.recal_samples.clear()
            # "low_pass" BASELINE_TRACKING_MODE state (harmless no-ops if
            # "stable_window" is selected instead -- these fields are simply
            # never read by that mode) -- anchor the report snapshot to the
            # just-computed startup baseline, and reset the update counter,
            # so a reconnect starts fresh rather than carrying over a stale
            # count/snapshot from a previous connection.
            fstate.baseline_update_count = 0
            fstate.baseline_at_last_report = dict(fstate.baseline)
        self.ctrl.on_frame = self._on_raw_frame
        self.calibrating = False
        return counts

    def poll(self) -> list[FingerResult] | None:
        """Returns per-finger results for the newest frame, or None if no
        new frame has arrived since the last call. Always jumps straight to
        the latest frame -- older frames are dropped, never queued/replayed,
        so a temporarily slow GUI repaint can't accumulate a backlog against
        the serial port (same guarantee as live_predictor_test.py's display
        loop)."""
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None or frame.seq == self._last_seq_shown:
            return None
        self._last_seq_shown = frame.seq
        now = time.time()
        # Re-sampled fresh every tick from whatever ramp (if any) is in
        # progress for each finger -- see _current_finger_position() /
        # CROSSTALK_POSITION_RAMP_S_PER_100PCT. During a servo transit this
        # tracks an estimate of where the finger actually is, not just its
        # eventual destination (see that constant's comment for why this
        # matters for the crosstalk correction specifically). No longer
        # quantized before use -- see the (reverted) CROSSTALK_POSITION_
        # DEADBAND_PCT comment above for why: quantizing measurably biased
        # the crosstalk correction on Index's dominant source pair.
        current_positions = {
            label: self._current_finger_position(label, now)
            for label in CROSSTALK_FINGER_LABELS
        }
        # Computed AFTER current_positions above, not before -- _current_
        # finger_position() clears a ramp entry to None as a side effect the
        # instant it naturally completes (elapsed >= duration), so checking
        # here (not before that call) correctly reads "did THIS finger's own
        # transit just finish, if it was going to, this very tick" rather
        # than a stale one-tick-late value. See CONTACT_SUPPRESS_DURING_
        # OWN_TRANSIT for what this feeds.
        #
        # Capped at CONTACT_TRANSIT_SUPPRESS_S (see that constant's own
        # comment for the full rationale/trade-off) -- min(), not a
        # replacement value, so a SHORT move (its own ramp duration already
        # under the cap, e.g. Grip-to-Target's small per-round steps) is
        # completely unaffected; only a LARGE move's transit-true window
        # gets capped short of its full commanded-distance duration. Feeds
        # not just CONTACT_SUPPRESS_DURING_OWN_TRANSIT below but also this
        # finger's own baseline-update eligibility and (via any_finger_in_
        # transit) the cross-finger baseline contamination gate -- the SAME
        # accepted trade-off (a finger past the cap is treated as settled
        # even if a large unobstructed move is still genuinely in progress)
        # applies uniformly to all three rather than tracking divergent
        # per-purpose transit signals.
        in_own_transit = {}
        for label in CROSSTALK_FINGER_LABELS:
            ramp = self._finger_position_ramps.get(label)
            if ramp is None:
                in_own_transit[label] = False
            else:
                _, ramp_start_time, _, ramp_duration_s = ramp
                capped_duration_s = min(ramp_duration_s, CONTACT_TRANSIT_SUPPRESS_S)
                in_own_transit[label] = (now - ramp_start_time) < capped_duration_s
        # Whether ANY finger (not just this one) is mid-transit right now --
        # feeds the baseline-tracking contamination gate (see BASELINE_
        # TRACKING_MODE). Computed once per poll() tick, shared across all 4
        # _process_one_finger() calls below (a finger sitting perfectly
        # still still needs to know if a NEIGHBOUR is moving).
        any_finger_in_transit = any(in_own_transit.values())

        # 2026-07-20: settled_positions -- for CROSSTALK CORRECTION
        # specifically, a finger currently in transit contributes its LAST
        # KNOWN SETTLED position (wherever it was BEFORE this move started
        # -- ramp[0]) rather than current_positions' live linear-
        # interpolation ESTIMATE of where it currently is mid-move. Every
        # crosstalk model (pairwise_additive/nearest_neighbor/
        # ridge_quadratic) was fit ONLY on static, fully-settled position
        # combinations (see model_v1/crosstalk/README.md's "Dataset"
        # section) -- none of them were ever validated against a rapidly-
        # changing, guessed intermediate position, so feeding one a moving
        # neighbour's live estimate risks an inaccurate predicted
        # correction that over/under-subtracts and can swamp genuine touch
        # signal on the finger being corrected. Most relevant for a
        # two-finger grip where BOTH fingers move at once, since each
        # one's correction then depends on the OTHER's live (uncertain)
        # position estimate. Deliberately uses the FULL, UNCAPPED ramp
        # status here (self._finger_position_ramps directly), not the
        # CONTACT_TRANSIT_SUPPRESS_S-capped in_own_transit above -- that
        # cap is specifically for how long CONTACT evaluation/baseline
        # eligibility stays paused, a different trade-off; a position
        # estimate can still be genuinely unreliable well past that cap
        # for a large move, so this uses "is a ramp still actually
        # running" directly, uncapped. Falls back to current_positions
        # once the ramp naturally completes (same fresh-this-tick check
        # current_positions itself already relies on).
        settled_positions = {}
        for label in CROSSTALK_FINGER_LABELS:
            ramp = self._finger_position_ramps.get(label)
            settled_positions[label] = ramp[0] if ramp is not None else current_positions[label]

        return [_process_one_finger(self.predictor, fstate, frame, now,
                                     crosstalk_models=self.crosstalk_models,
                                     finger_positions=current_positions,
                                     settled_finger_positions=settled_positions,
                                     in_own_transit=in_own_transit[CROSSTALK_FINGER_LABELS[fstate.index]],
                                     any_finger_in_transit=any_finger_in_transit)
                for fstate in self.finger_states]

    def set_finger_position(self, label: str, position_percent: float) -> None:
        """Records `label`'s newly commanded servo TARGET position (0=open..
        100=closed), used by the crosstalk correction in _process_one_finger
        to predict how much of every OTHER finger's raw_delta is geometric
        crosstalk from this finger, not contact.

        Does NOT jump the position estimate straight to the target -- starts
        a linear ramp (see _current_finger_position(), sampled fresh by
        poll() every tick) from wherever this finger is CURRENTLY estimated
        to be (which correctly starts a new ramp from mid-transit if this
        finger was already moving -- e.g. rapid consecutive clicks -- rather
        than from its last commanded target) to the new target, over a
        duration proportional to the distance travelled (see
        CROSSTALK_POSITION_RAMP_S_PER_100PCT for why: treating a move as an
        instantaneous step was found to be a real, measurable source of
        transient false-contact readings on Index specifically).

        This module never calls it itself (sensing_gui.py has no servo
        control and doesn't know finger positions on its own) -- it exists
        for hand_control_gui.py's UnifiedMainWindow to call whenever it
        already updates its own "current position %" label (see that
        module's _update_position_label), so this is a mirror of that
        existing tracking, not a second independent source of truth. Safe
        to call even if CROSSTALK_CORRECTION_ENABLED is False or no
        crosstalk model loaded -- the stored value is simply unused in that
        case (see the guard in _process_one_finger).
        """
        now = time.time()
        start_position = self._current_finger_position(label, now)
        distance = abs(position_percent - start_position)
        duration_s = CROSSTALK_POSITION_RAMP_S_PER_100PCT * distance / 100.0
        self._finger_positions[label] = position_percent
        if duration_s <= 0.0:
            self._finger_position_ramps[label] = None
        else:
            self._finger_position_ramps[label] = (start_position, now, position_percent, duration_s)

    def _current_finger_position(self, label: str, now: float) -> float:
        """Returns `label`'s ESTIMATED current position (0=open..100=closed):
        linearly interpolated between the ramp's start/target if a transit
        is still in progress (see set_finger_position()), the settled target
        once the ramp's duration has elapsed (and clears the now-finished
        ramp so future calls take the cheap no-ramp path), or the last
        commanded position directly if no ramp is in progress at all.
        """
        ramp = self._finger_position_ramps.get(label)
        if ramp is None:
            return self._finger_positions[label]
        start_position, start_time, target_position, duration_s = ramp
        elapsed = now - start_time
        if elapsed >= duration_s:
            self._finger_position_ramps[label] = None
            return target_position
        frac = elapsed / duration_s
        return start_position + (target_position - start_position) * frac

    def stop(self) -> None:
        self.ctrl.stop()

    def request_recalibration(self, finger_indices: "list[int] | None" = None) -> None:
        """Makes the given fingers (default: all) immediately eligible for a
        periodic baseline recalibration (see RECALIBRATION_INTERVAL_S) on
        their next stable no-contact window, instead of waiting out however
        much of the normal interval remains. Does NOT force an immediate
        commit -- RECALIBRATION_MIN_STABLE_NO_CONTACT_S and
        RECALIBRATION_MIN_SAMPLES in _process_one_finger still gate whether
        and when a commit actually happens, so this can never recalibrate
        mid-contact or off too few samples; it only clears the "not enough
        time has passed since last recal" gate early.

        For a caller (e.g. hand_control_gui.py) that knows a finger's
        resting baseline may have just been physically disturbed -- e.g. by
        a servo command -- rather than only by ordinary drift. This module
        never calls it itself, so sensing_gui.py's own standalone
        behaviour is unaffected.
        """
        indices = finger_indices if finger_indices is not None else range(len(self.finger_states))
        for i in indices:
            self.finger_states[i].last_recal_time = 0.0

    def start_manual_recalibration(self, exclude_indices: "set[int] | None" = None,
                                    on_done=None) -> None:
        """On-demand baseline reset -- runs a FRESH BASELINE_DURATION_S
        capture using the exact same mechanism as the one-time connect-time
        calibration in start()/finish_calibration() (same _on_calibration_
        frame stash callback, same per-channel averaging), rather than
        waiting on request_recalibration()'s passive, periodic, one-finger-
        at-a-time commit path. Every finger's baseline is recomputed from
        this SAME capture window EXCEPT those in `exclude_indices` (e.g. a
        finger currently confirmed-in-contact, whose caller-supplied
        exclusion set is what keeps this from recalibrating against a
        mid-press reading) -- an excluded finger's baseline, last_recal_time,
        etc. are left completely untouched, exactly as if this call had
        never happened for that one finger.

        Temporarily switches ctrl.on_frame back to the calibration-capture
        callback, so poll() returns no new results (and the display simply
        holds its last frame) for the ~BASELINE_DURATION_S duration of the
        capture -- same visible effect as the connect-time calibration's own
        countdown, on purpose, since the hand needs to sit still for this to
        mean anything either way. Callers should disable whatever UI
        triggered this and show a status message for that window rather
        than expect live updates through it (see hand_control_gui.py's
        Reset Baseline button).

        on_done(counts: list[int], excluded: set[int]), if given, is called
        once the capture window closes and the new baseline has been
        committed -- via QTimer.singleShot, so it fires on this method's
        OWN caller's thread (must be the GUI thread, since that's the only
        thread with a running Qt event loop to deliver it on).
        """
        excluded = set(exclude_indices) if exclude_indices else set()
        with self._frame_lock:
            self._baseline_samples = [[] for _ in range(NUM_FINGERS)]
        self.ctrl.on_frame = self._on_calibration_frame

        def _finish():
            with self._frame_lock:
                samples = [list(s) for s in self._baseline_samples]
            now = time.time()
            counts = []
            for i, fstate in enumerate(self.finger_states):
                s = samples[i]
                counts.append(len(s))
                if i in excluded:
                    continue
                if s:
                    fstate.baseline = {
                        col: sum(x[col] for x in s) / len(s)
                        for col in self.predictor.raw_cols
                    }
                # Same clock/state reset finish_calibration() does for every
                # finger -- restarts this finger's periodic-recalibration
                # timer fresh from this moment, and clears any no-contact-
                # window bookkeeping left over from before the reset.
                fstate.last_recal_time = now
                fstate.last_successful_recal_time = now
                fstate.no_contact_since = None
                fstate.recal_samples.clear()
                fstate.baseline_update_count = 0
                fstate.baseline_at_last_report = dict(fstate.baseline)
            self.ctrl.on_frame = self._on_raw_frame
            if on_done is not None:
                on_done(counts, excluded)

        QtCore.QTimer.singleShot(int(BASELINE_DURATION_S * 1000), _finish)


# ============================================================================
# Widgets
# ============================================================================

class VerticalForceGauge(QtWidgets.QWidget):
    """A vertical bar that fills bottom-up proportionally to Force, on a
    fixed 0..FORCE_GAUGE_MAX_N scale (fixed, not auto-scaled, so the same
    fill height always means the same force across fingers and over time).
    """

    def __init__(self, max_n: float = FORCE_GAUGE_MAX_N, parent=None):
        super().__init__(parent)
        self._force = 0.0
        self._max_n = max_n
        self.setMinimumSize(46, 130)
        # Now grows with a stretch factor alongside ForceHistoryChart (see
        # FingerPanel's content_row) instead of staying pinned at its
        # minimum width -- capped here so it grows into a slightly wider,
        # still clearly bar-shaped gauge on a large window rather than
        # ballooning into a shapeless block.
        self.setMaximumWidth(90)

    def update_force(self, force: float) -> None:
        self._force = force
        self.update()

    def paintEvent(self, event) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QtGui.QColor(24, 24, 24))

        margin, label_h = 4, 16
        bar_x, bar_w = margin, w - 2 * margin
        bar_top, bar_bottom = label_h, h - margin
        bar_h = max(bar_bottom - bar_top, 1)
        p.fillRect(bar_x, bar_top, bar_w, bar_h, QtGui.QColor(46, 46, 46))

        frac = 0.0 if self._max_n <= 0 else max(0.0, min(1.0, self._force / self._max_n))
        fill_h = int(frac * bar_h)
        if fill_h > 0:
            p.fillRect(bar_x, bar_bottom - fill_h, bar_w, fill_h, QtGui.QColor(80, 200, 255))

        p.setPen(QtGui.QColor(230, 230, 230))
        f = p.font(); f.setPointSize(9); f.setBold(True); p.setFont(f)
        p.drawText(QtCore.QRectF(0, 0, w, label_h),
                    QtCore.Qt.AlignmentFlag.AlignCenter, f"{self._force:.2f}N")
        p.end()


class ForceHistoryChart(QtWidgets.QWidget):
    """Rolling line chart of Force over the trailing HISTORY_WINDOW_S
    seconds, fixed 0..FORCE_GAUGE_MAX_N scale so a slip's dip is visually
    comparable across fingers and doesn't get hidden by autoscaling.
    """

    def __init__(self, window_s: float = HISTORY_WINDOW_S,
                 max_n: float = FORCE_GAUGE_MAX_N, parent=None):
        super().__init__(parent)
        self._history: list[tuple[float, float]] = []
        self._now = time.time()
        self._window_s = window_s
        self._max_n = max_n
        self.setMinimumSize(160, 130)

    def update_data(self, history: list[tuple[float, float]], now: float) -> None:
        self._history = history
        self._now = now
        self.update()

    def paintEvent(self, event) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QtGui.QColor(20, 20, 20))

        # Gridlines at round Force values (thirds of full scale, e.g. 5N/
        # 10N/15N for the default 15N gauge) with an "N" label on each --
        # a bare gridline with no value on it is what made a large/maximized
        # chart look sparse/empty rather than like a real chart; the label
        # gives it content that scales with the widget instead of just
        # stretching three faint lines further apart.
        label_font = p.font(); label_font.setPointSize(8)
        p.setFont(label_font)
        if self._max_n > 0:
            for i in (1, 2, 3):
                value = self._max_n * i / 3
                y = int(h * (1 - i / 3))
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 20), 1))
                p.drawLine(0, y, w, y)
                p.setPen(QtGui.QColor(150, 150, 150))
                p.drawText(QtCore.QRectF(2, y - 12, 40, 12),
                            QtCore.Qt.AlignmentFlag.AlignLeft, f"{value:.0f}N")

        t0 = self._now - self._window_s
        pts = [(t, f) for t, f in self._history if t >= t0]
        if len(pts) >= 2:
            def to_xy(t, f):
                x = (t - t0) / self._window_s * w
                y = h - (max(0.0, min(f, self._max_n)) / self._max_n) * h if self._max_n > 0 else h
                return x, y

            path = QtGui.QPainterPath()
            x0, y0 = to_xy(*pts[0])
            path.moveTo(x0, y0)
            for t, f in pts[1:]:
                path.lineTo(*to_xy(t, f))
            p.setPen(QtGui.QPen(QtGui.QColor(80, 200, 255), 2.0))
            p.drawPath(path)
        else:
            p.setPen(QtGui.QColor(110, 110, 110))
            f = p.font(); f.setPointSize(8); p.setFont(f)
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "collecting…")

        # X-axis time context (rolling window endpoints) -- same reasoning
        # as the Force value labels above: unlabelled axes are what read as
        # "sparse" once the chart is large.
        p.setPen(QtGui.QColor(110, 110, 110))
        p.drawText(QtCore.QRectF(2, h - 13, 50, 12),
                    QtCore.Qt.AlignmentFlag.AlignLeft, f"-{self._window_s:.0f}s")
        p.drawText(QtCore.QRectF(w - 42, h - 13, 40, 12),
                    QtCore.Qt.AlignmentFlag.AlignRight, "now")
        p.end()


_STATE_STYLE = {
    "idle": ("#3a3a3a", "#777777", "#bbbbbb"),   # background, border, text
    "contact": ("#123a12", "#2fbf2f", "#7dff7d"),
    "slip": ("#3a1212", "#e03030", "#ff8080"),
}
_STATE_TEXT = {
    "idle": "●  NO CONTACT",
    "contact": "●  CONTACT",
    "slip": "●  SLIP DETECTED",
}


class FingerPanel(QtWidgets.QFrame):
    """One finger's full display block: title, contact-state indicator
    (also drives the panel border/background colour), contact-duration
    timer, vertical force gauge, rolling force-history chart, and plain
    Theta/Phi numeric readouts.
    """

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self._index = index
        self._label = f"Finger {index + 1}"
        self._last_state: str | None = None
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(4)

        title_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(self._label)
        title.setStyleSheet("font-weight:bold;font-size:13px;background:transparent;")
        title_row.addWidget(title)
        title_row.addStretch()
        self._state_lbl = QtWidgets.QLabel(_STATE_TEXT["idle"])
        self._state_lbl.setStyleSheet("font-weight:bold;font-size:11px;background:transparent;")
        title_row.addWidget(self._state_lbl)
        layout.addLayout(title_row)

        subtitle = QtWidgets.QLabel(f"(mux Group {index + 1})")
        subtitle.setStyleSheet("color:#888;font-size:9px;background:transparent;")
        layout.addWidget(subtitle)

        self._duration_lbl = QtWidgets.QLabel("Contact: —")
        self._duration_lbl.setStyleSheet("color:#ccc;font-size:10px;background:transparent;")
        layout.addWidget(self._duration_lbl)

        content_row = QtWidgets.QHBoxLayout()
        content_row.setSpacing(6)
        self._gauge = VerticalForceGauge()
        # Both get a stretch factor (not just the chart) so the gauge grows
        # alongside the chart on a large/maximized window instead of staying
        # pinned at its minimum width while the chart balloons next to it --
        # that mismatch (a thin sliver of a bar next to a huge chart) was
        # the "gauge and chart look mismatched" complaint. 1:5 keeps the
        # gauge visibly growing but still clearly the narrower of the two;
        # VerticalForceGauge's own setMaximumWidth caps how far it goes so
        # it stays bar-shaped rather than ballooning into a blob.
        content_row.addWidget(self._gauge, stretch=1)
        self._chart = ForceHistoryChart()
        content_row.addWidget(self._chart, stretch=5)
        layout.addLayout(content_row, stretch=1)

        self._angles_lbl = QtWidgets.QLabel("Theta: —      Phi: —")
        self._angles_lbl.setStyleSheet("color:#ccc;font-size:10px;background:transparent;")
        layout.addWidget(self._angles_lbl)

        self._glitch_lbl = QtWidgets.QLabel("")
        self._glitch_lbl.setStyleSheet("color:#e0a030;font-size:9px;background:transparent;")
        layout.addWidget(self._glitch_lbl)

        self._apply_state("idle")

    def set_calibrating(self) -> None:
        self._state_lbl.setText("●  CALIBRATING…")
        self._duration_lbl.setText("Contact: —")
        self._angles_lbl.setText("Theta: —      Phi: —")
        self._apply_state("idle")

    def apply_result(self, fr: FingerResult) -> None:
        self._gauge.update_force(fr.force)
        self._chart.update_data(fr.history, time.time())

        if fr.slip_active:
            state = "slip"
        elif fr.contact:
            state = "contact"
        else:
            state = "idle"
        self._apply_state(state)
        self._state_lbl.setText(_STATE_TEXT[state])

        if fr.contact_duration_s is not None:
            self._duration_lbl.setText(f"Contact: {fr.contact_duration_s:.1f}s")
        else:
            self._duration_lbl.setText("Contact: —")

        if fr.contact:
            # Phi display gate (see PHI_DISPLAY_MIN_THETA_DEG /
            # PHI_DISPLAY_MIN_CONF): azimuth is only shown when it is
            # physically meaningful (press angled enough) AND the classifier
            # is confident -- otherwise an honest "--" instead of a
            # near-random value.
            if (abs(fr.theta) >= PHI_DISPLAY_MIN_THETA_DEG
                    and fr.phi_conf >= PHI_DISPLAY_MIN_CONF):
                phi_text = f"{fr.phi:+d}°"
            else:
                phi_text = "--"
            self._angles_lbl.setText(f"Theta: {fr.theta:+.1f}°      Phi: {phi_text}")
        else:
            self._angles_lbl.setText("Theta: —      Phi: —")

        self._glitch_lbl.setText("⚠ sensor glitch (frozen)" if fr.glitched else "")

    def _apply_state(self, state: str) -> None:
        if state == self._last_state:
            return
        self._last_state = state
        bg, border, _text = _STATE_STYLE[state]
        self.setStyleSheet(
            f"FingerPanel {{ background-color:{bg}; border:2px solid {border}; border-radius:6px; }}"
        )
        _bg, _border, text_color = _STATE_STYLE[state]
        self._state_lbl.setStyleSheet(
            f"font-weight:bold;font-size:11px;background:transparent;color:{text_color};")


# ============================================================================
# Main window
# ============================================================================

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, port: str | None = None, baud: int = 2_000_000):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(MIN_WINDOW_W, MIN_WINDOW_H)

        self._baud = baud
        self._engine = HandSensingEngine(self)
        self._engine.status.connect(self._on_status)
        self._engine.error.connect(self._on_error)
        self._engine.connected.connect(self._on_connected)
        self._engine.disconnected.connect(self._on_disconnected)

        self._calibration_deadline: float | None = None
        self._calib_timer = QtCore.QTimer(self)
        self._calib_timer.timeout.connect(self._update_calibration_countdown)

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._on_tick)

        # Throttle for the console [DIAG] block -- see DIAG_PRINT_INTERVAL_S.
        self._last_diag_print = 0.0

        self._setup_ui()

        if port:
            self._port_edit.setText(port)
            QtCore.QTimer.singleShot(0, self._on_connect_clicked)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        conn_row = QtWidgets.QHBoxLayout()
        conn_row.addWidget(QtWidgets.QLabel("Port:"))
        self._port_edit = QtWidgets.QLineEdit()
        self._port_edit.setPlaceholderText("blank = auto-discover")
        self._port_edit.setMaximumWidth(140)
        conn_row.addWidget(self._port_edit)

        self._connect_btn = QtWidgets.QPushButton("🔌  Connect")
        self._connect_btn.clicked.connect(self._on_connect_clicked)
        conn_row.addWidget(self._connect_btn)

        self._disc_btn = QtWidgets.QPushButton("⏻  Disconnect")
        self._disc_btn.setEnabled(False)
        self._disc_btn.clicked.connect(self._on_disconnect_clicked)
        conn_row.addWidget(self._disc_btn)

        conn_row.addSpacing(16)
        self._status_lbl = QtWidgets.QLabel("Ready — click Connect to begin.")
        self._status_lbl.setStyleSheet("color:#aaa;font-size:10px;")
        conn_row.addWidget(self._status_lbl, stretch=1)
        root.addLayout(conn_row)

        self._total_lbl = QtWidgets.QLabel("Total Grip Force: —")
        self._total_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._total_lbl.setStyleSheet(
            "font-size:22px;font-weight:bold;color:#5cf;padding:6px;"
            "background:#1c1c1c;border-radius:6px;")
        root.addWidget(self._total_lbl)

        grid_w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_w)
        grid.setSpacing(8)
        grid.setContentsMargins(0, 0, 0, 0)
        self._panels: list[FingerPanel] = []
        for i in range(NUM_FINGERS):
            panel = FingerPanel(i)
            self._panels.append(panel)
            grid.addWidget(panel, i // 2, i % 2)
        root.addWidget(grid_w, stretch=1)

        log_box = QtWidgets.QGroupBox("Event Log (slips + baseline recalibrations)")
        log_layout = QtWidgets.QVBoxLayout(log_box)
        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(300)
        self._log.setMaximumHeight(120)
        self._log.setStyleSheet(
            "QPlainTextEdit{background:#1a1a1a;color:#cfd8e3;"
            "font-family:Consolas,monospace;font-size:10px;border:none;}")
        log_layout.addWidget(self._log)
        root.addWidget(log_box)

        # Live per-finger diagnostics -- see _update_diagnostics(). A plain
        # in-place readout (not an event log like Slip Events above), since
        # this shows CURRENT rolling state, not discrete past events.
        diag_box = QtWidgets.QGroupBox(
            "Diagnostics (raw Contact toggle rate / hysteresis streaks / glitches)")
        diag_layout = QtWidgets.QVBoxLayout(diag_box)
        self._diag_lbl = QtWidgets.QLabel("—")
        self._diag_lbl.setStyleSheet(
            "color:#ccc;font-size:10px;font-family:Consolas,monospace;"
            "padding:2px 4px;")
        self._diag_lbl.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        diag_layout.addWidget(self._diag_lbl)
        root.addWidget(diag_box)

        footer = QtWidgets.QLabel(f"Model: {MODEL_PATH.name}")
        footer.setStyleSheet("color:#666;font-size:9px;")
        root.addWidget(footer)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _on_connect_clicked(self) -> None:
        self._connect_btn.setEnabled(False)
        self._port_edit.setEnabled(False)
        self._status("Connecting…")
        port = self._port_edit.text().strip() or None
        # EfleshController.start() blocks synchronously while auto-
        # discovering (port=None) -- run it off the GUI thread so the
        # window stays responsive during the scan, same precaution
        # unified_gui.py takes around its ControllerAdapter.connect_all.
        threading.Thread(
            target=lambda: self._engine.start(port=port, baud=self._baud),
            daemon=True).start()

    def _on_connected(self, port: str) -> None:
        self._disc_btn.setEnabled(True)
        for panel in self._panels:
            panel.set_calibrating()
        self._calibration_deadline = time.time() + BASELINE_DURATION_S
        self._calib_timer.start(100)

    def _update_calibration_countdown(self) -> None:
        remaining = self._calibration_deadline - time.time()
        if remaining <= 0:
            self._calib_timer.stop()
            self._finish_calibration()
            return
        self._status(f"Calibrating baseline — keep the hand at rest ({remaining:.1f}s remaining)…")

    def _finish_calibration(self) -> None:
        counts = self._engine.finish_calibration()
        parts = []
        for i, count in enumerate(counts):
            if count == 0:
                parts.append(f"F{i + 1}: FAILED")
            elif count < MIN_BASELINE_SAMPLES:
                parts.append(f"F{i + 1}: weak ({count})")
            else:
                parts.append(f"F{i + 1}: ok ({count})")
        self._status("Baseline calibrated — " + ", ".join(parts))
        self._poll_timer.start(int(DISPLAY_INTERVAL_S * 1000))

    def _on_disconnect_clicked(self) -> None:
        self._poll_timer.stop()
        self._calib_timer.stop()
        self._engine.stop()
        self._reset_connection_ui()
        self._status("Disconnected.")

    def _on_disconnected(self) -> None:
        self._poll_timer.stop()
        self._calib_timer.stop()
        self._reset_connection_ui()

    def _on_error(self, msg: str) -> None:
        self._poll_timer.stop()
        self._calib_timer.stop()
        self._status(f"⚠ {msg}")
        self._reset_connection_ui()

    def _reset_connection_ui(self) -> None:
        self._connect_btn.setEnabled(True)
        self._port_edit.setEnabled(True)
        self._disc_btn.setEnabled(False)
        self._total_lbl.setText("Total Grip Force: —")
        self._diag_lbl.setText("—")
        for panel in self._panels:
            panel.set_calibrating()

    def _on_status(self, msg: str) -> None:
        self._status(msg)

    def _status(self, msg: str) -> None:
        self._status_lbl.setText(msg)

    # ------------------------------------------------------------------
    # Live display loop
    # ------------------------------------------------------------------

    def _on_tick(self) -> None:
        results = self._engine.poll()
        if results is None:
            return
        total = 0.0
        for fr in results:
            self._panels[fr.index].apply_result(fr)
            total += fr.force
            if fr.slip_message:
                self._append_log(fr.slip_message)
            if fr.recal_message:
                self._append_log(fr.recal_message)
        self._total_lbl.setText(
            f"Total Grip Force: {total:.2f} N   (avg {total / NUM_FINGERS:.2f} N/finger)")
        self._update_diagnostics(results)

    def _update_diagnostics(self, results: list[FingerResult]) -> None:
        """Updates the on-screen Diagnostics label every tick (cheap, plain
        text) and prints the same per-finger breakdown to the console every
        DIAG_PRINT_INTERVAL_S, so a session can be captured to a log file for
        offline comparison across fingers -- see the module-level constants
        for why these numbers matter: raw_contact_toggle_rate_hz answers
        "is the raw classifier oscillating at the confirm/release boundary"
        (hypothesis: too-tight CONTACT_CONFIRM_FRAMES/CONTACT_RELEASE_FRAMES,
        or a noisy/systemic signal source); the streak counters show exactly
        where each finger currently sits in the hysteresis state machine;
        glitch_frame_count rules the SANE_DELTA_MAX guard in or out as the
        cause of any given flicker.
        """
        lines = [
            f"{fr.label}: raw_toggle={fr.contact_toggle_rate_hz:5.2f}/s  "
            f"raw_Contact={fr.raw_contact if fr.raw_contact is not None else '?'}  "
            f"streak(contact/no-contact)={fr.consecutive_contact_frames}/"
            f"{fr.consecutive_no_contact_frames}  "
            f"confirmed={fr.contact}  glitches={fr.glitch_frame_count}"
            for fr in results
        ]
        self._diag_lbl.setText("\n".join(lines))

        now = time.time()
        if now - self._last_diag_print >= DIAG_PRINT_INTERVAL_S:
            self._last_diag_print = now
            print(f"[DIAG] thresholds: CONTACT_CONFIRM_FRAMES={CONTACT_CONFIRM_FRAMES}  "
                  f"CONTACT_RELEASE_FRAMES={CONTACT_RELEASE_FRAMES}  "
                  f"(toggle rate measured over trailing {TOGGLE_RATE_WINDOW_S:.1f}s)")
            for line in lines:
                print(f"[DIAG] {line}")

    def _append_log(self, msg: str) -> None:
        ts = _dt.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{ts}] {msg}"
        self._log.appendPlainText(line)
        # Also printed to console (not just the on-screen widget) so a long
        # session can be captured to a log file and the recalibration lines
        # compared against the [DIAG] toggle-rate trend afterwards, to
        # confirm recalibration is actually reducing drift-related flicker
        # over time rather than just trusting it silently.
        print(f"[EVENT] {line}", flush=True)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._poll_timer.stop()
        self._calib_timer.stop()
        self._engine.stop()
        event.accept()


# ============================================================================
# Entry point
# ============================================================================

def _dark_palette() -> QtGui.QPalette:
    p = QtGui.QPalette()
    for role, color in {
        QtGui.QPalette.ColorRole.Window: QtGui.QColor(40, 40, 40),
        QtGui.QPalette.ColorRole.WindowText: QtGui.QColor(220, 220, 220),
        QtGui.QPalette.ColorRole.Base: QtGui.QColor(30, 30, 30),
        QtGui.QPalette.ColorRole.AlternateBase: QtGui.QColor(50, 50, 50),
        QtGui.QPalette.ColorRole.Text: QtGui.QColor(220, 220, 220),
        QtGui.QPalette.ColorRole.Button: QtGui.QColor(55, 55, 55),
        QtGui.QPalette.ColorRole.ButtonText: QtGui.QColor(220, 220, 220),
        QtGui.QPalette.ColorRole.Highlight: QtGui.QColor(100, 150, 255),
        QtGui.QPalette.ColorRole.HighlightedText: QtGui.QColor(30, 30, 30),
        QtGui.QPalette.ColorRole.ToolTipBase: QtGui.QColor(220, 220, 220),
        QtGui.QPalette.ColorRole.ToolTipText: QtGui.QColor(30, 30, 30),
    }.items():
        p.setColor(role, color)
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="eFlesh 4-finger hand sensing monitor")
    parser.add_argument("--port", help="Teensy COM port (auto-discovers if omitted)")
    parser.add_argument("--baud", type=int, default=2_000_000, help="Serial baud rate")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())

    window = MainWindow(port=args.port, baud=args.baud)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
