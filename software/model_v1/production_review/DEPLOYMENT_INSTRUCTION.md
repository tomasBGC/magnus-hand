# Copy-paste Claude Code instruction — deploy the production-review improvements

Everything below the line is a single, self-contained instruction to paste
into a fresh Claude Code session. It deploys only the changes validated in
`model_v1/production_review/` (see `00_REVIEW_SUMMARY.md` for evidence and
ranking), each isolated behind a named flag or a per-finger config entry so
any single piece can be reverted independently.

---

Deploy the validated improvements from `model_v1/production_review/` (see its
`00_REVIEW_SUMMARY.md` for the evidence behind each change; all candidate
artifacts referenced below already exist and were validated offline in that
review). Work through the steps in order. Do not modify any file not listed
here. After each step, run the verification listed for it before moving on.

**Step 1 — Back up and deploy the candidate model bundle.**
Copy `model_v1/active_model.pkl` to
`model_v1/production_review/active_model_BACKUP_pre_review.pkl` (do not skip
the backup). Then copy
`model_v1/production_review/active_model_review_candidate.pkl` over
`model_v1/active_model.pkl`. This bundle differs from the deployed one in
exactly two ways: the Phi head is retrained without the physically-degenerate
theta=0 rows (held-out theta=90 Phi accuracy 0.416 -> 0.644, theta>=30
Units-1+2 accuracy 0.837 -> 0.841, validated in `08_phi_head_report.md`), and
a new additive `contact_calibrator` key (isotonic, Units-1+2-fit, test ECE
0.053 -> 0.007) that nothing reads until Step 4. Force regressor, Contact
classifier, and scaler are byte-identical to the current deployment, so the
self-closure contact fix is untouched.
Verify: `uv run python model_v1/production_review/01_baseline_eval.py` — the
report's overall Phi accuracy should now read ~0.66 (was 0.5707), Force MAE
and Contact accuracy unchanged (1.3288 N / 0.7771).

**Step 2 — Contact-probability Schmitt trigger in `model_v1/sensing_gui.py`.**
Add three module constants next to `CONTACT_CONFIRM_FRAMES` (with a comment
citing `production_review/04_calibration_report.md`):
`CONTACT_PROB_SCHMITT_ENABLED = True`, `CONTACT_P_CONFIRM = 0.6`,
`CONTACT_P_RELEASE = 0.3`.
In `_process_one_finger`, replace the current single-signal hysteresis input

```python
contact_signal = result["Contact"] == 1 and result["Force"] > MIN_CONTACT_FORCE_N
```

with a two-threshold pair on the classifier's probability
(`p1 = result["Contact_probs"][1]`): when the flag is True,
`confirm_signal = p1 >= CONTACT_P_CONFIRM and result["Force"] > MIN_CONTACT_FORCE_N`
and `release_signal = p1 <= CONTACT_P_RELEASE`; when False, reproduce the old
behaviour exactly (`confirm_signal` as before, `release_signal = not
confirm_signal`). Update the streak logic so `confirm_signal` increments
`consecutive_contact_frames` (resetting the other counter), `release_signal`
increments `consecutive_no_contact_frames` (resetting the other counter), and
a frame in the dead zone between the two thresholds leaves BOTH counters
unchanged (that hold is the point of a Schmitt trigger). The raw_contact /
toggle-rate diagnostics must stay based on `result["Contact"]`, ungated, as
they are now. Evidence: on held-out Units-1+2 test rows this cuts idle frames
feeding the confirm streak by 28% (0.0633 -> 0.0454) AND mid-press release
pressure by 29% (0.1277 -> 0.0908); light-touch (<1 N) confirm sensitivity
cost is 2–6 pp (see the addendum table in `04_calibration_report.md`).

**Step 3 — Transient spike guard in `model_v1/sensing_gui.py`.**
Add constants `SPIKE_JUMP_GUARD_ENABLED = True` and
`SPIKE_JUMP_MAX_PER_TICK = 300.0` (comment: genuine per-tick channel jumps on
Units 1+2 never exceeded 19.4 raw units at the 20 Hz live cadence including
during presses, while a single-channel 300-unit spike — far below
SANE_DELTA_MAX=2000 — drives the model to ~22% false confirm-feed frames at
~7.6 N phantom force; see `05_robustness_report.md` and the jump-distribution
measurement in `00_REVIEW_SUMMARY.md`). Add two fields to `FingerState`:
`last_accepted_delta: dict[str, float] | None = None` and
`spike_hold_active: bool = False`. In `_process_one_finger`, immediately
after the crosstalk correction produces the final `raw_delta`: if the guard
is enabled, `last_accepted_delta` is not None, the max per-channel
`|raw_delta - last_accepted_delta|` exceeds `SPIKE_JUMP_MAX_PER_TICK`, and
`spike_hold_active` is False, then set `spike_hold_active = True` and treat
the frame exactly like the existing SANE_DELTA_MAX glitch path (reuse last
result, increment `glitch_frame_count`). Otherwise set
`spike_hold_active = False` and `last_accepted_delta = dict(raw_delta)` and
continue normally. The one-tick hold means a genuine fast press is delayed at
most 50 ms once; a single-frame glitch is suppressed entirely.

**Step 4 — Phi display gating in `model_v1/sensing_gui.py`.**
Add constants `PHI_DISPLAY_MIN_THETA_DEG = 25.0` and
`PHI_DISPLAY_MIN_CONF = 0.7` (comment: azimuth is physically undefined for a
near-normal press; with the new Phi head, gating display at max-prob >= 0.7
yields 98.8% displayed-Phi accuracy at ~71% coverage on held-out
Units-1+2 theta>=30 rows — `08_phi_head_report.md`). Add a `phi_conf: float`
field to `FingerResult` (set it from `max(result["Phi_probs"].values())` in
the normal path, and carry the last value in the glitched path like the
other reused fields). In `MainWindow`'s angle label update (currently
`self._angles_lbl.setText(f"Theta: {fr.theta:+.1f}°      Phi: {fr.phi:+d}°")`),
show `Phi: --` whenever there is no confirmed contact, or
`abs(fr.theta) < PHI_DISPLAY_MIN_THETA_DEG`, or
`fr.phi_conf < PHI_DISPLAY_MIN_CONF`.

**Step 5 — Register the ridge crosstalk backend and switch Middle + Index to
it (live-A/B-gated).**
Copy `model_v1/production_review/ridge_crosstalk_model.py` to
`model_v1/crosstalk_lookup/ridge_crosstalk_model.py` and
`model_v1/production_review/candidate_crosstalk_ridge_v1.json` to
`model_v1/crosstalk_lookup/crosstalk_ridge_v1.json` (update the module's
`DEFAULT_MODEL_PATH` to that new json name/location). In
`model_v1/sensing_gui.py`: import `load_model`/`predict_shift` from it (same
sys.path block the other crosstalk_lookup imports already use), add a
`RIDGE_CROSSTALK_MODEL_PATH` constant, load it in `HandSensingEngine.__init__`
into `self.crosstalk_models["ridge_quadratic"]` with the same
fail-to-None-and-print pattern the other two backends use, add
`"ridge_quadratic": ridge_predict_shift` to `CROSSTALK_CORRECTION_BACKENDS`,
and change `CROSSTALK_CORRECTION_SOURCE` to `Index: "ridge_quadratic"`,
`Middle: "ridge_quadratic"`, leaving Ring and Thumb unchanged. Add a comment
citing `07_crosstalk_ridge_report.md`: 5x5-fold CV over all 100 measured
4-finger combinations gives Middle 46.8 -> 41.7 MAE vs its current pairwise
backend and Index 18.2 -> 17.7 vs its current NN-smoothed backend, with the
same structural exact-zero-at-origin guarantee that made NN-smoothed safe;
and note explicitly that offline MAE mispredicted live behaviour once before
(plain NN), so this switch must be confirmed by the counterbalanced live A/B
below before being considered settled.

**Step 6 — Offline verification.**
Run `uv run python model_v1/production_review/ridge_crosstalk_model.py`
(from its new crosstalk_lookup location) — all 4 fingers must print
`origin=0 OK`. Then start `uv run model_v1/sensing_gui.py --help` (or import
it in a python -c) to confirm no import errors, and if hardware is attached,
run `uv run model_v1/live_predictor_test.py` briefly to confirm the model
loads and predicts.

**Step 7 — Live verification (hardware attached).**
Launch `uv run model_v1/hand_control_gui.py` and check, in order: (a) the
console prints all THREE crosstalk backends loaded, including
`ridge_quadratic`, with no "FAILED to load" lines; (b) at rest, all 4 fingers
show Force 0.0 and no CONTACT, and Phi shows `--`; (c) a firm press on each
fingertip confirms CONTACT within ~0.5 s, shows a plausible Force, and
releases cleanly within ~0.5 s of letting go — pay specific attention to
Index's force NOT reading weak (that was the symptom that forced the
deadband revert); (d) an angled press shows a Phi value, a straight-down
press shows `Phi: --`; (e) Close All / Open All produces no false CONTACT on
any finger. Then run the project's own counterbalanced A/B harness
(`model_v1/crosstalk_lookup/realistic_session_diagnostic.py`) comparing
`ridge_quadratic` vs each switched finger's previous backend, per the
methodology already used for the NN-smoothed decision; if Index or Middle
shows MORE confirmed false-contact episodes than its previous backend,
revert that finger's `CROSSTALK_CORRECTION_SOURCE` entry (one-line change)
and note the result in `production_review/00_REVIEW_SUMMARY.md`.

**Step 8 — Documentation.**
Update `model_v1/README.md`'s "What's live" section (the active model now
includes the no-theta0 Phi head + contact calibrator; note the backup path),
and append a short "deployed on <date>" note to
`model_v1/production_review/00_REVIEW_SUMMARY.md` recording which steps went
live and the live-A/B outcome for Step 5. Do NOT enable
`FEATURE_EMA`-style feature smoothing or swap any other model head — those
were reviewed and either rejected or left as documented options (see the
summary's "Rejected / optional" section).
