# Thumb Contact-confirm threshold: 0.30 -> 0.10 (per-finger override)

> **⚠ STATUS WARNING -- READ BEFORE CITING THIS FIX ⚠**
> **Live validation was planned and tooling was built
> (`live_validate_thumb_threshold.py`, `analyze_thumb_threshold_live_
> validation.py`), but has not yet been run, due to time/hardware
> constraints. This fix should be described as OFFLINE-HOLDOUT-VALIDATED
> ONLY, NOT LIVE-CONFIRMED, until this test is completed.** See "Live
> validation (hardware)" below for exactly what is and isn't done.

`sensing_gui.py`'s global Contact-confirm Schmitt trigger (`CONTACT_P_CONFIRM`
/ `CONTACT_P_RELEASE` = 0.30/0.15) applies to all four fingers by default.
On 2026-09-07, after retraining the Contact classifier on the new
load-cell grasping dataset (`grasping_dashboard/retrain_contact_with_
load_cell_data.py`), a per-finger override was added for Thumb specifically:

```python
CONTACT_P_CONFIRM_PER_FINGER: dict[str, float] = {"Thumb": 0.10}
CONTACT_P_RELEASE_PER_FINGER: dict[str, float] = {"Thumb": 0.05}
```

(kept at the same 2:1 confirm:release ratio as the global default).

## Motivation

The retrained classifier's own holdout accuracy report (embedded in
`active_model.pkl`'s `contact_classifier_fix_note`) showed a much smaller
improvement for Thumb than the other three fingers: Index 11%->99%, Middle
28%->93%, Ring 51%->98%, **Thumb 34%->53%** -- using sklearn's raw
`.predict()` (an implicit P>=0.5 cutoff). This is explicitly flagged in
that same note as not representative of live behaviour, since the live
system never uses a naive 0.5 cutoff -- it uses the tuned Schmitt-trigger
thresholds instead.

## Offline holdout evidence (2026-09-07)

Re-examining Thumb's held-out prediction *probabilities* (not the raw 0.5
cutoff) against the same ground truth:

| P_CONFIRM threshold | Precision | Recall |
|---|---|---|
| 0.30 (old global default) | 1.000 | 0.815 |
| 0.10 (new Thumb override) | 1.000 | 0.974 |
| down to 0.02 (not adopted) | 1.000 | (not adopted -- see below) |

- Dataset: 783 held-out ticks from 6 Thumb reps never used in training
  (rep-grouped holdout, same methodology as `retrain_contact_with_load_
  cell_data.py`'s own evaluation).
- Precision stayed a **perfect 1.000 down to P=0.02** -- there is no
  false-positive cost anywhere in this sweep. The missed touches at 0.30
  were real, low-force contacts (0.6-0.8N) the classifier already assigned
  a real but moderate probability to, not ambiguous/noisy cases.
- 0.10 was chosen (not the more aggressive values tested down to 0.02) to
  keep the same 2:1 confirm:release ratio as the global default and leave
  margin, since this is a single holdout, not exhaustive testing across
  every object/condition.
- No dedicated sweep script/report was produced for this analysis (unlike
  most other threshold decisions in this project) -- the conclusion is
  recorded only as the code comment above `CONTACT_P_CONFIRM_PER_FINGER`
  in `sensing_gui.py`. This report is the first standalone writeup of it.

## Deployment status

**Active in the current live code** as of `sensing_gui.py`'s 2026-09-07
20:32 save (confirmed by direct file-timestamp inspection) -- not merely a
candidate on disk. Both `hand_control_gui.py` and the grasping dashboard
import this module and pick up the override automatically; no separate
deploy step exists or is needed. It postdates `active_model.pkl`'s own
final 2026-09-07 20:11 save, so it sits on top of the currently deployed
classifier, not an older one.

## Live validation (hardware)

**Status: NOT YET RUN, as of this report (2026-09-08).** Live validation
was planned and the tooling for it was built (`live_validate_thumb_
threshold.py`, `analyze_thumb_threshold_live_validation.py`), but the
actual hardware session has not been run, due to time/hardware
constraints -- this requires a human physically presenting a load cell to
the robot's Thumb rep by rep, which is outside what could be completed in
this pass. **Until this test is completed and this section is updated
with real TP/FN/FP/TN counts, this fix must be described as
offline-holdout-validated only, not live-confirmed.**

Confirmed by direct timestamp comparison that the one live-validation session run that day
(`grasping_dashboard_data/live_validation_candidate/session_20260907_
194056`, 19:40-19:46) predates this threshold change (saved 20:32) and
used `live_validate_candidate_model.py`, which only swaps in the retrained
classifier *model* -- it does not exercise this threshold override, which
was still at the old global 0.30/0.15 at the time that session ran. **No
live hardware test of this specific fix has been run.**

Tooling built and verified (see both scripts' own docstrings for detail):

- `live_validate_thumb_threshold.py` -- launches the dashboard against the
  currently deployed model, asserts the Thumb threshold override is
  actually active before opening (refuses to run otherwise -- verified
  both the pass and refusal paths headlessly), and redirects session data
  to `grasping_dashboard_data/live_validation_thumb_threshold/`.
- `analyze_thumb_threshold_live_validation.py` -- computes Thumb's live
  recall (genuine-touch ticks, load cell facing Thumb, `Force >= 0.3N`
  ground truth -- same definition as the offline evaluation) and
  false-positive rate (control reps + no-touch ticks), scored against both
  the confirmed (`sensing.Thumb.contact`) and raw (`sensing.Thumb.
  raw_contact`) signals, for direct comparison to the offline numbers
  above. Smoke-tested against existing (non-Thumb) session data to confirm
  correct parsing; not yet run against real Thumb data.

**Planned protocol** (mirrors the offline test's scope rather than a full
re-collection): ~3 reps each at light/medium/firm in a Thumb-involving
scenario with "Load cell facing" set to Thumb (recall side), plus ~4-5
control reps (false-positive side).

**Reference to beat:** ~~offline holdout predicted recall 0.974, zero false
positives, at P_CONFIRM=0.10~~ -- **UPDATE (2026-09-08): this 0.974 figure
could not be reproduced and should not be cited.** A full investigation
(`thumb_recall_discrepancy_investigation.md`) found the reproducible,
checked-in-script number is **recall 0.713 at P_CONFIRM=0.10** (vs. 0.569
at the old global 0.30) -- use THIS as the reference to beat instead. The
same investigation also found Thumb's offline holdout has zero true
negatives, so it cannot test false-positive behaviour at all -- live
validation is the only way to assess Thumb's false-positive rate, making
this pending hardware test more important for Thumb than originally
understood.

*(This section to be filled in with live TP/FN/FP/TN counts once the
session above is run and analyzed.)*
