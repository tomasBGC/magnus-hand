# Per-finger Contact-confirm threshold sweep: Index, Middle, Ring (+ Thumb re-check)

> **⚠ STATUS WARNING -- READ BEFORE CITING THIS FIX ⚠**
> **Offline-holdout-validated only, NOT live-confirmed**, consistent with
> the Thumb threshold's own status (see `thumb_threshold_live_validation_
> report.md`). No live hardware test has been run against Middle's new
> override or against Index/Ring's "no change" conclusion.
>
> **A second, independent finding in this report also affects Thumb's
> already-deployed 0.10 threshold**: re-running the analysis with a
> proper, checked-in script hit the identical held-out reps Thumb's
> original (undocumented) sweep must have used, but could NOT reproduce
> its recall numbers (0.815/0.974). **Fully investigated in
> `thumb_recall_discrepancy_investigation.md` -- verdict: do not cite
> 0.815/0.974, cite 0.569/0.713 instead.** Thumb's override is LEFT AS-IS
> (still justified by the corrected numbers), not reverted by this report.

## Methodology

Same approach as Thumb's original per-finger threshold decision
(`sensing_gui.py`'s `CONTACT_P_CONFIRM_PER_FINGER`, 2026-09-07), but now
implemented as a standalone, reusable, checked-in script --
`grasping_dashboard/sweep_finger_thresholds.py` -- rather than one-off,
undocumented analysis (no script or report existed for the original Thumb
sweep; this is the first time this methodology has been captured in a
rerunnable form).

- **Ground truth / holdout**: `grasping_dashboard_data/training_protocol/
  load_cell_dataset.csv` (18,067 rows, the same 143-rep load-cell dataset
  used for the Contact classifier retrain), split via `retrain_contact_
  with_load_cell_data.py`'s own `_split_load_cell_dataset` -- an unmodified
  import, not a reimplementation, so this is guaranteed to be the exact
  same seeded (RNG seed 42), per-finger-stratified, rep-grouped 20%
  holdout that `active_model.pkl`'s deployed `contact_classifier` was
  itself evaluated against when it was retrained. Zero leakage: the
  classifier was trained only on the complementary 80% split.
- **Model under test**: the currently deployed `active_model.pkl`'s
  `contact_classifier` + `scaler` (loaded directly, not retrained).
- **Label**: `Contact = 1` iff the load cell's own force reading for that
  tick's facing finger is `>= 0.3N` (`CONTACT_FORCE_THRESHOLD`, matching
  `retrain_contact_with_load_cell_data.py` exactly).
- **Candidate thresholds swept**: `[0.30, 0.25, 0.20, 0.15, 0.10, 0.05,
  0.02]` -- the same coarse-grained range documented for Thumb's sweep, to
  avoid reading single-episode noise as a real trend (this project's
  established practice, e.g. `select_thresholds_consolidated.py`).
- **Selection standard** (same one applied to Thumb): maximize recall
  subject to precision staying perfect or near-perfect (no meaningful
  false-positive cost), then back off from the most aggressive threshold
  tested to leave margin, rather than adopting the absolute floor.

## Results

Holdout: 3,597 rows total across all 4 fingers, 28 reps.

| Finger | Reps held out | Threshold | Precision | Recall |
|---|---|---|---|---|
| **Index** | 9 | 0.30 (current) | 0.983 | **1.000** |
| | | 0.02 (most aggressive) | 0.947 | **1.000** |
| **Middle** | 7 | 0.30 (current) | 1.000 | 0.910 |
| | | 0.10 (**adopted**) | 1.000 | 0.928 |
| | | 0.02 (most aggressive) | 1.000 | 0.948 |
| **Ring** | 6 | 0.30 (current) | 0.946 | **1.000** |
| | | 0.02 (most aggressive) | 0.825 | **1.000** |
| **Thumb** (re-check) | 6 | 0.30 (current) | 1.000 | 0.569 |
| | | 0.10 (already deployed) | 1.000 | 0.713 |
| | | 0.02 (most aggressive) | 1.000 | 0.820 |

Full 7-point sweep per finger (`n` = holdout ticks for that finger, `n_pos_true` = genuine-touch ticks, `n_pos_pred` = ticks the classifier called contact at that threshold):

**Index** (n=1148, n_pos_true=1022)

| Threshold | Precision | Recall |
|---|---|---|
| 0.30 | 0.9827 | 1.0000 |
| 0.25 | 0.9817 | 1.0000 |
| 0.20 | 0.9771 | 1.0000 |
| 0.15 | 0.9687 | 1.0000 |
| 0.10 | 0.9660 | 1.0000 |
| 0.05 | 0.9534 | 1.0000 |
| 0.02 | 0.9472 | 1.0000 |

**Middle** (n=908, n_pos_true=653)

| Threshold | Precision | Recall |
|---|---|---|
| 0.30 | 1.0000 | 0.9096 |
| 0.25 | 1.0000 | 0.9173 |
| 0.20 | 1.0000 | 0.9188 |
| 0.15 | 1.0000 | 0.9250 |
| 0.10 | 1.0000 | 0.9280 |
| 0.05 | 1.0000 | 0.9387 |
| 0.02 | 1.0000 | 0.9479 |

**Ring** (n=775, n_pos_true=383)

| Threshold | Precision | Recall |
|---|---|---|
| 0.30 | 0.9457 | 1.0000 |
| 0.25 | 0.9433 | 1.0000 |
| 0.20 | 0.9387 | 1.0000 |
| 0.15 | 0.9207 | 1.0000 |
| 0.10 | 0.8991 | 1.0000 |
| 0.05 | 0.8665 | 1.0000 |
| 0.02 | 0.8254 | 1.0000 |

**Thumb** (n=766, n_pos_true=766 -- re-check, see caveat below)

| Threshold | Precision | Recall |
|---|---|---|
| 0.30 | 1.0000 | 0.5692 |
| 0.25 | 1.0000 | 0.5888 |
| 0.20 | 1.0000 | 0.6162 |
| 0.15 | 1.0000 | 0.6554 |
| 0.10 | 1.0000 | 0.7128 |
| 0.05 | 1.0000 | 0.7794 |
| 0.02 | 1.0000 | 0.8198 |

## Decisions

**Index -- no change.** Recall is already a perfect 1.000 at every
threshold tested, including the current 0.30 default -- there is no
recall problem for this holdout to fix. Lowering the threshold only costs
precision (0.983 -> 0.947 by 0.02), for zero benefit. Applying "maximize
recall subject to precision staying perfect" to this data means: **stay at
the current global default.** (Separately worth noting, out of scope for
this task: precision at 0.30 is only 0.983, meaning some false positives
already exist at the current threshold -- if anything this points toward
possibly RAISING Index's threshold in a future pass, the opposite
direction from what was being tested here.)

**Middle -- lowered to 0.10/0.05.** Precision is perfect (1.000) across
the entire sweep, and recall climbs cleanly and monotonically as the
threshold drops (0.910 -> 0.948). This is a genuine, zero-cost
improvement, the same shape Thumb's sweep is recorded as having shown.
0.10 (not the more aggressive 0.02) was chosen to match Thumb's existing
precedent -- same 2:1 confirm:release ratio, same margin-for-a-single-
holdout reasoning. **Added to `sensing_gui.py`**:
```python
CONTACT_P_CONFIRM_PER_FINGER["Middle"] = 0.10
CONTACT_P_RELEASE_PER_FINGER["Middle"] = 0.05
```
Live in the current code as of this report (2026-09-08) -- confirmed by
reloading `sensing_gui.py` after the edit.

**Ring -- no change.** Identical situation to Index: recall is already
1.000 at every threshold, including 0.30. Lowering only costs precision,
more severely than Index (0.946 -> 0.825 by 0.02). No override added.

## Thumb re-check -- discrepancy found, not resolved

Thumb was included in this sweep purely as a consistency check against
the already-adopted 0.10/0.05 override, using the identical `_split_load_
cell_dataset` call the original (undocumented) analysis must have used.

**What matches**: the holdout landed on the exact same **6 reps** the
original comment in `sensing_gui.py` cites ("6 reps never used in
training") -- confirming the seeded rep-grouped split is deterministic and
reproducible. Precision is perfect (1.000) throughout both analyses,
qualitatively matching the original claim.

**What doesn't match**: the recall numbers themselves. The original
comment records recall 0.815 at threshold 0.30 and 0.974 at 0.10. This
sweep measured **0.569 at 0.30 and 0.713 at 0.10** on the same 6 reps --
substantially lower at every threshold. Tick count is also slightly off
(766 here vs. 783 cited originally, a ~2% difference).

**UPDATE (2026-09-08): investigated in full, see `thumb_recall_discrepancy_
investigation.md`.** Ruled out as causes: a different model file (bit-
identical classifier/scaler/regressor confirmed between `active_model.pkl`
and the candidate) and randomness/seed-dependence (both the holdout split
and classifier inference are fully deterministic). Tested and found
insufficient: simulating the full live Schmitt-trigger + frame-hysteresis
state machine instead of a naive per-tick threshold -- it moves the 0.10
number partway (0.713 -> 0.782) but makes the 0.30 number **worse**, not
better (0.569 -> 0.488), the opposite of what would explain 0.815. Also
found: Thumb's holdout has zero true negatives, so "perfect precision" is
trivially guaranteed here regardless of classifier quality.

**Verdict: do not cite 0.815/0.974 anywhere.** No tested methodology
reproduces them and no original script/report exists to check against.
**Cite 0.569 (@0.30) / 0.713 (@0.10) instead** -- the numbers in this
report, backed by a checked-in, rerunnable script.

**Consequence**: the 0.10 Thumb override already deployed is still
justified as a real recall improvement **by these numbers** (0.569 ->
0.713, a genuine +14.4pp gain) -- so it has NOT been reverted. But
precision can't be meaningfully assessed for Thumb from this holdout at
all (no true negatives exist to test against), which makes live hardware
validation more important for Thumb specifically than for the other three
fingers. `sensing_gui.py`'s own comment has been updated with the full
verdict (see that file directly).

## Summary of changes made by this task

| Finger | Action | Live in current code? |
|---|---|---|
| Index | No change (already saturated recall) | n/a |
| Middle | New override added: 0.10 / 0.05 | Yes, confirmed 2026-09-08 |
| Ring | No change (already saturated recall) | n/a |
| Thumb | No change -- pre-existing 0.10/0.05 left as-is; discrepancy flagged | Unchanged |

All of the above is **offline-holdout-validated only**. None of it --
Middle's new override included -- has been tested on real hardware.
