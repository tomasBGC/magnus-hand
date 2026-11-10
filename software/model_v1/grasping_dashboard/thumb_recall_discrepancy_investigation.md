# Investigation: Thumb recall discrepancy (0.569/0.713 measured vs. 0.815/0.974 originally cited)

> **Verdict up front**: the originally-cited 0.815/0.974 recall figures for
> Thumb could **not** be reproduced by any evaluation methodology tested
> here, including a faithful simulation of the live Schmitt-trigger
> hysteresis. **Do not cite 0.815/0.974 anywhere.** The numbers in
> `sweep_finger_thresholds.py`'s output (0.569 @ 0.30, 0.713 @ 0.10 --
> simple per-tick precision/recall) are the ones backed by a checked-in,
> rerunnable script and should be treated as authoritative until/unless the
> original methodology is recovered.

## (1) Is the same evaluation code / methodology being used?

**No original code exists to compare against** -- confirmed again in this
investigation (searched every `.py`/`.md` file for any trace of a separate
Thumb sweep script or report; none exists, same finding as the earlier
`thumb_threshold_live_validation_report.md` investigation).

Tested whether the discrepancy is explained by a **different definition of
"confirmed"** -- specifically, whether the original figures were computed
against the live, stateful Schmitt-trigger + frame-hysteresis
`confirmed_contact` signal (`sensing_gui.py`'s `_process_one_finger`:
`CONTACT_CONFIRM_FRAMES=4` / `CONTACT_RELEASE_FRAMES=5`, a dead zone
between the release and confirm thresholds, plus the `MIN_CONTACT_FORCE_N`
gate) rather than a memoryless per-tick `probability >= threshold`
comparison, which is what `sweep_finger_thresholds.py` actually measures.
This is a real, plausible gap: the codebase itself distinguishes these two
notions elsewhere ("live Schmitt-trigger recall is higher" appears in
`active_model.pkl`'s own `contact_classifier_fix_note` for the raw 34%->53%
figure).

Built `investigate_thumb_recall_discrepancy.py`, a byte-for-byte port of
`_process_one_finger`'s streak logic, run forward in tick order over each
held-out rep (state reset per rep):

| Threshold | Naive per-tick (no hysteresis) | With full hysteresis + Force gate | Originally cited |
|---|---|---|---|
| 0.30 | recall=0.569 | recall=**0.488** | 0.815 |
| 0.10 | recall=0.713 | recall=**0.782** | 0.974 |

**Hysteresis moves the number, but not consistently toward the claim, and
not far enough.** At 0.10 it closes part of the gap (0.713 -> 0.782, still
19 points short of 0.974). At 0.30 it makes the number **worse**, not
better (0.569 -> 0.488) -- the opposite of what would be needed to explain
0.815. The `MIN_CONTACT_FORCE_N` gate (using the model's own regressor
Force output, not the load-cell ground truth) turned out to almost never
bind -- 97% of ticks already exceed it -- so it contributes essentially
nothing to the discrepancy (599 vs 600 TP, a 1-tick difference).

Also tested **per-rep "was contact ever confirmed"** (a coarser,
event-level recall notion) as a second methodology hypothesis:

| Threshold | Rep-level recall (3/6 reps at 0.30, 5/6 at 0.10) |
|---|---|
| 0.30 | 3/6 = 0.500 |
| 0.10 | 5/6 = 0.833 |

This revealed a striking **bimodal pattern**: at 0.30, 3 of the 6 held-out
reps are confirmed for ~97.7% of their ticks, and the other 3 are confirmed
for **0%** of their ticks (never sustain 4 consecutive above-threshold
frames at all) -- an entirely-hit-or-entirely-missed split, not a gradual
one. At 0.10, one previously-0% rep jumps to 95.3%, another to 81.2%, but
one rep still never confirms at all. This qualitatively matches the
original comment's own description ("the missed touches at 0.30 were real,
low-force contacts... not ambiguous/noisy cases") -- but the resulting
numbers (0.500, 0.833) don't match 0.815/0.974 either, and 783/6 isn't a
clean fit for a rep-count denominator in the first place (0.815 and 0.974
are far more consistent with a tick-level metric, given the originally
cited "783 held-out ticks").

**Conclusion for (1): methodology likely differed in some way (hysteresis
vs. naive, possibly something else entirely), but no tested variant
reproduces the cited numbers, and the 0.30 result actively contradicts the
hysteresis hypothesis. Root cause not fully identified.**

## (2) Is this the same exact model file?

**Yes -- confirmed bit-identical.** Directly compared `active_model.pkl`'s
`contact_classifier` MLP weight arrays (`.coefs_`) against `active_model_
load_cell_contact_candidate.pkl`'s: identical architecture, `np.array_equal`
True on every layer. Same for the `scaler` (`mean_`/`scale_` arrays
identical) and the `regressor`. The only difference between the two bundle
files is the `contact_classifier_fix_note` text itself (edited after the
candidate was saved, to add the "LIVE-VALIDATED... and DEPLOYED" sentence)
-- which also explains the file-size difference flagged in an earlier
session. **Ruled out as a cause of the discrepancy.**

## (3) Randomness / seed-dependence in holdout processing?

**Ruled out.** The rep-grouped holdout split (`_split_load_cell_dataset`,
seed 42) is deterministic: this investigation's fresh run landed on the
same 6 source files the original comment's rep count implies, both here
and in the earlier `sweep_finger_thresholds.py` run. `MLPClassifier.
predict_proba` on fixed, already-trained weights is a deterministic
forward pass -- no randomness at evaluation time. No seed-dependence found
anywhere in the pipeline that could explain a result this large.

## Additional finding: Thumb's holdout has zero true negatives

Every one of the 766 held-out Thumb ticks has `Contact=1`
(`n_pos_true == n`) -- there is no genuinely-no-touch Thumb tick anywhere
in this holdout. This means **"precision = 1.000" is trivially guaranteed
for Thumb at every threshold tested, regardless of how good or bad the
classifier actually is** -- there are no true negatives available to be
misclassified as false positives. This doesn't affect the recall
discrepancy directly, but it means the "perfect precision" finding (both
in the original claim and in this reproduction) carries less evidentiary
weight than it might appear to -- it isn't testing false-positive behavior
at all for Thumb, only recall. Index/Middle/Ring's holdouts do contain
true negatives (confirmed: `n_pos_true < n` for all three), so their
precision numbers in `per_finger_threshold_sweep_report.md` don't have
this issue.

## Recommendation

- **Cite the numbers from `sweep_finger_thresholds.py`** (0.569 @ 0.30,
  0.713 @ 0.10, simple per-tick precision/recall, same metric used for
  Index/Middle/Ring) as the current best-supported figures for Thumb --
  they're backed by a checked-in, rerunnable script using the deployed
  model.
- **Do not cite 0.815/0.974** until/unless the original methodology is
  recovered or independently reproduced by some means not yet tried.
- The Thumb 0.10 override remains justified either way (naive: 0.569 ->
  0.713, hysteresis: 0.488 -> 0.782 -- both show a real, if smaller than
  claimed, zero-precision-cost improvement... except precision at Thumb
  isn't a meaningful test either way, per the finding above). Left
  deployed, not reverted.
- Given precision can't be meaningfully assessed for Thumb from this
  holdout at all, **live hardware validation matters more for Thumb than
  for the other three fingers** -- it's the only way to actually test
  Thumb's false-positive rate, since this offline holdout structurally
  cannot.
