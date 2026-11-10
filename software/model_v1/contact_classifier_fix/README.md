# Contact classifier self-closure fix

**Status: promoted and live.** `active_model_contact_selfclosure_fix.pkl`
was deployed to `model_v1/active_model.pkl` on 2026-07-13T23:02:43Z, and a
later Contact-classifier retrain (`true_zero_dataset/`) built directly on
top of it; see `model_v1/README.md`'s "How `active_model.pkl` was built"
section for the current full lineage. The description below documents the
pipeline as originally designed and run, when it was still a standalone,
additive fix pipeline that only *read* `model_v1/crosstalk/raw/*.json`,
`model_v1/train_data_backregion.csv`, `model_v1/test_data_backregion.csv`,
and `model_v1/active_model.pkl` as data/library sources, before its output
was promoted.

## The bug

`model_v1/self_motion/`'s investigation found that the deployed Contact
classifier misreads a finger sitting at full self-closure (100%, servo
fully closed, nothing touching it) as genuine contact:

- **Middle, Ring, Thumb**: a STABLE, sustained false `Contact=1` for
  100% of frames captured at rest at 100% closure -- not flickering
  noise, a confidently wrong reading that would show as a persistent
  "CONTACT" indicator in the live GUI.
- **Index**: doesn't sustain a false confirmed contact, but its raw
  classifier oscillates abnormally (~40 Hz) at the same position.

Root cause: the classifier was trained exclusively on jig data
(`master_characterisation*.csv`, Units 1+2) -- bench presses of a
stationary fingertip against a load cell. That protocol has no concept
of "this finger's own AmazingHand servo has closed it against nothing"
-- a self-closure delta this large, with no touch, was simply never a
training example.

## The fix

Add supplementary Contact=0 training data spanning the full 0-100%
self-closure range, on all 4 real fingers, and retrain ONLY the Contact
classifier -- everything else (Force/Theta regressor, Phi classifier,
shared scaler) is carried over from the deployed model byte-for-byte
unchanged.

## Pipeline

```
python model_v1/contact_classifier_fix/build_selfclosure_contact_dataset.py
python model_v1/contact_classifier_fix/retrain_contact_classifier.py
python model_v1/contact_classifier_fix/evaluate_contact_fix.py
```

### 1. `build_selfclosure_contact_dataset.py`

**No new hardware recording was needed.** `model_v1/crosstalk/raw/*.json`
already IS exactly this experiment, collected for an unrelated purpose:
each of the 4 fingers swept through 0/25/50/75/100% closure, idle (no
touch), other 3 fingers held fixed at 0% throughout, ~4s settled capture
per position. `model_v1/self_motion/`'s own finding -- no early-vs-late
decay within a stationary phase -- confirms this is a steady-state
effect, not a transient the crosstalk protocol's 1.5s settle window
would have missed, so that data is fully representative of the exact
condition being fixed. Re-recording identical data would have been
redundant.

For each of the 20 crosstalk recordings, this script extracts the MOVING
finger's own sensor readings (its own group, while sitting at that
position -- a different use of the same file than
`model_v1/crosstalk/build_crosstalk_dataset.py` makes, which looks at
the effect on the OTHER 3 stationary fingers instead), computes delta
relative to that finger's own 0%-position mean (the same reference
`sensing_gui.py`'s live baseline uses), and the matching `S{n}_mag`
delta-magnitude features -- in the exact column convention
`characterisation_delta_combined.csv` uses (`dS1X..dS5Z`, `S1_mag..S5_mag`),
so the output drops straight into the existing training pipeline.

`Force=0.0` for every row -- a true, faithful label (these are
genuinely idle, no-touch captures), which the existing
`Contact = Force >= 0.3` formula turns into `Contact=0` automatically,
no special-casing needed. `Unit=4` marks this data source distinctly
from jig Units 1/2/3 (it is NOT a jig unit -- it's the deployed
AmazingHand's own 4 fingertips). `Theta`/`Phi` are unused placeholders
(0) -- this data is never fed to the regressor or Phi classifier.

Output: `selfclosure_contact_dataset.csv`, ~8400 rows (4 fingers x 5
positions x ~400-420 frames each).

### 2. `retrain_contact_classifier.py`

Combines the existing jig-only (Units 1+2) Contact training rows from
`train_data_backregion.csv` with all of `selfclosure_contact_dataset.csv`,
transforms both through the **existing deployed scaler, reused
unmodified** (never refit -- a classifier's decision boundary is
calibrated to whatever coordinate system its training scaler produces;
refitting would silently miscalibrate the regressor/Phi classifier being
carried over unchanged), computes Contact-balanced sample weights on the
combined set, and trains a fresh `MLPClassifier` with the identical
architecture/hyperparameters `train_model_forcefix_v2.py` uses for this
head.

Saves the new bundle to `active_model_contact_selfclosure_fix.pkl` --
identical to `active_model.pkl` in every field except `contact_classifier`
(replaced) and `head_training_data`/a new `contact_classifier_fix_note`
(updated metadata). **`active_model.pkl` is never written to.**

### 3. `evaluate_contact_fix.py`

Three checks -- see `contact_fix_report.md` for the actual numbers:

- **(a) Leave-one-finger-out cross-validation** on the self-closure data:
  for each finger, trains on jig + the OTHER 3 fingers' self-closure data,
  tests on the held-out 4th finger's own data -- genuinely unseen-unit
  generalisation, broken out per position with 100% called out
  specifically (this is the primary check the fix request asked for).
- **(b) Regression check**: the fix vs. the currently-deployed model,
  both scored on the SAME existing held-out jig test set
  (`test_data_backregion.csv`) `evaluate_model_forcefix_v2.py` already
  uses -- confirms genuine-touch detection didn't get worse.
- **(c) Training-fit sanity check**: the final model against ALL
  self-closure data (not held out -- explicitly labelled as training-fit,
  matching this project's own convention, e.g.
  `build_split_backregion.py`'s Phi=-45/+45 documentation), alongside the
  currently-deployed model's score on the same data (which reproduces
  the original bug numerically).

## Results (2026-07 run, see `contact_fix_report.md` for full tables)

**The fix works, cleanly, for the actual deployed hand.** The
methodologically important point: evaluating this required TWO different
held-out tests, because they answer different questions, and the first
one I ran gave a misleading headline number by answering the wrong one.

- **(a) PRIMARY check -- leave-100%-out per finger, same 4 units**: for
  each finger, its own 100% rows were held out of training entirely
  (trained on jig + that finger's own 0/25/50/75% + the other 3 fingers'
  full data including their 100%), then tested on the held-out 100%
  rows. **100.0% accuracy for all 4 fingers** (Index, Middle, Ring,
  Thumb) -- vs. the currently-deployed model's 14.2% accuracy at the
  same position (i.e. it wrongly reads Contact=1 85.8% of the time with
  nothing touching it). This is the directly relevant result: these are
  the exact 4 fingertips this model runs on in production, and the 100%
  data was genuinely never seen during that fold's training.
- **(a2) SECONDARY check -- leave-one-finger-out (cross-UNIT transfer)**:
  a harder, different question -- would this fix transfer to a fingertip
  that contributed NO data at all (e.g. after a physical swap)? Only
  25.0% mean accuracy at 100% across the 4 held-out-unit folds (ranging
  0% for 3 fingers to 100% for Thumb, whose signature happened to
  resemble the other 3's enough to generalise). **This is an honest,
  real limitation, not a failure of the fix for its actual purpose**:
  each fingertip has its own absolute magnetic scale (the same pattern
  already documented elsewhere in this project, e.g. Unit 3's deltas
  being 10-40x Units 1/2's), so a decision boundary learned from 3
  fingertips doesn't automatically transfer to a 4th, never-seen one. If
  a fingertip is ever physically swapped, this fix should be re-derived
  with fresh self-closure data for the new unit.
- **(b) Regression check**: no regression on genuine-touch detection --
  held-out jig test (`test_data_backregion.csv`, same set
  `evaluate_model_forcefix_v2.py` uses) Contact accuracy went 88.4% ->
  88.8% (+0.41pp).
- **(c) Training-fit sanity check** (not held out, included for
  reference only): the final model scores 99.7% on all self-closure
  data combined (100% at every position except 0%'s 98.3%), vs. the
  currently-deployed model's 38.4% overall on the same data -- the
  original bug numerically reproduced (25-28% accuracy at 25/50/75%,
  14.2% at 100%, only 0% itself already being fine since that's what
  the model was calibrated against).

## Known limitations

- Only 20 source recordings (4 fingers x 5 positions, one recording
  each) -- no repeat sessions to check consistency over time/thermal
  drift, unlike the crosstalk additivity validation's order-reversed
  replicates.
- The self-closure data all comes from ONE session on the currently
  mounted 4 fingertips -- if fingertips are ever swapped/remounted, this
  supplementary data (and therefore the fix) may not transfer as well as
  intended, the same caveat that already applies to any hand-specific
  data in this pipeline.
- Combined jig+self-closure training-fit accuracy (~90%, see
  `retrain_contact_classifier.py`'s own console output) is noticeably
  below either dataset's accuracy in isolation -- expected, since a
  large self-closure delta and a large genuine-touch delta can land in
  overlapping regions of the 20-feature space despite having opposite
  labels; this is a genuine class-overlap problem the MLP has to
  compromise on, not a bug in the training code.
- This only fixes the CLASSIFIER's response to self-closure at the 4
  positions this hand was tested at (plus whatever the classifier
  interpolates in between) -- it does not change the crosstalk
  correction, the Force/Theta regressor, or the Phi classifier, all
  deliberately left untouched per the fix's scope.

## Deploying the fix

Not done automatically by this pipeline itself -- per the original request,
`active_model.pkl` was left unchanged until a human reviewed
`contact_fix_report.md` and decided to promote it. That review happened and
the fix **was promoted** on 2026-07-13T23:02:43Z (copied/renamed
`active_model_contact_selfclosure_fix.pkl` to `model_v1/active_model.pkl`,
the same mechanism used for every other model iteration in this project). Its
Contact classifier was subsequently superseded by the `true_zero_dataset/`
retrain (2026-07-17) and then by the load-cell retrain of 2026-09-07
(`grasping_dashboard/retrain_contact_with_load_cell_data.py`), which is what is
currently live; see `model_v1/README.md` for the up-to-date lineage.
