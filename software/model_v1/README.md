# model_v1 — Force/Theta/Phi Model + Live Inference

Maps raw eFlesh magnetometer readings to contact **Force** (N), **Theta**
(deg, regression) and **Phi** (deg, classified into {-90, 0, +90} in the
currently deployed model; see "How `active_model.pkl` was built" below for
why this is 3 classes rather than the 5-class {-90,-45,0,45,90} set an
earlier back-region baseline used), plus a dedicated no-contact **Contact**
classifier. Trained on delta (baseline-subtracted) sensor features from
`recordings/master_characterisation*.csv`.

> **Note on this repository's layout.** The text below was written for the
> original working folder. In this repository the model files live in
> `../models/` (`models/deployed/active_model.pkl` is the deployed model) and
> the datasets in `../data/`; intermediate datasets and raw recordings are not
> included. Paths in the scripts and prose below still use the original
> layout (`model_v1/...`, `recordings/...`). See `../models/README.md` and
> `../data/README.md`.

## What's live

Only these files are used at runtime — everything else in this folder is
training/experiment history, kept for reference:

| File | Role |
|------|------|
| `active_model.pkl` | **The deployed model.** Loaded by `MODEL_PATH` in `live_predictor_test.py`. |
| `live_phiclass_predictor.py` | `LivePhiClassPredictor` — loads the model bundle, runs per-finger inference on live sensor frames. |
| `live_predictor_test.py` | Headless console test: connects, predicts, prints. No GUI. Good first sanity check. |
| `sensing_gui.py` | Standalone 4-finger live tactile-monitoring GUI (no servo control, no recording). |
| `hand_control_gui.py` | Full demo: `sensing_gui.py`'s display + `AmazingHand/hand_controller.py`'s servo control, one shared serial connection. **This is the "run the hand" entry point.** Subclasses `sensing_gui.py`'s `MainWindow` rather than duplicating it, so `sensing_gui.py` is a hard dependency, not a redundant standalone copy. |

```bash
uv run model_v1/live_predictor_test.py     # console only
uv run model_v1/sensing_gui.py             # sensing GUI, no servos
uv run model_v1/hand_control_gui.py        # full hand demo
```

If you retrain and want a new model to go live, save it as (or rename it to)
`active_model.pkl` and it's picked up automatically by all three tools above
(`sensing_gui.py` and `hand_control_gui.py` both import `MODEL_PATH`
from `live_predictor_test.py`, so there's exactly one place the active
model's filename is defined).

## How `active_model.pkl` was built (data lineage)

**Last verified against the live pickle: 2026-11 (this repository's copy of
`models/deployed/active_model.pkl`).** The three heads share
one scaler but were retrained independently over several rounds, so the
current bundle is a composite of multiple promotion events, not the output
of one training run. This can be inspected directly at any time:

```python
import joblib
bundle = joblib.load("models/deployed/active_model.pkl")  # original layout: model_v1/active_model.pkl
print(bundle["head_training_data"])
# {'regressor_force_theta': 'combined_units_1_2_3_balanced_scaler_and_weights',
#  'classifier_phi': 'jigonly_units_1_2_no_theta0',
#  'contact_classifier': 'jigonly_units_1_2_PLUS_selfclosure_PLUS_true_zero_500combo_simultaneous_4finger_PLUS_load_cell_dataset'}
```

### Regressor (Force, Theta) + scaler — unchanged since 2026-07-13

```
recordings/master_characterisation.csv    (Unit 1)
recordings/master_characterisation_2.csv  (Unit 2)
recordings/master_characterisation_3.csv  (Unit 3, back-region supplement)
        │
        ▼  build_delta_dataset_backregion.py   (baseline-subtract, add magnitude features)
characterisation_delta_combined_backregion.csv
        │
        ▼  build_split_backregion.py   (recording-level train/test split)
train_data_backregion.csv + test_data_backregion.csv
        │
        ▼  train_model_forcefix_v2.py   (per-unit scalers, median-combined; fixes a
        │                                scaler-dilution bug in train_model_forcefix.py)
regressor + scaler   (evaluated by evaluate_model_forcefix_v2.py)
```

Every later promotion below reuses this regressor and scaler byte-for-byte;
none of them retrain it.

### Phi classifier — promoted 2026-07-17

```
train_data_backregion.csv, filtered to Units 1+2 only ("jig-only"), Theta != 0
        │
        ▼  production_review/08_phi_head_no_theta0.py
phi classifier   (3 classes: {-90, 0, 90} — jig-only data never contains
                  Phi=-45/+45, those only occur in Unit 3 back-region rows,
                  which this head deliberately excludes; see "Why only 3
                  Phi classes" below)
```

Promoted as part of the `production_review/` candidate bundle alongside the
isotonic `contact_calibrator` described in that folder's README. (The
calibrator was later removed again in the 2026-09-07 Contact retrain below,
because it was fitted to the old Contact classifier's outputs and is not valid
for a differently trained one; the deployed bundle has `contact_calibrator =
None`.)

### Contact classifier — four sequential retrains, most recent live since 2026-09-07

```
train_data_backregion.csv, Units 1+2 ("jig-only")
        │
        ▼  train_model_contact.py / train_model_contact_backregion.py
baseline contact classifier
        │
        ▼  contact_classifier_fix/build_selfclosure_contact_dataset.py
        │   (adds Contact=0 rows at 0/25/50/75/100% self-closure, all 4
        │    AmazingHand fingers, sourced from model_v1/crosstalk/raw/*.json)
        ▼  contact_classifier_fix/retrain_contact_classifier.py
active_model_contact_selfclosure_fix.pkl   → promoted 2026-07-13T23:02:43Z
        │
        ▼  true_zero_dataset/build_true_zero_dataset.py
        │   (adds true_zero_dataset.csv, 95,448 rows from 400 of 500
        │    simultaneous-4-finger no-touch position combinations)
        ▼  true_zero_dataset/retrain_contact_from_true_zero.py
active_model_true_zero_fix.pkl   → promoted 2026-07-17 as candidate_v3,
                                    reverted to the pre-true-zero (selfclosure-fix)
                                    classifier earlier on 2026-07-20 after live
                                    A/B testing showed reduced light-touch
                                    recall, then restored 2026-07-20T18:35:15Z
                                    after weighing that tradeoff against the
                                    ~100x multi-finger false-positive reduction
                                    (see true_zero_dataset/v2_revert_verification_report.md
                                    and true_zero_dataset/true_zero_restore_verification_report.md)
        ▼  grasping_dashboard/build_load_cell_dataset.py
        │   (load_cell_dataset.csv: 18,067 rows from 140 grasp reps with an
        │    independent load-cell force reference)
        ▼  grasping_dashboard/retrain_contact_with_load_cell_data.py
        │   (ONLY the Contact head; jig + self-closure + ALL 500 true-zero
        │    combinations + the load-cell dataset's TRAIN split -- 14,470 rows,
        │    reps grouped so no rep is in both train and holdout)
CURRENT contact classifier   → live-validated on hardware and deployed 2026-09-07
```

Note on the true-zero step: `active_model_true_zero_fix.pkl` (kept in this
repository as `models/after_true_zero/`) was trained on only 400 of the 500
true-zero combinations, with 100 combinations held out for evaluation. The
2026-09-07 retrain reuses the whole 500-combination `true_zero_dataset.csv`.
The deployed bundle therefore differs from `models/after_true_zero/` in its
Contact classifier (and has no `contact_calibrator`); its Force/Theta regressor,
Phi classifier and scaler are unchanged. Offline held-out contact accuracy on
the load-cell holdout moved from Index 11% to 99%, Middle 28% to 93%, Ring 51%
to 98% and Thumb 34% to 53% (per-tick `predict()` cutoff; the live detector
uses a probability Schmitt trigger, so live recall is higher).

### Why only 3 Phi classes

`train_model_contact_backregion.py` originally expanded Phi to 5 classes
({-90,-45,0,45,90}) by combining Units 1+2+3. That combined-data Phi head
regressed on Units 1+2 (the deployed hand's own units) and scored near
chance level on the back region, so `train_model_forcefix.py` /
`_v2` and every Phi retrain since (including the currently deployed
`08_phi_head_no_theta0.py` head) reverted to jig-only (Units 1+2) training
data, where Phi never takes the -45/+45 values. The 5-class set therefore
only ever existed in that one abandoned intermediate baseline, not in any
model that was subsequently deployed.

## Experiment history (superseded or not deployed)

These document the path taken to get to `active_model.pkl` and are **not**
expected to run out of the box — several reference intermediate CSVs or
earlier `.pkl` models that were deleted once superseded (they're regenerable
from the raw `master_characterisation*.csv` files if ever needed again).
Entries marked **[live]** contributed a head to the currently deployed
`active_model.pkl`; the rest are superseded or were evaluated and rejected.

- `build_delta_dataset.py` / `build_split.py` — pre-back-region dataset (Units 1+2 only). Superseded by the `_backregion` versions below.
- `train_model_contact.py` / `evaluate_model_contact.py` — the original baseline model, trained on the pre-back-region split.
- `train_model_contact_backregion.py` / `evaluate_model_contact_backregion.py` — added Unit 3 (back-region) data; its 5-class Phi head was later abandoned (see above).
- `train_model_forcefix.py` / `evaluate_model_forcefix.py` — first fix for a scaler-dilution issue, superseded by `_v2`.
- `train_model_forcefix_v2.py` / `evaluate_model_forcefix_v2.py` — **[live]** fixes the scaler-dilution issue found in v1; produced the regressor and scaler still in use today.
- `contact_classifier_fix/` — **[live]** self-closure Contact classifier fix, promoted 2026-07-13. See `contact_classifier_fix/README.md`.
- `production_review/` — **[live]** Phi head retrained without Theta=0 rows, plus an isotonic contact probability calibrator, promoted 2026-07-17. See `production_review/README.md`. Also evaluated and rejected a full GBM architecture swap, rate-of-change features, and a GBM Contact head.
- `true_zero_dataset/` — **[live]** simultaneous-4-finger true-zero Contact classifier fix, promoted 2026-07-17, reverted and restored 2026-07-20 (see above). See `true_zero_dataset/README.md`.
- `grasping_dashboard/` — **[live]** load-cell grasp data collection (`dashboard_gui.py`), dataset builder and the 2026-09-07 Contact retrain; `retrain_contact_with_load_cell_data.py` defines the train/holdout split reused by `analysis/common.py`.
- `candidate_v2/`, `candidate_v4/`, `candidate_v5/` — (not included in this repository; `models/before_true_zero/` is candidate_v2's model) packaged snapshots of the live system at various points; `candidate_v4`'s Schmitt-threshold change and `candidate_v5`'s baseline-settle-delay change were both evaluated and reverted/rejected as GUI-only decision-layer changes, and never altered `active_model.pkl`.

Each script's docstring explains what it changed and why relative to the
previous one, if you want the full history.
