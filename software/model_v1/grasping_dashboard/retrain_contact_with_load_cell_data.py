"""Retrains ONLY the Contact classifier head using the new load-cell dataset
(build_load_cell_dataset.py's output) added on top of the existing training
data -- same "ONLY this head" pattern already used three times in this
project's history (self-closure fix, true_zero fix, production review; see
active_model.pkl's own contact_classifier_fix_note / true_zero_fix_note).
The regressor (Force+Theta), Phi classifier, and StandardScaler are copied
BYTE-FOR-BYTE from active_model.pkl, never refit -- LivePhiClassPredictor
applies one shared scaler before all three heads' predict() calls, so
touching it would silently break the other two heads' calibration, and
there is no Theta/Phi ground truth in load-cell data to retrain them with
anyway (the load cell measures scalar force only, not direction).

Training data (Contact label = Force >= CONTACT_FORCE_THRESHOLD throughout,
matching train_model_forcefix.py exactly):
  - train_data_backregion.csv, Units 1+2 only (jig-only, matching the
    deployed classifier's own "jigonly_units_1_2" lineage)
  - contact_classifier_fix/selfclosure_contact_dataset.csv
  - true_zero_dataset/true_zero_dataset.csv
  - load_cell_dataset.csv's TRAIN split (see below)

Evaluation is the new_load_cell_dataset's HOLDOUT split ONLY -- grouped by
source rep file (not row), so near-identical hold-phase ticks from the same
rep never appear on both sides. This is the first Contact evaluation in
this project's history against a ground truth that is BOTH independent
(physically separate load cell, not the model's own estimate) AND capable
of covering genuine multi-finger simultaneous grasps (unlike the original
single-finger jig). Reports the EXISTING deployed classifier's accuracy on
this holdout as the baseline, then the new candidate's, broken down by
finger -- Index is the specific, previously-documented failure case this
is aimed at (see index-contact-threshold-not-fixable in project history).

Usage: uv run python model_v1/grasping_dashboard/retrain_contact_with_load_cell_data.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.neural_network import MLPClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_V1_DIR = SCRIPT_DIR.parent
TRAINING_DIR = SCRIPT_DIR.parent / "grasping_dashboard_data" / "training_protocol"

ACTIVE_MODEL_PATH = MODEL_V1_DIR / "active_model.pkl"
BACKREGION_CSV = MODEL_V1_DIR / "train_data_backregion.csv"
SELFCLOSURE_CSV = MODEL_V1_DIR / "contact_classifier_fix" / "selfclosure_contact_dataset.csv"
TRUE_ZERO_CSV = MODEL_V1_DIR / "true_zero_dataset" / "true_zero_dataset.csv"
LOAD_CELL_CSV = TRAINING_DIR / "load_cell_dataset.csv"
CANDIDATE_OUT = MODEL_V1_DIR / "active_model_load_cell_contact_candidate.pkl"

CONTACT_FORCE_THRESHOLD = 0.3
DELTA_COLS = [f"d{c}" for c in [f"S{i}{axis}" for i in range(1, 6) for axis in "XYZ"]]
MAGNITUDE_COLS = [f"S{i}_mag" for i in range(1, 6)]
INPUT_COLS = DELTA_COLS + MAGNITUDE_COLS

MLP_KWARGS = dict(
    hidden_layer_sizes=(64, 32, 16), activation="relu", solver="adam",
    alpha=1e-4, learning_rate_init=1e-3, max_iter=500, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=15, random_state=42, verbose=False,
)

HOLDOUT_FRACTION = 0.2
RNG = np.random.default_rng(42)


def _balanced_weight(contact: pd.Series) -> np.ndarray:
    n0, n1 = (contact == 0).sum(), (contact == 1).sum()
    weight_map = {0: len(contact) / (2 * n0), 1: len(contact) / (2 * n1)}
    return contact.map(weight_map).values


def _split_load_cell_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rep-level (source_file) holdout, stratified per finger so every
    finger appears in both splits."""
    train_frames, holdout_frames = [], []
    for finger, sub in df.groupby("finger"):
        reps = np.array(sub["source_file"].unique(), dtype=object)
        RNG.shuffle(reps)
        n_holdout = max(1, round(len(reps) * HOLDOUT_FRACTION))
        holdout_reps = set(reps[:n_holdout])
        holdout_frames.append(sub[sub["source_file"].isin(holdout_reps)])
        train_frames.append(sub[~sub["source_file"].isin(holdout_reps)])
    return pd.concat(train_frames, ignore_index=True), pd.concat(holdout_frames, ignore_index=True)


def _evaluate(name: str, clf, scaler, holdout: pd.DataFrame) -> None:
    X = scaler.transform(holdout[INPUT_COLS].values)
    y_true = holdout["Contact"].values
    y_pred = clf.predict(X)
    print(f"\n--- {name} on load-cell holdout (n={len(holdout)}) ---")
    print(f"  overall accuracy={accuracy_score(y_true, y_pred):.3f} "
          f"precision={precision_score(y_true, y_pred, zero_division=0):.3f} "
          f"recall={recall_score(y_true, y_pred, zero_division=0):.3f}")
    for finger, sub in holdout.groupby("finger"):
        # holdout's index is a fresh 0..len-1 range (built with
        # ignore_index=True), so it doubles directly as positions into the
        # y_true/y_pred arrays, which follow the same row order.
        pos = sub.index.to_numpy()
        yt, yp = y_true[pos], y_pred[pos]
        acc = accuracy_score(yt, yp)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        print(f"  {finger:6s} n={len(pos):4d} acc={acc:.3f}  "
              f"(TP={tp} FN={fn} TN={tn} FP={fp})")


def main() -> None:
    bundle = joblib.load(ACTIVE_MODEL_PATH)
    scaler = bundle["scaler"]
    old_contact_clf = bundle["contact_classifier"]

    print("Loading existing training data...")
    backregion = pd.read_csv(BACKREGION_CSV, usecols=["Unit", "Force"] + INPUT_COLS)
    backregion = backregion[backregion["Unit"].isin([1, 2])]
    backregion["Contact"] = (backregion["Force"] >= CONTACT_FORCE_THRESHOLD).astype(int)

    selfclosure = pd.read_csv(SELFCLOSURE_CSV, usecols=["Force"] + INPUT_COLS)
    selfclosure["Contact"] = (selfclosure["Force"] >= CONTACT_FORCE_THRESHOLD).astype(int)

    true_zero = pd.read_csv(TRUE_ZERO_CSV, usecols=["Force"] + INPUT_COLS)
    true_zero["Contact"] = (true_zero["Force"] >= CONTACT_FORCE_THRESHOLD).astype(int)

    load_cell = pd.read_csv(LOAD_CELL_CSV)
    lc_train, lc_holdout = _split_load_cell_dataset(load_cell)
    print(f"load_cell_dataset.csv: {len(load_cell)} rows -> {len(lc_train)} train / {len(lc_holdout)} holdout "
          f"({lc_holdout['source_file'].nunique()} held-out reps)")

    print("\n=== BASELINE: existing deployed contact_classifier ===")
    _evaluate("DEPLOYED (current active_model.pkl)", old_contact_clf, scaler, lc_holdout)

    train_df = pd.concat([
        backregion[INPUT_COLS + ["Contact"]],
        selfclosure[INPUT_COLS + ["Contact"]],
        true_zero[INPUT_COLS + ["Contact"]],
        lc_train[INPUT_COLS + ["Contact"]],
    ], ignore_index=True)
    print(f"\nCombined training set: {len(train_df)} rows "
          f"(backregion={len(backregion)}, selfclosure={len(selfclosure)}, "
          f"true_zero={len(true_zero)}, load_cell_train={len(lc_train)})")
    print("Contact balance:", train_df["Contact"].value_counts().to_dict())

    X_train = scaler.transform(train_df[INPUT_COLS].values)
    y_train = train_df["Contact"].values
    sample_weight = _balanced_weight(train_df["Contact"])

    print("\nTraining candidate contact_classifier...")
    new_clf = MLPClassifier(**MLP_KWARGS)
    new_clf.fit(X_train, y_train, sample_weight=sample_weight)

    print("\n=== CANDIDATE: retrained with load-cell data ===")
    _evaluate("CANDIDATE (load-cell-augmented)", new_clf, scaler, lc_holdout)

    candidate_bundle = dict(bundle)
    candidate_bundle["contact_classifier"] = new_clf
    candidate_bundle["contact_calibrator"] = None
    candidate_bundle["contact_calibrator_note"] = (
        "REMOVED in this candidate -- was isotonic-calibrated for the OLD "
        "contact_classifier's probability outputs specifically; not valid "
        "for a differently-trained classifier without being refit.")
    candidate_bundle["contact_classifier_fix_note"] = (
        "2026-09-07: retrained (ONLY this head) adding load_cell_dataset.csv "
        f"({len(lc_train)} train rows from the grasping_dashboard load-cell "
        "protocol -- the first Contact training data from a physically "
        "independent, multi-finger-capable ground truth). Regressor, Phi "
        "classifier, and scaler are byte-for-byte unchanged from the prior "
        "active_model.pkl. See retrain_contact_with_load_cell_data.py's own "
        "holdout evaluation (grouped by rep, held out from training) for "
        "before/after per-finger accuracy. NOT yet live-validated or deployed.")
    candidate_bundle["head_training_data"] = dict(bundle.get("head_training_data", {}))
    candidate_bundle["head_training_data"]["contact_classifier"] = (
        "jigonly_units_1_2_PLUS_selfclosure_PLUS_true_zero_500combo_simultaneous_4finger_PLUS_load_cell_dataset")

    joblib.dump(candidate_bundle, CANDIDATE_OUT)
    print(f"\nCandidate bundle saved to {CANDIDATE_OUT} -- NOT deployed (active_model.pkl untouched).")


if __name__ == "__main__":
    main()
