"""Step 5: retrains ONLY the Contact classifier, adding true_zero_dataset.csv
(500 simultaneous-4-finger combinations, genuinely no touch, Force=0 by
construction -- see record_true_zero_dataset.py) to the EXISTING Contact
training data already used by the currently-deployed model (jig Units 1+2
+ the self-closure dataset) -- extending, not replacing, the same fix
methodology contact_classifier_fix/retrain_contact_classifier.py already
established and validated for this exact head.

Held out by COMBO, not by row: true_zero_dataset.csv has ~40-60 highly-
correlated ticks per (combo, finger) (the same combo's readings barely
change tick-to-tick), so a plain random row split would leak near-identical
ticks from the same combo into both train and test, inflating the reported
held-out accuracy. An 80/20 split by combo_index (whole combos held out
entirely) is the honest generalization check.

Critical: reuses the EXISTING deployed scaler from active_model.pkl
UNMODIFIED -- never refit (same reasoning as retrain_contact_classifier.py's
own docstring: refitting would silently miscalibrate the regressor/Phi
classifier being carried over unchanged).

Saves a NEW model file -- does NOT overwrite active_model.pkl (the
currently-deployed/live model, which is itself the 2026-07-17 candidate_v2
promotion). See build_candidate_v3.py for packaging this into a full
self-contained candidate folder (same pattern as candidate_v2).

Usage: uv run python model_v1/true_zero_dataset/retrain_contact_from_true_zero.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

import true_zero_common as tzc

SCRIPT_DIR = tzc.SCRIPT_DIR
MODEL_V1_DIR = SCRIPT_DIR.parent

DEPLOYED_MODEL_PATH = MODEL_V1_DIR / "active_model.pkl"
JIG_TRAIN_CSV = MODEL_V1_DIR / "train_data_backregion.csv"
JIG_TEST_CSV = MODEL_V1_DIR / "test_data_backregion.csv"
SELFCLOSURE_CSV = MODEL_V1_DIR / "contact_classifier_fix" / "selfclosure_contact_dataset.csv"
TRUE_ZERO_CSV = tzc.DATASET_CSV
OUTPUT_MODEL_PATH = SCRIPT_DIR / "active_model_true_zero_fix.pkl"
REPORT_MD = SCRIPT_DIR / "retrain_report.md"

JIG_UNITS = [1, 2]
HELD_OUT_COMBO_FRACTION = 0.2
SPLIT_SEED = 20260717

# Identical architecture/hyperparameters to train_model_forcefix_v2.py's /
# retrain_contact_classifier.py's MLP_KWARGS -- same head, same recipe,
# further-expanded dataset.
MLP_KWARGS = dict(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    random_state=42,
    verbose=True,
)


def balanced_weight(labels: np.ndarray) -> tuple[np.ndarray, dict]:
    n0 = (labels == 0).sum()
    n1 = (labels == 1).sum()
    if n0 == 0 or n1 == 0:
        return np.ones(len(labels)), {0: 1.0, 1: 1.0}
    weight_map = {0: len(labels) / (2 * n0), 1: len(labels) / (2 * n1)}
    return np.array([weight_map[v] for v in labels]), weight_map


def main() -> None:
    lines = ["# True-zero Contact classifier retrain -- before/after report", ""]
    add = lines.append

    if not TRUE_ZERO_CSV.exists():
        raise SystemExit(f"No true-zero training data at {TRUE_ZERO_CSV} -- run "
                          f"record_true_zero_dataset.py then build_true_zero_dataset.py first.")

    deployed = joblib.load(DEPLOYED_MODEL_PATH)
    input_cols = deployed["input_cols"]
    contact_col = deployed["contact_col"]
    contact_force_threshold = deployed["contact_force_threshold"]
    scaler = deployed["scaler"]  # REUSED, never refit
    print(f"Loaded currently-deployed model from {DEPLOYED_MODEL_PATH}")

    # --- Existing jig-only (Units 1+2) Contact training data, unchanged ----
    jig_df = pd.read_csv(JIG_TRAIN_CSV)
    jig_df = jig_df[jig_df["Unit"].isin(JIG_UNITS)].reset_index(drop=True)
    jig_df[contact_col] = (jig_df["Force"] >= contact_force_threshold).astype(int)
    print(f"Jig Contact training rows (Units {JIG_UNITS}): {len(jig_df)}")

    # --- Existing self-closure supplementary data, unchanged ---------------
    sc_df = pd.read_csv(SELFCLOSURE_CSV)
    sc_df[contact_col] = (sc_df["Force"] >= contact_force_threshold).astype(int)
    print(f"Self-closure rows (already in the deployed model): {len(sc_df)}")

    # --- New true-zero data: held out BY COMBO, not by row -----------------
    tz_df = pd.read_csv(TRUE_ZERO_CSV)
    tz_df[contact_col] = (tz_df["Force"] >= contact_force_threshold).astype(int)
    assert (tz_df[contact_col] == 0).all(), "Expected every true-zero row to be Contact=0"

    combo_ids = sorted(tz_df["combo_index"].unique())
    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = rng.permutation(combo_ids)
    n_test = max(1, int(len(combo_ids) * HELD_OUT_COMBO_FRACTION))
    test_combo_ids = set(shuffled[:n_test].tolist())
    train_combo_ids = set(shuffled[n_test:].tolist())

    tz_train = tz_df[tz_df["combo_index"].isin(train_combo_ids)].reset_index(drop=True)
    tz_test = tz_df[tz_df["combo_index"].isin(test_combo_ids)].reset_index(drop=True)
    print(f"True-zero data: {len(combo_ids)} combos -> {len(train_combo_ids)} train "
          f"({len(tz_train)} rows), {len(test_combo_ids)} held-out test ({len(tz_test)} rows)")

    add(f"- Jig (Units 1+2) Contact rows: {len(jig_df)}")
    add(f"- Self-closure rows (already deployed): {len(sc_df)}")
    add(f"- True-zero rows: {len(tz_df)} total, from {len(combo_ids)} combinations")
    add(f"- True-zero TRAIN split: {len(train_combo_ids)} combos, {len(tz_train)} rows")
    add(f"- True-zero HELD-OUT split: {len(test_combo_ids)} combos, {len(tz_test)} rows "
        f"(combo_index: {sorted(test_combo_ids)})")
    add("")

    # --- Combine, transform through the REUSED deployed scaler -------------
    combined = pd.concat([jig_df, sc_df, tz_train], ignore_index=True)
    print(f"\nCombined Contact training set: {len(combined)} rows "
          f"({len(jig_df)} jig + {len(sc_df)} self-closure + {len(tz_train)} true-zero-train)")

    X_scaled = scaler.transform(combined[input_cols].values)
    y = combined[contact_col].values
    sample_weight, weight_map = balanced_weight(y)
    print(f"Contact balance weights: idle x{weight_map[0]:.3f}, contact x{weight_map[1]:.3f}")

    print("\n--- Training Contact classifier: jig + self-closure + true-zero-train, "
          "reused deployed scaler ---")
    new_clf = MLPClassifier(**MLP_KWARGS)
    new_clf.fit(X_scaled, y, sample_weight=sample_weight)
    train_acc = accuracy_score(y, new_clf.predict(X_scaled))
    print(f"  Training-fit Contact accuracy (combined set): {train_acc:.4f}")

    old_clf = deployed["contact_classifier"]

    # ---------------------------------------------------------------- eval (a)
    print("\n=== (a) Regression check: held-out jig test set ===")
    jig_test = pd.read_csv(JIG_TEST_CSV)
    jig_test[contact_col] = (jig_test["Force"] >= contact_force_threshold).astype(int)
    X_jig_test = scaler.transform(jig_test[input_cols].values)
    y_jig_test = jig_test[contact_col].values
    old_jig_acc = accuracy_score(y_jig_test, old_clf.predict(X_jig_test))
    new_jig_acc = accuracy_score(y_jig_test, new_clf.predict(X_jig_test))
    print(f"  Old (currently deployed): {old_jig_acc:.4f}")
    print(f"  New (true-zero fix):      {new_jig_acc:.4f}  (delta: {new_jig_acc - old_jig_acc:+.4f})")

    add("## (a) Regression check: held-out jig test set (test_data_backregion.csv)")
    add("")
    add(f"| Model | Contact accuracy |")
    add(f"|---|---|")
    add(f"| Currently deployed | {old_jig_acc:.4f} |")
    add(f"| True-zero fix | {new_jig_acc:.4f} |")
    add(f"| Delta | {new_jig_acc - old_jig_acc:+.4f} |")
    add("")

    # ---------------------------------------------------------------- eval (b)
    print("\n=== (b) True-zero HELD-OUT combos: does the fix actually work? ===")
    X_tz_test = scaler.transform(tz_test[input_cols].values)
    y_tz_test = tz_test[contact_col].values  # all zeros by construction
    old_tz_pred = old_clf.predict(X_tz_test)
    new_tz_pred = new_clf.predict(X_tz_test)
    old_tz_acc = accuracy_score(y_tz_test, old_tz_pred)
    new_tz_acc = accuracy_score(y_tz_test, new_tz_pred)
    old_fp_rate = (old_tz_pred == 1).mean()
    new_fp_rate = (new_tz_pred == 1).mean()
    print(f"  Old (currently deployed): accuracy={old_tz_acc:.4f}  false-positive rate={old_fp_rate:.4f}")
    print(f"  New (true-zero fix):      accuracy={new_tz_acc:.4f}  false-positive rate={new_fp_rate:.4f}")

    add("## (b) True-zero HELD-OUT combos (never seen during retraining)")
    add("")
    add(f"| Model | Contact accuracy (all should read 0) | False-positive rate |")
    add(f"|---|---|---|")
    add(f"| Currently deployed | {old_tz_acc:.4f} | {old_fp_rate:.4f} |")
    add(f"| True-zero fix | {new_tz_acc:.4f} | {new_fp_rate:.4f} |")
    add("")

    # Per-finger breakdown on held-out true-zero combos
    add("### Per-finger false-positive rate, held-out true-zero combos")
    add("")
    add("| Finger | Old FP rate | New FP rate |")
    add("|---|---|---|")
    for finger in tzc.FINGER_LABELS:
        mask = (tz_test["finger"] == finger).values
        if mask.sum() == 0:
            continue
        old_fp = (old_tz_pred[mask] == 1).mean()
        new_fp = (new_tz_pred[mask] == 1).mean()
        add(f"| {finger} | {old_fp:.4f} | {new_fp:.4f} |")
    add("")

    # ---------------------------------------------------------------- save
    new_bundle = dict(deployed)
    new_bundle["contact_classifier"] = new_clf
    new_bundle["head_training_data"] = dict(deployed["head_training_data"])
    new_bundle["head_training_data"]["contact_classifier"] = (
        "jigonly_units_1_2_PLUS_selfclosure_PLUS_true_zero_500combo_simultaneous_4finger"
    )
    new_bundle["true_zero_fix_note"] = (
        f"Retrained (ONLY this head) adding true_zero_dataset.csv "
        f"({len(tz_train)} training rows from {len(train_combo_ids)} of {len(combo_ids)} "
        f"simultaneous-4-finger combinations, genuinely no-touch, Force=0 by construction; "
        f"see model_v1/true_zero_dataset/README.md) to the existing jig+self-closure Contact "
        f"training data. Held out {len(test_combo_ids)} whole combinations for evaluation -- "
        f"see model_v1/true_zero_dataset/retrain_report.md. Regressor, Phi classifier, and "
        f"scaler are byte-for-byte unchanged from the currently-deployed active_model.pkl."
    )
    joblib.dump(new_bundle, OUTPUT_MODEL_PATH)
    print(f"\nSaved new candidate Contact classifier to {OUTPUT_MODEL_PATH}")
    print("(active_model.pkl -- the currently-deployed/live model -- is UNCHANGED)")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written -> {REPORT_MD}")


if __name__ == "__main__":
    main()
