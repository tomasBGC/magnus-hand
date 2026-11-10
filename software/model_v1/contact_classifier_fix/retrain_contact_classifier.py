"""Step 2: retrains ONLY the Contact classifier, adding
selfclosure_contact_dataset.csv (self-closure idle data, Contact=0 at
every position 0-100%) to the existing jig-only (Units 1+2) Contact
training data -- fixing the false-contact bug at full self-closure found
by model_v1/self_motion/, without touching anything else.

Matches the existing pattern of scoping a fix to the specific head that
needs it (see train_model_forcefix_v2.py's own docstring: "Phi classifier
and Contact classifier stayed jig-only... not re-litigating the... Phi
decision already made" when only Force/Theta needed a fix that time).
Here it's the reverse: Force/Theta regressor and Phi classifier are
loaded from the CURRENTLY DEPLOYED model and copied into the new bundle
completely UNCHANGED (not retrained, not even re-scored) -- only the
Contact classifier is refit.

Critical: reuses the EXISTING deployed scaler from active_model.pkl
UNMODIFIED -- never refit. A classifier's decision boundary is calibrated
to whatever coordinate system its training scaler produces (the same
principle train_model_forcefix_v2.py's own docstring states for why Phi/
Contact needed refitting when THAT scaler changed) -- refitting the
scaler here would silently miscalibrate the regressor and Phi classifier
being carried over unchanged, since they'd receive differently-scaled
inputs than they were trained on. Both the existing jig Contact rows and
the new self-closure rows are transformed through this SAME reused
scaler before combining.

Saves a NEW model file -- does NOT overwrite active_model.pkl. The
currently-deployed model stays exactly as it is until this fix is
validated (see evaluate_contact_fix.py) and a human decides to swap
MODEL_PATH.

Usage: uv run python model_v1/contact_classifier_fix/retrain_contact_classifier.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_V1_DIR = SCRIPT_DIR.parent

DEPLOYED_MODEL_PATH = MODEL_V1_DIR / "active_model.pkl"
JIG_TRAIN_CSV = MODEL_V1_DIR / "train_data_backregion.csv"
SELFCLOSURE_CSV = SCRIPT_DIR / "selfclosure_contact_dataset.csv"
OUTPUT_MODEL_PATH = SCRIPT_DIR / "active_model_contact_selfclosure_fix.pkl"

JIG_UNITS = [1, 2]  # unchanged from active_model.pkl's own Contact methodology

# Identical architecture/hyperparameters to train_model_forcefix_v2.py's
# MLP_KWARGS -- same head, same training recipe, just an expanded dataset.
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
    """Identical formula to train_model_forcefix_v2.py's balanced_weight()."""
    n0 = (labels == 0).sum()
    n1 = (labels == 1).sum()
    weight_map = {0: len(labels) / (2 * n0), 1: len(labels) / (2 * n1)}
    return np.array([weight_map[v] for v in labels]), weight_map


def main() -> None:
    deployed = joblib.load(DEPLOYED_MODEL_PATH)
    input_cols = deployed["input_cols"]
    contact_col = deployed["contact_col"]
    contact_force_threshold = deployed["contact_force_threshold"]
    scaler = deployed["scaler"]  # REUSED, never refit -- see module docstring
    print(f"Loaded deployed model from {DEPLOYED_MODEL_PATH}")
    print(f"  input_cols: {len(input_cols)} features, contact_force_threshold={contact_force_threshold}")

    # --- Existing jig-only (Units 1+2) Contact training data, unchanged ----
    jig_df = pd.read_csv(JIG_TRAIN_CSV)
    jig_df = jig_df[jig_df["Unit"].isin(JIG_UNITS)].reset_index(drop=True)
    jig_df[contact_col] = (jig_df["Force"] >= contact_force_threshold).astype(int)
    print(f"\nJig Contact training rows (Units {JIG_UNITS}): {len(jig_df)}  "
          f"(Contact=1: {(jig_df[contact_col] == 1).sum()}, "
          f"Contact=0: {(jig_df[contact_col] == 0).sum()})")

    # --- New self-closure supplementary data, all Contact=0 ----------------
    sc_df = pd.read_csv(SELFCLOSURE_CSV)
    sc_df[contact_col] = (sc_df["Force"] >= contact_force_threshold).astype(int)
    assert (sc_df[contact_col] == 0).all(), "Expected every self-closure row to be Contact=0"
    print(f"Self-closure supplementary rows: {len(sc_df)} (all Contact=0, "
          f"positions 0-100% x 4 fingers, see build_selfclosure_contact_dataset.py)")

    # --- Combine, transform through the REUSED deployed scaler -------------
    combined = pd.concat([jig_df, sc_df], ignore_index=True)
    print(f"\nCombined Contact training set: {len(combined)} rows "
          f"({len(jig_df)} jig + {len(sc_df)} self-closure)")

    X_scaled = scaler.transform(combined[input_cols].values)
    y = combined[contact_col].values
    sample_weight, weight_map = balanced_weight(y)
    print(f"Contact balance weights (combined set): idle x{weight_map[0]:.3f}, "
          f"contact x{weight_map[1]:.3f}")

    print("\n--- Training Contact classifier: jig (Units 1+2) + self-closure "
          "(all 4 fingers, 0-100%), reused deployed scaler ---")
    contact_classifier = MLPClassifier(**MLP_KWARGS)
    contact_classifier.fit(X_scaled, y, sample_weight=sample_weight)
    train_acc = accuracy_score(y, contact_classifier.predict(X_scaled))
    print(f"  Training-fit Contact accuracy (combined set): {train_acc:.4f}")

    # --- Save new bundle: identical to the deployed one except the Contact
    # classifier and its metadata -- everything else (regressor, Phi
    # classifier, scaler) copied through completely unchanged.
    new_bundle = dict(deployed)
    new_bundle["contact_classifier"] = contact_classifier
    new_bundle["head_training_data"] = dict(deployed["head_training_data"])
    new_bundle["head_training_data"]["contact_classifier"] = (
        "jigonly_units_1_2_PLUS_selfclosure_hand_units_4fingers_0to100pct"
    )
    new_bundle["contact_classifier_fix_note"] = (
        "Retrained (ONLY this head) to fix a false-contact bug found by "
        "model_v1/self_motion/: the jig-only-trained classifier read a "
        "stable, sustained Contact=1 at 100% self-closure (Middle/Ring/"
        "Thumb) or abnormal ~40Hz raw oscillation (Index), with nothing "
        "touching the sensor. Added Contact=0 supplementary training rows "
        "spanning 0/25/50/75/100% self-closure on all 4 AmazingHand "
        "fingers (see build_selfclosure_contact_dataset.py, sourced from "
        "model_v1/crosstalk/raw/*.json). Regressor, Phi classifier, and "
        "scaler are byte-for-byte unchanged from active_model.pkl. See "
        "evaluate_contact_fix.py / contact_fix_report.md for validation."
    )

    joblib.dump(new_bundle, OUTPUT_MODEL_PATH)
    print(f"\nSaved new model to {OUTPUT_MODEL_PATH}")
    print(f"(active_model.pkl / the currently-deployed model is UNCHANGED)")


if __name__ == "__main__":
    main()
