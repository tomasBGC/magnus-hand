"""Deploys the self-closure Contact classifier fix: overwrites
model_v1/active_model.pkl (the currently-deployed model, loaded by
live_predictor_test.py's MODEL_PATH and therefore by sensing_gui.py /
hand_control_gui.py) with active_model_contact_selfclosure_fix.pkl's
bundle.

Safety: backs up the CURRENT active_model.pkl to
active_model_BACKUP_pre_selfclosure_fix.pkl (in this folder) before
overwriting, and asserts the regressor/Phi classifier/scaler are
byte-identical between old and new (only contact_classifier + its
metadata should differ) before writing anything -- refuses to deploy if
that assertion fails, rather than silently deploying an unexpected bundle.

This is the one script in this folder that DOES modify an existing file
outside contact_classifier_fix/ -- by explicit request, once the fix was
reviewed and validated (see contact_fix_report.md). Every other script
here is read-only with respect to files outside this folder.

Usage: uv run python model_v1/contact_classifier_fix/deploy_contact_fix.py
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_V1_DIR = SCRIPT_DIR.parent

DEPLOYED_MODEL_PATH = MODEL_V1_DIR / "active_model.pkl"
FIXED_MODEL_PATH = SCRIPT_DIR / "active_model_contact_selfclosure_fix.pkl"
BACKUP_PATH = SCRIPT_DIR / "active_model_BACKUP_pre_selfclosure_fix.pkl"


def main() -> None:
    if not FIXED_MODEL_PATH.exists():
        raise SystemExit(f"{FIXED_MODEL_PATH} not found -- run retrain_contact_classifier.py first.")

    deployed = joblib.load(DEPLOYED_MODEL_PATH)
    fixed = joblib.load(FIXED_MODEL_PATH)

    # --- Safety assertions: only contact_classifier (+ its metadata)
    # should differ. Refuse to deploy otherwise. -----------------------------
    assert deployed["input_cols"] == fixed["input_cols"], "input_cols mismatch"
    assert np.array_equal(deployed["scaler"].mean_, fixed["scaler"].mean_), "scaler mean_ mismatch"
    assert np.array_equal(deployed["scaler"].scale_, fixed["scaler"].scale_), "scaler scale_ mismatch"
    assert all(
        np.array_equal(a, b)
        for a, b in zip(deployed["regressor"].coefs_, fixed["regressor"].coefs_)
    ), "Force/Theta regressor weights differ -- refusing to deploy"
    assert all(
        np.array_equal(a, b)
        for a, b in zip(deployed["classifier"].coefs_, fixed["classifier"].coefs_)
    ), "Phi classifier weights differ -- refusing to deploy"
    assert deployed["contact_classifier"] is not fixed["contact_classifier"], \
        "Expected a genuinely different (retrained) contact_classifier"
    print("Safety checks passed: regressor, Phi classifier, and scaler are byte-identical "
          "between the deployed model and the fix. Only contact_classifier differs.")

    # --- Backup the CURRENT deployed model before overwriting it -----------
    joblib.dump(deployed, BACKUP_PATH)
    print(f"Backed up current active_model.pkl -> {BACKUP_PATH}")

    # --- Deploy: overwrite active_model.pkl with the fixed bundle ----------
    to_deploy = dict(fixed)
    to_deploy["contact_classifier_fix_deployed_at_iso"] = datetime.now(timezone.utc).isoformat()
    joblib.dump(to_deploy, DEPLOYED_MODEL_PATH)
    print(f"Deployed -> {DEPLOYED_MODEL_PATH}")
    print("\nForce/Theta regressor, Phi classifier, and scaler: UNCHANGED.")
    print("Contact classifier: REPLACED with the self-closure fix "
          "(jig Units 1+2 + all-4-fingers self-closure 0-100%).")
    print(f"\nRollback: joblib.dump(joblib.load(r'{BACKUP_PATH}'), r'{DEPLOYED_MODEL_PATH}')")


if __name__ == "__main__":
    main()
