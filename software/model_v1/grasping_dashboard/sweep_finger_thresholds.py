"""Per-finger Contact-confirm probability threshold sweep for Index,
Middle, and Ring -- same methodology as the (undocumented) sweep that
produced CONTACT_P_CONFIRM_PER_FINGER["Thumb"]=0.10 (sensing_gui.py,
2026-09-07), now run for the other three fingers and, unlike that one,
recorded as an actual reusable script (see per_finger_threshold_sweep_
report.md for the writeup this feeds). Thumb is included too, purely as a
consistency check against the already-adopted 0.10/0.05 numbers.

Ground truth / holdout split matches retrain_contact_with_load_cell_data.py
EXACTLY -- imports its own _split_load_cell_dataset with the same seeded
RNG, so this is the SAME 20%-of-reps, per-finger-stratified holdout that
active_model.pkl's deployed contact_classifier was itself evaluated
against when it was retrained. Zero leakage: the classifier was trained
only on the complementary TRAIN split, never these rows.

Usage: uv run python model_v1/grasping_dashboard/sweep_finger_thresholds.py
"""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from retrain_contact_with_load_cell_data import (
    ACTIVE_MODEL_PATH, INPUT_COLS, LOAD_CELL_CSV, _split_load_cell_dataset,
)

# Same coarse-grained candidate set the Thumb sweep is documented as having
# used (0.30 down to 0.02) -- see the Thumb report for why this is a
# deliberately non-infinitesimal step size (avoids reading single-episode
# noise as a real plateau/cliff, per this project's own established
# threshold-sweep practice elsewhere, e.g. select_thresholds_consolidated).
CANDIDATE_THRESHOLDS = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.02]


def sweep_finger(finger: str, holdout: pd.DataFrame, clf, scaler) -> pd.DataFrame:
    sub = holdout[holdout["finger"] == finger]
    X = scaler.transform(sub[INPUT_COLS].values)
    y_true = sub["Contact"].values
    probs = clf.predict_proba(X)[:, 1]
    rows = []
    for t in CANDIDATE_THRESHOLDS:
        y_pred = (probs >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "n": len(sub),
            "n_pos_true": int(y_true.sum()),
            "n_pos_pred": int(y_pred.sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    bundle = joblib.load(ACTIVE_MODEL_PATH)
    scaler = bundle["scaler"]
    clf = bundle["contact_classifier"]

    load_cell = pd.read_csv(LOAD_CELL_CSV)
    _, holdout = _split_load_cell_dataset(load_cell)
    print(f"Holdout: {len(holdout)} rows, {holdout['source_file'].nunique()} reps "
          f"(same seeded split active_model.pkl's own contact_classifier was evaluated against "
          "when it was retrained -- zero leakage).")

    for finger in ["Index", "Middle", "Ring", "Thumb"]:
        print(f"\n=== {finger} ===")
        table = sweep_finger(finger, holdout, clf, scaler)
        print(table.to_string(index=False, formatters={
            "precision": "{:.4f}".format, "recall": "{:.4f}".format}))


if __name__ == "__main__":
    main()
