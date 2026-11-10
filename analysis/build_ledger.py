"""Builds RESULTS_LEDGER.md: one row per numeric cell in analysis/results/*.csv,
plus one row per documented caveat. Reads the CSVs only; run after the a-e scripts."""
import math
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
_OUT = os.environ.get("EFLESH_OUT_DIR")  # optional: read results from / write ledger to another folder
RESULTS = Path(_OUT) / "results" if _OUT else HERE / "results"
DATA = "data/load_cell_dataset.csv"

SCRIPT = {"a_contact_accuracy.csv": "a_contact_accuracy.py", "b_force_error.csv": "b_force_error.py",
          "c_level_confusion.csv": "c_level_confusion.py", "d_control_bias.csv": "d_control_bias.py",
          "e_level_stats.csv": "e_level_stats.py"}
# columns that identify a row (everything else numeric is a reported value)
KEYS = {"a_contact_accuracy.csv": ["finger"],
        "b_force_error.csv": ["split", "fold", "subset", "finger"],
        "c_level_confusion.csv": ["split", "finger", "true_level"],
        "d_control_bias.csv": ["split", "finger"],
        "e_level_stats.csv": ["source", "finger", "level"]}
IGNORE = {"boundaries"}


def split_of(fname, key):
    s = key.get("split") or key.get("source", "")
    if fname == "a_contact_accuracy.csv":
        return "standard rep-grouped 80/20 holdout"
    if "loso" in s:
        return "leave-one-scenario-out (7 folds)"
    if "full_dataset" in s:
        return "none (full dataset, true force)"
    return "standard rep-grouped 80/20 holdout"


def tuned(fname):
    if fname == "a_contact_accuracy.csv":
        return "n/a - no thresholds fitted here (deployed classifier, default predict())"
    if fname == "c_level_confusion.csv":
        return "yes: level boundaries from TRAIN data only (per fold for LOSO)"
    if fname == "d_control_bias.csv":
        return "n/a - fixed 0.3 N contact threshold from active_model.pkl, not tuned"
    return "n/a - nothing tuned; hyperparameters copied from deployed head"


def status(fname, key, col):
    # unvalidated: cells resting on missing or near-empty holdout data
    if key.get("split") == "standard" or key.get("source") == "pred_standard_holdout":
        if key.get("finger") == "Ring" and (key.get("level") == "firm" or key.get("true_level") == "firm"):
            return "unvalidated"
        if key.get("finger") == "Ring" and key.get("true_level") == "firm":
            return "unvalidated"
    if fname == "b_force_error.csv" and key["split"] == "loso" and key["subset"] == "touch_only" \
            and col in ("mae", "rmse"):
        return "unvalidated"  # no better than constant baseline; see caveat C3
    return "offline-only"


rows = []
for fname, script in SCRIPT.items():
    df = pd.read_csv(RESULTS / fname)
    for _, r in df.iterrows():
        key = {k: r[k] for k in KEYS[fname]}
        for col in df.columns:
            if col in KEYS[fname] or col in IGNORE:
                continue
            v = r[col]
            if not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)):
                continue
            rows.append((fname, " / ".join(str(x) for x in key.values()), col, v, script,
                         DATA, split_of(fname, key), "42", tuned(fname), status(fname, key, col)))

caveats = [
    ("C1", "Thumb has NO no-touch (control) rows in the standard holdout (n=0), so Thumb standard-split "
           "no-touch bias is undefined and Thumb standard-split contact accuracy (0.529) is touch-only.",
     "d_control_bias.csv; e_level_stats.csv; a_contact_accuracy.csv", "unvalidated"),
    ("C2", "Ring has NO firm rows in the standard holdout, so Ring standard-split level accuracy (0.530) and "
           "its firm row are not meaningful.", "c_level_confusion.csv; e_level_stats.csv", "unvalidated"),
    ("C3", "Touch-only leave-one-scenario-out force error (MAE 0.580 N) is no better than the "
           "constant-mean baseline (0.573 N).", "b_force_error.csv", "unvalidated"),
    ("C4", "Pooled leave-one-scenario-out MAE (0.657 N) beats baseline (0.720 N), but touch-only does not (C3), so the margin "
           "comes from the no-touch rows; Index and Middle LOSO correlation is 0.13 / 0.36.", "b_force_error.csv", "offline-only"),
    ("C5", "MLP early stopping uses sklearn's internal random 10% validation split of the training rows "
           "(row-level, not rep-grouped), so it sees near-duplicate ticks from the same reps. Holdout is unaffected.",
     "all", "offline-only"),
    ("C6", "The deployed contact classifier was itself retrained with this dataset's TRAIN split "
           "(active_model.pkl head_training_data). Result (a) is a fair holdout only because the split is "
           "reproduced exactly (same seed and order); it is not an independent dataset.",
     "a_contact_accuracy.csv", "offline-only"),
    ("C7", "Single seed (42), single split; no confidence intervals. Holdout is ~20% of reps per finger "
           "(Index 9, Middle 7, Ring 6, Thumb 6 reps), so per-finger cells are high-variance.", "all", "offline-only"),
    ("C8", "No result here has been live-validated on hardware.", "all", "offline-only"),
]

out = ["# Results ledger", "",
       "Generated by `build_ledger.py` from `results/*.csv`. Status: **validated** = checked on hardware/independent "
       "data; **offline-only** = held-out data from the same dataset; **unvalidated** = rests on missing/near-empty "
       "holdout cells or fails its baseline. No row is currently `validated`.", "",
       "## Caveats", "", "| ID | Caveat | Affects | Status |", "|---|---|---|---|"]
out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in caveats]
out += ["", f"## Numbers ({len(rows)} rows)", "",
        "| File | Cell | Metric | Value | Script | Input | Split | Seed | Tuned on train only? | Status |",
        "|---|---|---|---|---|---|---|---|---|---|"]
out += [f"| {f} | {k} | {m} | {v:.6g} | {s} | `{i}` | {sp} | {sd} | {t} | {st} |"
        for f, k, m, v, s, i, sp, sd, t, st in rows]
((Path(_OUT) if _OUT else HERE) / "RESULTS_LEDGER.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"wrote RESULTS_LEDGER.md: {len(rows)} number rows, {len(caveats)} caveat rows")
