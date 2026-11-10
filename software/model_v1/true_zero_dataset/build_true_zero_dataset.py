"""Step 3: flattens raw/*.json (see record_true_zero_dataset.py) into two
outputs:

  - true_zero_summary.csv: one row per (combination, finger) -- mean/max
    displayed Force, max raw Force, confirmed-contact fraction, glitch
    fraction. For the "which combos/patterns show nonzero Force" analysis
    (see analyze_true_zero_results.py).
  - true_zero_dataset.csv: one row per (combination, finger, TICK) -- the
    exact 20-dim input feature vector the live model saw (dS1X..dS5Z,
    S1_mag..S5_mag) plus Force=0.0/Contact=0 ground truth (every tick in
    this whole dataset is genuinely no-touch, by construction of the
    recording protocol -- see record_true_zero_dataset.py's docstring),
    in the exact column convention model_v1's own training pipeline uses,
    so this drops straight into retrain_contact_from_true_zero.py
    unmodified.

Deliberately derives the combo list from combinations_manifest.json +
which raw/*.json files actually exist on disk -- NOT from
raw/session_log.json's own bookkeeping -- so this is robust regardless of
how many separate --limit-ed/--resume batches record_true_zero_dataset.py
was run across (a session that got interrupted and resumed several times,
e.g. after an unexpected process kill, still builds a complete dataset
from whatever combos are actually present on disk). Reports any combos
still missing rather than silently building a partial dataset unnoticed.

Usage:
    python model_v1/true_zero_dataset/build_true_zero_dataset.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import true_zero_common as tzc

RAW_DIR = tzc.RAW_DIR
MANIFEST_JSON = tzc.MANIFEST_JSON
SUMMARY_CSV = RAW_DIR.parent / "true_zero_summary.csv"
TRAINING_CSV = tzc.DATASET_CSV

DELTA_COLS = ['dS1X', 'dS1Y', 'dS1Z', 'dS2X', 'dS2Y', 'dS2Z', 'dS3X', 'dS3Y',
              'dS3Z', 'dS4X', 'dS4Y', 'dS4Z', 'dS5X', 'dS5Y', 'dS5Z']
MAGNITUDE_COLS = ['S1_mag', 'S2_mag', 'S3_mag', 'S4_mag', 'S5_mag']
FEATURE_COLS = DELTA_COLS + MAGNITUDE_COLS  # matches predictor.raw_cols + magnitude order exactly


def raw_path_for(combo: dict) -> Path:
    idx = combo["combo_index"]
    return RAW_DIR / (f"true_zero_combo{idx:04d}_idx{combo['Index']:03.0f}"
                       f"_mid{combo['Middle']:03.0f}_ring{combo['Ring']:03.0f}"
                       f"_thumb{combo['Thumb']:03.0f}.json")


def summarize_ticks(ticks: list) -> dict:
    summary = {}
    for finger in tzc.FINGER_LABELS:
        forces, raw_forces, contacts, glitches = [], [], [], []
        for tick in ticks:
            t = tick.get(finger)
            if t is None:
                continue
            forces.append(t["force"])
            raw_forces.append(t["raw_force"])
            contacts.append(t["contact"])
            glitches.append(t["glitched"])
        n = max(1, len(forces))
        summary[finger] = {
            "mean_force": float(np.mean(forces)) if forces else 0.0,
            "max_force": float(np.max(forces)) if forces else 0.0,
            "max_raw_force": float(np.max(raw_forces)) if raw_forces else 0.0,
            "confirmed_contact_fraction": sum(contacts) / n,
            "glitch_fraction": sum(glitches) / n,
        }
    return summary


def main() -> None:
    if not MANIFEST_JSON.exists():
        raise SystemExit(f"No manifest at {MANIFEST_JSON} -- run sample_combinations.py first.")
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    combinations = manifest["combinations"]

    summary_rows = []
    training_rows = []
    missing = []

    for combo in combinations:
        idx = combo["combo_index"]
        positions = {label: combo[label] for label in tzc.FINGER_LABELS}
        raw_path = raw_path_for(combo)
        if not raw_path.exists():
            missing.append(idx)
            continue

        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        ticks = raw["ticks"]
        summary = summarize_ticks(ticks)

        for finger, stats in summary.items():
            summary_rows.append({
                "combo_index": idx,
                "is_boundary_case": combo["is_boundary_case"],
                "finger": finger,
                "Index_pos": positions["Index"],
                "Middle_pos": positions["Middle"],
                "Ring_pos": positions["Ring"],
                "Thumb_pos": positions["Thumb"],
                **stats,
            })

        for tick in ticks:
            for finger in tzc.FINGER_LABELS:
                t_data = tick.get(finger)
                if t_data is None or "features" not in t_data or t_data["glitched"]:
                    continue  # a glitched tick reused the LAST good result -- not a fresh reading
                row = {
                    "combo_index": idx,
                    "is_boundary_case": combo["is_boundary_case"],
                    "finger": finger,
                    "Index_pos": positions["Index"],
                    "Middle_pos": positions["Middle"],
                    "Ring_pos": positions["Ring"],
                    "Thumb_pos": positions["Thumb"],
                    "t": tick["t"],
                    "Force": 0.0,   # ground truth: genuinely no touch, by construction
                    "Theta": 0.0,   # unused placeholder, matching selfclosure_contact_dataset.csv's convention
                    "Phi": 0,       # unused placeholder
                    "model_displayed_force": t_data["force"],
                    "model_raw_force": t_data["raw_force"],
                    "model_confirmed_contact": int(t_data["contact"]),
                }
                for col, val in zip(FEATURE_COLS, t_data["features"]):
                    row[col] = val
                training_rows.append(row)

    if missing:
        print(f"NOTE: {len(missing)}/{len(combinations)} combinations not yet captured on disk "
              f"(combo_index: {missing[:20]}{'...' if len(missing) > 20 else ''}) -- building "
              f"from the {len(combinations) - len(missing)} that ARE present. Re-run this script "
              f"after collection finishes for the complete dataset.")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    training_df = pd.DataFrame(training_rows)
    training_df.to_csv(TRAINING_CSV, index=False)

    n_combos = summary_df["combo_index"].nunique() if len(summary_df) else 0
    n_nonzero = (summary_df["max_force"] > 0.01).sum() if len(summary_df) else 0
    print(f"\nWrote {SUMMARY_CSV} -- {len(summary_df)} rows ({n_combos} combinations x 4 fingers)")
    print(f"Wrote {TRAINING_CSV} -- {len(training_df)} training rows")
    if len(summary_df):
        print(f"\nCombo-level rows with nonzero max_force (>0.01N) despite no touch: {n_nonzero} "
              f"({n_nonzero / len(summary_df):.1%})")
        print(f"Tick-level rows where the model wrongly confirmed contact: "
              f"{training_df['model_confirmed_contact'].sum()} / {len(training_df)} "
              f"({training_df['model_confirmed_contact'].mean():.1%})")
        print("\nPer-finger nonzero-force rate (combo-level):")
        for finger, g in summary_df.groupby("finger"):
            nz = (g["max_force"] > 0.01).sum()
            add_str = (f"mean max_force where nonzero: "
                       f"{g.loc[g['max_force'] > 0.01, 'max_force'].mean():.3f}N" if nz else "")
            print(f"  {finger:8s} {nz}/{len(g)} ({nz/len(g):.1%})  {add_str}")


if __name__ == "__main__":
    main()
