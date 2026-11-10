"""Builds a training-ready dataset from the load-cell grasp training protocol
(dashboard_gui.py's Scenario Runner) -- the first Contact/Force training data
in this project with a genuinely independent ground truth (a physically
separate load cell) rather than the model's own self-referential estimate or
a single-finger jig that can't produce multi-finger simultaneous grasps.

Only VERIFIED reps are used (same standing rule as build_genuine_grasp_
dataset.py: an unverified rep's data is not trustworthy enough to train on),
and only HOLD-phase ticks (the stable window once the fingers have stopped
moving, not the noisier approach/release ticks). Only the finger the load
cell was FACING for a given rep gets a labeled row -- the load cell cannot
attribute force to a specific finger, so the co-grasping finger(s) in the
same rep have no independent ground truth and are excluded (see the
episode's own load_cell_orientation field). Every field in this file's
output schema matches train_model_forcefix.py's INPUT_COLS/REGRESSION_OUTPUT_
COLS exactly (dS1X..dS5Z, S1_mag..S5_mag, Force, Contact) so this CSV can be
concatenated with train_data_backregion.csv directly.

Baseline (for the delta features) is estimated per rep, per facing finger,
as the mean of that finger's raw magnetometer channels over the first
BASELINE_TICKS ticks of the rep's OWN approach phase -- before any closing
step has been dispatched, when the finger is still at its post-reset resting
position (the user resets the sensing baseline before every rep in this
protocol, so this is a fresh, clean reference each time, not a stale one).

Usage: uv run python model_v1/grasping_dashboard/build_load_cell_dataset.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent / "grasping_dashboard_data" / "training_protocol"
MASTER_XLSX = TRAINING_DIR / "master_tracker.xlsx"
OUT_CSV = TRAINING_DIR / "load_cell_dataset.csv"

BASELINE_TICKS = 10
CONTACT_FORCE_THRESHOLD = 0.3  # matches train_model_forcefix.py exactly

RAW_COLS = [f"S{i}{axis}" for i in range(1, 6) for axis in "XYZ"]
DELTA_COLS = [f"d{c}" for c in RAW_COLS]
MAGNITUDE_COLS = [f"S{i}_mag" for i in range(1, 6)]


def _load_verified_map() -> dict[str, bool]:
    """{confirmed_jsonl relative path (as stored in the tracker): verified}."""
    wb = load_workbook(MASTER_XLSX, read_only=True, data_only=True)
    ws = wb["Reps"]
    header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    col = {name: i for i, name in enumerate(header)}
    out = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[col["scenario"]] is None:
            continue
        out[row[col["confirmed_jsonl"]]] = bool(row[col["verified"]])
    return out


def _rows_for_rep(fp: Path) -> list[dict]:
    with fp.open(encoding="utf-8") as f:
        ticks = [json.loads(line) for line in f]
    if not ticks:
        return []
    meta = ticks[0]["episode_meta"]
    finger = meta.get("load_cell_orientation")
    if finger not in ("Index", "Middle", "Ring", "Thumb"):
        return []

    approach = [t for t in ticks if t["episode_meta"].get("phase") == "approach"]
    hold = [t for t in ticks if t["episode_meta"].get("phase") == "hold"]
    if not hold or not approach:
        return []

    baseline_ticks = approach[:BASELINE_TICKS]
    baseline = {
        col: sum(t["sensing"][finger]["raw_channels"][col] for t in baseline_ticks) / len(baseline_ticks)
        for col in RAW_COLS
    }

    rows = []
    for i, t in enumerate(hold):
        raw = t["sensing"][finger]["raw_channels"]
        delta = {f"d{col}": raw[col] - baseline[col] for col in RAW_COLS}
        magnitudes = {
            f"S{s}_mag": (
                delta[f"dS{s}X"] ** 2 + delta[f"dS{s}Y"] ** 2 + delta[f"dS{s}Z"] ** 2
            ) ** 0.5
            for s in range(1, 6)
        }
        force = t["load_cell_force"]
        row = {
            **delta, **magnitudes,
            "Force": force,
            "Contact": int(force >= CONTACT_FORCE_THRESHOLD),
            "finger": finger,
            "scenario": meta.get("scenario"),
            "level": "control" if meta.get("is_control") else meta.get("level"),
            "is_control": bool(meta.get("is_control")),
            "source_file": fp.name,
            "tick_index": i,
        }
        rows.append(row)
    return rows


def build() -> pd.DataFrame:
    verified_map = _load_verified_map()
    all_rows: list[dict] = []
    skipped_unverified = 0
    skipped_missing = 0

    for fp in sorted(TRAINING_DIR.glob("confirmed/*/*/*.jsonl")):
        rel = str(fp.relative_to(TRAINING_DIR))
        verified = verified_map.get(rel)
        if verified is None:
            skipped_missing += 1
            continue
        if not verified:
            skipped_unverified += 1
            continue
        all_rows.extend(_rows_for_rep(fp))

    df = pd.DataFrame(all_rows, columns=DELTA_COLS + MAGNITUDE_COLS + [
        "Force", "Contact", "finger", "scenario", "level", "is_control",
        "source_file", "tick_index",
    ])
    print(f"Built {len(df)} rows from {df['source_file'].nunique()} verified reps "
          f"(skipped {skipped_unverified} unverified reps, {skipped_missing} not found in tracker).")
    print("\nRows per finger:")
    print(df["finger"].value_counts())
    print("\nContact balance:")
    print(df["Contact"].value_counts())
    print("\nRows per scenario:")
    print(df["scenario"].value_counts())
    return df


if __name__ == "__main__":
    dataset = build()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUT_CSV, index=False)
    print(f"\nSaved to {OUT_CSV}")
