"""Step 1: builds selfclosure_contact_dataset.csv -- supplementary
Contact=0 (no-touch) training data covering the full 0-100% self-closure
range, to fix the false-contact bug found by model_v1/self_motion/'s
investigation (Middle/Ring/Thumb read a STABLE, sustained false Contact=1
at 100% self-closure with nothing touching; Index shows abnormal ~40Hz
raw-classifier oscillation there instead).

Root cause (see model_v1/self_motion/README.md): the deployed Contact
classifier was trained exclusively on jig data (Units 1+2,
`master_characterisation*.csv`) -- bench presses of a STATIONARY
fingertip against a load cell at various angles. That protocol has no
concept of "this finger's own AmazingHand servo has closed it against
nothing" at all -- a self-closure delta this large, with no touch, was
never a training example. This dataset fills exactly that gap.

Data source: **reuses model_v1/crosstalk/raw/*.json directly (read-only,
unmodified)** rather than recording anything new. That folder's own
protocol (record_crosstalk.py) already IS this exact experiment for an
unrelated purpose: each finger swept through 0/25/50/75/100%, idle (no
touch, confirmed in that folder's own recording purpose text), other 3
fingers held fixed at 0% throughout, ~4s settled capture per position
(1.5s settle before capture starts). model_v1/self_motion/'s own finding
(no early-vs-late decay within a stationary phase) confirms this is a
STEADY-STATE effect, not a transient the crosstalk protocol's settle
window would have missed -- so this data is fully representative of the
condition being fixed. Re-recording identical data would be redundant.

For each of the 20 crosstalk recordings, extracts the MOVING finger's own
sensor group (its OWN readings while sitting at that position, not the
crosstalk effect on other fingers -- an entirely different use of the
same file than model_v1/crosstalk/build_crosstalk_dataset.py makes),
computes delta relative to THAT finger's own 0%-position mean (matching
sensing_gui.py's live baseline convention -- delta relative to a captured-
at-rest reference, the same quantity the live Contact classifier actually
sees), and the 5 per-sensor delta-magnitude features -- in the EXACT
column-naming/formula convention `characterisation_delta_combined.csv`
uses (dS1X..dS5Z, S1_mag..S5_mag = sqrt(dX^2+dY^2+dZ^2)), so this output
can be concatenated directly with `train_data_backregion.csv` for
retraining.

Force is set to 0.0 for every row (a true, faithful ground-truth label --
these are genuinely idle captures, no touch, matching how existing rows'
Force column ultimately drives Contact = Force >= CONTACT_FORCE_THRESHOLD
via the SAME formula, so no special-casing is needed downstream).
Unit=4 is used to mark this data source distinctly from jig Units 1/2/3
(NOT a jig unit -- this is the deployed AmazingHand's own 4 fingertips).
Theta/Phi are unused placeholders (0) -- this data is never fed to the
Force/Theta regressor or Phi classifier, only the Contact classifier
(see retrain_contact_classifier.py, which trains ONLY that head).

Run standalone (no hardware needed -- reads existing files only).
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CROSSTALK_DIR = SCRIPT_DIR.parent / "crosstalk"
if str(CROSSTALK_DIR) not in sys.path:
    sys.path.insert(0, str(CROSSTALK_DIR))

import crosstalk_common as cc  # RAW_DIR, RAW_COLS, FINGER_LABELS, POSITIONS -- read-only reuse

OUTPUT_CSV = SCRIPT_DIR / "selfclosure_contact_dataset.csv"

SELFCLOSURE_UNIT = 4  # NOT a jig unit -- see module docstring

SENSORS = ["S1", "S2", "S3", "S4", "S5"]
FIELDNAMES = [
    "Row Number", "Recording Name", "Time", "Unit", "Theta", "Phi", "Force",
    *[f"d{col}" for col in cc.RAW_COLS],
    *[f"{s}_mag" for s in SENSORS],
]


def frame_channel_values(frame: dict, group_idx: int) -> dict[str, float]:
    values = {}
    for i, sensor in enumerate(frame["groups"][group_idx]["sensors"]):
        n = i + 1
        for axis in "XYZ":
            values[f"S{n}{axis}"] = sensor[axis.lower()]
    return values


def main() -> None:
    files = sorted(cc.RAW_DIR.glob("crosstalk_movingfinger*_pos*.json"))
    if not files:
        raise SystemExit(f"No recordings found in {cc.RAW_DIR} -- expected "
                          f"model_v1/crosstalk/raw/*.json to already exist.")

    # Group by moving finger's group_index -> {position: path}
    by_finger: dict[int, dict[int, Path]] = {}
    for path in files:
        info = json.loads(path.read_text(encoding="utf-8"))["recording_info"]
        gi = info["moving_finger"]["group_index"]
        pos = info["moving_finger"]["position_percent"]
        by_finger.setdefault(gi, {})[pos] = path

    rows: list[dict] = []
    row_number = 0
    for gi in sorted(by_finger):
        positions = by_finger[gi]
        missing = [p for p in cc.POSITIONS if p not in positions]
        if missing:
            raise SystemExit(f"Finger group {gi}: missing positions {missing} in "
                              f"{cc.RAW_DIR} -- expected the full crosstalk sweep.")

        label = json.loads(positions[0].read_text(encoding="utf-8"))["recording_info"]["moving_finger"]["label"]

        # Baseline: this finger's own 0%-position mean -- same reference
        # sensing_gui.py's live pipeline calibrates against at startup
        # (the finger at rest, fully open), so the delta computed below
        # matches exactly what the live Contact classifier is actually
        # fed when a finger sits at some closure position with no touch.
        zero_data = json.loads(positions[0].read_text(encoding="utf-8"))
        zero_frames = zero_data["frames"]
        baseline_series = {col: [] for col in cc.RAW_COLS}
        for frame in zero_frames:
            vals = frame_channel_values(frame, gi)
            for col in cc.RAW_COLS:
                baseline_series[col].append(vals[col])
        baseline = {col: statistics.fmean(v) for col, v in baseline_series.items()}

        for pos in cc.POSITIONS:
            data = json.loads(positions[pos].read_text(encoding="utf-8"))
            frames = data["frames"]
            recording_name = f"selfclosure_finger{gi + 1}_{label.lower()}_pos{pos:03d}"
            for frame in frames:
                row_number += 1
                raw_values = frame_channel_values(frame, gi)
                delta = {col: raw_values[col] - baseline[col] for col in cc.RAW_COLS}

                row = {
                    "Row Number": row_number,
                    "Recording Name": recording_name,
                    "Time": frame["timestamp"],
                    "Unit": SELFCLOSURE_UNIT,
                    "Theta": 0,
                    "Phi": 0,
                    "Force": 0.0,  # ground truth: genuinely idle, no touch
                }
                for col in cc.RAW_COLS:
                    row[f"d{col}"] = delta[col]
                for s in SENSORS:
                    dx, dy, dz = delta[f"{s}X"], delta[f"{s}Y"], delta[f"{s}Z"]
                    row[f"{s}_mag"] = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

                rows.append(row)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows ({len(by_finger)} fingers x {len(cc.POSITIONS)} "
          f"positions) -> {OUTPUT_CSV}")
    for gi in sorted(by_finger):
        n = sum(1 for r in rows if r["Recording Name"].startswith(f"selfclosure_finger{gi + 1}_"))
        print(f"  Finger {gi + 1}: {n} rows")


if __name__ == "__main__":
    main()
