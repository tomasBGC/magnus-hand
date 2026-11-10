"""Step 3: builds crosstalk_lookup_dataset.csv from the raw/ captures
produced by record_crosstalk_lookup.py -- one reference (all fingers 0%)
recording plus up to 100 combination recordings, each with all 4 fingers'
raw magnetic frames captured simultaneously.

For each combination and EACH of the 4 fingers (not just "the other 3" --
unlike model_v1/crosstalk/build_crosstalk_dataset.py, this is a direct
measurement of every finger's own baseline at that pose, not a crosstalk-
from-others delta), computes the mean/std of each of the 15 raw S1X..S5Z
channels over the captured frames, plus delta_from_reference: that same
value minus the single reference (all-0%) recording's mean for the same
finger/channel. delta_from_reference is the "true baseline shift" target
quantity -- directly comparable to model_v1/crosstalk/crosstalk_model.py's
predict_shift() output, which also predicts a shift from the 0% reference
(see crosstalk_lookup_common.py's module docstring).

Run after record_crosstalk_lookup.py. Output: crosstalk_lookup_dataset.csv,
one row per (combination, finger, channel) -- up to 100 x 4 x 15 = 6000 rows.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

_CROSSTALK_DIR = SCRIPT_DIR.parent / "crosstalk"
if str(_CROSSTALK_DIR) not in sys.path:
    sys.path.insert(0, str(_CROSSTALK_DIR))

from MagneticSensors.eflesh_controller import NUM_GROUPS

import crosstalk_lookup_common as clc

FIELDNAMES = [
    "combo_index", "index_pos", "middle_pos", "ring_pos", "thumb_pos",
    "finger_label", "finger_group", "channel", "mean_value", "std_value",
    "n_frames", "delta_from_reference",
]

GROUP_INDEX_BY_LABEL = {label: i for i, label in enumerate(clc.FINGER_LABELS)}


def extract_channel_means(frames: list[dict], group_idx: int) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {col: [] for col in clc.RAW_COLS}
    for frame in frames:
        sensors = frame["groups"][group_idx]["sensors"]
        for i, sensor in enumerate(sensors):
            n = i + 1
            for axis in "XYZ":
                series[f"S{n}{axis}"].append(sensor[axis.lower()])
    return series


def load_recording(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    raw_dir = clc.RAW_DIR
    combo_files = sorted(
        raw_dir.glob("crosstalk_lookup_combo*.json"),
        key=lambda p: int(re.search(r"combo(\d+)_", p.name).group(1)),
    )
    ref_files = list(raw_dir.glob("crosstalk_lookup_reference_*.json"))
    if not ref_files:
        raise SystemExit(f"No reference recording found in {raw_dir} -- run "
                          f"record_crosstalk_lookup.py first (it captures the reference "
                          f"pose before the first combination).")
    if not combo_files:
        raise SystemExit(f"No combination recordings found in {raw_dir} -- run "
                          f"record_crosstalk_lookup.py first.")

    ref_data = load_recording(ref_files[0])
    ref_frames = ref_data["frames"]
    ref_means: dict[str, dict[str, float]] = {}
    for label in clc.FINGER_LABELS:
        series = extract_channel_means(ref_frames, GROUP_INDEX_BY_LABEL[label])
        ref_means[label] = {
            col: statistics.fmean(values) for col, values in series.items() if values
        }
        missing = [col for col in clc.RAW_COLS if col not in ref_means[label]]
        if missing:
            raise SystemExit(f"Reference recording missing frames for {label} channels {missing}")

    rows: list[dict] = []
    skipped_incomplete: list[str] = []
    for path in combo_files:
        data = load_recording(path)
        info = data["recording_info"]
        combo_index = info["combo_index"]
        positions = info["positions"]
        frames = data["frames"]
        if len(frames) == 0:
            skipped_incomplete.append(path.name)
            continue

        for label in clc.FINGER_LABELS:
            group_idx = GROUP_INDEX_BY_LABEL[label]
            series = extract_channel_means(frames, group_idx)
            for col in clc.RAW_COLS:
                values = series[col]
                if not values:
                    raise SystemExit(f"{path.name}: no frames for {label}, channel {col}")
                mean_value = statistics.fmean(values)
                rows.append({
                    "combo_index": combo_index,
                    "index_pos": positions["Index"],
                    "middle_pos": positions["Middle"],
                    "ring_pos": positions["Ring"],
                    "thumb_pos": positions["Thumb"],
                    "finger_label": label,
                    "finger_group": group_idx,
                    "channel": col,
                    "mean_value": mean_value,
                    "std_value": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "n_frames": len(values),
                    "delta_from_reference": mean_value - ref_means[label][col],
                })

    with open(clc.DATASET_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    n_combos = len(set(r["combo_index"] for r in rows))
    print(f"Reference recording: {ref_files[0].name} ({len(ref_frames)} frames)")
    print(f"Wrote {len(rows)} rows from {n_combos} combinations "
          f"({n_combos} x {len(clc.FINGER_LABELS)} fingers x {len(clc.RAW_COLS)} channels) "
          f"-> {clc.DATASET_CSV}")
    if skipped_incomplete:
        print(f"Skipped {len(skipped_incomplete)} recording(s) with 0 captured frames: "
              f"{skipped_incomplete}")


if __name__ == "__main__":
    main()
