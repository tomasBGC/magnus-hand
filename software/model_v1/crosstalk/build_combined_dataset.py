"""Builds combined_dataset.csv from raw_combined/*.json -- the
additivity-validation set (see record_crosstalk_combined.py).

For each of the 6 combined recordings and each of its 2 affected
(stationary) fingers, computes the mean and within-recording standard
deviation of each of the 15 raw channels, and a delta_from_reference
against a POOLED reference baseline: the average of the 4 original
all-fingers-at-0% recordings already in raw/ (crosstalk_movingfinger*_pos000.json).

Unlike build_crosstalk_dataset.py's delta_from_p0 (which references that
SAME sweep's own P=0 recording, cancelling within-sweep drift),
raw_combined/ has no P=0 recording of its own -- there was no budget in
this small 6-recording validation set for one. Pooling the original
sweeps' 4 existing P=0 recordings is the best available reference, but it
means drift between the original data-collection session and this
combined-validation session (recorded separately, see the timestamps in
each recording_info) is NOT cancelled out the way it is in
crosstalk_dataset.csv. See README.md's "Known limitations" for this.

Run after record_crosstalk_combined.py.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import crosstalk_common as cc

COMBINED_DIR = SCRIPT_DIR / "raw_combined"
COMBINED_CSV = SCRIPT_DIR / "combined_dataset.csv"

FIELDNAMES = [
    "recording_file", "source_labels", "position_percent",
    "affected_label", "affected_group", "channel",
    "mean_value", "std_value", "n_frames",
    "reference_mean", "delta_from_reference",
]


def extract_channel_means(frames: list[dict], group_idx: int) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {col: [] for col in cc.RAW_COLS}
    for frame in frames:
        sensors = frame["groups"][group_idx]["sensors"]
        for i, sensor in enumerate(sensors):
            n = i + 1
            for axis in "XYZ":
                series[f"S{n}{axis}"].append(sensor[axis.lower()])
    return series


def compute_pooled_reference() -> dict[int, dict[str, float]]:
    """Pools all 4 original all-fingers-at-0% recordings in raw/ into one
    reference mean per (mux group index, channel) -- the most robust
    "everyone at rest" baseline available, since raw_combined/ contains no
    P=0 recording of its own (see module docstring)."""
    ref_files = sorted(cc.RAW_DIR.glob("crosstalk_movingfinger*_pos000.json"))
    if not ref_files:
        raise SystemExit(f"No reference (pos000) recordings found in {cc.RAW_DIR} -- "
                          f"run build_crosstalk_dataset.py's prerequisite (record_crosstalk.py) first.")

    pooled: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in ref_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        info = data["recording_info"]
        frames = data["frames"]
        all_fingers = [info["moving_finger"]] + info["other_fingers"]
        for finger in all_fingers:
            gi = finger["group_index"]
            series = extract_channel_means(frames, gi)
            for col in cc.RAW_COLS:
                pooled[gi][col].extend(series[col])

    return {gi: {col: statistics.fmean(vals) for col, vals in channels.items()}
            for gi, channels in pooled.items()}


def main() -> None:
    files = sorted(COMBINED_DIR.glob("crosstalk_combined_*_pos*.json"))
    if not files:
        raise SystemExit(f"No combined recordings found in {COMBINED_DIR} -- "
                          f"run record_crosstalk_combined.py first.")

    reference = compute_pooled_reference()

    rows: list[dict] = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        info = data["recording_info"]
        moving = info["moving_fingers"]
        source_labels = "+".join(f["label"] for f in moving)
        position = moving[0]["position_percent"]
        frames = data["frames"]

        for other in info["other_fingers"]:
            gi = other["group_index"]
            series = extract_channel_means(frames, gi)
            for col in cc.RAW_COLS:
                values = series[col]
                if not values:
                    raise SystemExit(f"{path.name}: no frames for affected finger "
                                      f"{other['label']}, channel {col}")
                mean_v = statistics.fmean(values)
                ref_v = reference[gi][col]
                rows.append({
                    "recording_file": path.name,
                    "source_labels": source_labels,
                    "position_percent": position,
                    "affected_label": other["label"],
                    "affected_group": gi,
                    "channel": col,
                    "mean_value": mean_v,
                    "std_value": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "n_frames": len(values),
                    "reference_mean": ref_v,
                    "delta_from_reference": mean_v - ref_v,
                })

    with open(COMBINED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows from {len(files)} combined recordings "
          f"(6 recordings x 2 affected fingers x {len(cc.RAW_COLS)} channels) -> {COMBINED_CSV}")


if __name__ == "__main__":
    main()
