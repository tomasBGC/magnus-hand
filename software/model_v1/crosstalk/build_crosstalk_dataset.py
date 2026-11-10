"""Step 1: builds crosstalk_dataset.csv from the raw idle recordings in raw/.

For each of the 20 recordings, extracts which finger moved (source), its
commanded position, and -- for each of the OTHER 3 (stationary, idle,
untouched) fingers -- the mean and within-recording standard deviation of
each of the 15 raw S1X..S5Z channels over that recording's captured
frames. Also computes, per (source finger, affected finger, channel), the
delta from that SAME sweep's own position=0 recording (i.e. all 4 fingers
at the reference/open position) -- this is the "baseline shift" the
crosstalk model (train_crosstalk_model.py) is fitted against.

Run once after record_crosstalk.py has produced the 20 files in raw/.
Output: crosstalk_dataset.csv, one row per
(recording, affected finger, channel) -- 20 x 3 x 15 = 900 rows.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import crosstalk_common as cc

FIELDNAMES = [
    "recording_file", "source_label", "source_group", "position_percent",
    "affected_label", "affected_group", "channel", "mean_value", "std_value",
    "n_frames", "delta_from_p0",
]


def extract_channel_means(frames: list[dict], group_idx: int) -> dict[str, list[float]]:
    """Returns {channel: [values across every captured frame]} for one mux group."""
    series: dict[str, list[float]] = {col: [] for col in cc.RAW_COLS}
    for frame in frames:
        sensors = frame["groups"][group_idx]["sensors"]
        for i, sensor in enumerate(sensors):
            n = i + 1
            for axis in "XYZ":
                series[f"S{n}{axis}"].append(sensor[axis.lower()])
    return series


def main() -> None:
    files = sorted(cc.RAW_DIR.glob("crosstalk_movingfinger*_pos*.json"))
    if not files:
        raise SystemExit(f"No recordings found in {cc.RAW_DIR} -- run record_crosstalk.py first.")

    rows: list[dict] = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        info = data["recording_info"]
        source = info["moving_finger"]
        frames = data["frames"]

        for other in info["other_fingers"]:
            series = extract_channel_means(frames, other["group_index"])
            for col in cc.RAW_COLS:
                values = series[col]
                if not values:
                    raise SystemExit(f"{path.name}: no frames for affected finger "
                                      f"{other['label']}, channel {col}")
                rows.append({
                    "recording_file": path.name,
                    "source_label": source["label"],
                    "source_group": source["group_index"],
                    "position_percent": source["position_percent"],
                    "affected_label": other["label"],
                    "affected_group": other["group_index"],
                    "channel": col,
                    "mean_value": statistics.fmean(values),
                    "std_value": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "n_frames": len(values),
                })

    # delta_from_p0: for each (source, affected, channel), subtract that
    # SAME sweep's position=0 mean (all 4 fingers at reference/open) --
    # not a cross-sweep reference, so within-sweep drift over the ~2 min
    # this script ran cancels out rather than leaking into the delta.
    by_key: dict[tuple[str, str, str], dict[int, dict]] = {}
    for r in rows:
        key = (r["source_label"], r["affected_label"], r["channel"])
        by_key.setdefault(key, {})[r["position_percent"]] = r

    for r in rows:
        key = (r["source_label"], r["affected_label"], r["channel"])
        ref = by_key[key].get(0)
        if ref is None:
            raise SystemExit(f"Missing position=0 reference recording for {key}")
        r["delta_from_p0"] = r["mean_value"] - ref["mean_value"]

    with open(cc.DATASET_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows from {len(files)} recordings "
          f"({len(cc.FINGER_LABELS)} fingers x {len(cc.POSITIONS)} positions x "
          f"3 affected fingers x {len(cc.RAW_COLS)} channels) -> {cc.DATASET_CSV}")


if __name__ == "__main__":
    main()
