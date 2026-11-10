"""Build the model_v1_contact_backregion training dataset: baseline-subtracted
(delta) sensor features, combined across all three fingertip units, including
the newly collected backregion supplementary data.

Same delta/magnitude feature computation as build_delta_dataset.py -- this
script only adds a third source (the backregion recordings) and assigns it
Unit 3, so force_direction_model_v1_contact.pkl's original combined dataset
(characterisation_delta_combined.csv) is left untouched and reproducible.

Inputs:
    recordings/master_characterisation.csv    (Unit 1)
    recordings/master_characterisation_2.csv  (Unit 2)
    recordings/master_characterisation_3.csv  (Unit 3, backregion supplement:
        Theta=90 only, Phi in {-90,-45,0,45,90} -- fills in the phi=-45/+45
        angles at theta=90 that neither Unit 1 nor Unit 2 ever tested, and
        adds a third recording set at the theta=90 "back region" that was
        previously the weakest-covered part of the sensing envelope)

Output:
    model_v1/characterisation_delta_combined_backregion.csv

For each recording (grouped by Recording Name):
  - baseline = mean of the 15 raw sensor channels over idle samples (Force < 0.3N,
    falling back to the lowest-force 10% of samples if no sample is below 0.3N)
  - delta = raw - baseline for all 15 channels
  - 5 magnitude features: sqrt(dX^2 + dY^2 + dZ^2) per sensor
  - Force / Theta / Phi labels are kept unchanged
"""

import math
import pandas as pd

SENSORS = ["S1", "S2", "S3", "S4", "S5"]
RAW_COLS = [f"{s}{axis}" for s in SENSORS for axis in ("X", "Y", "Z")]

SOURCES = [
    ("recordings/master_characterisation.csv", 1),
    ("recordings/master_characterisation_2.csv", 2),
    ("recordings/master_characterisation_3.csv", 3),
]

BASELINE_FORCE_THRESHOLD = 0.3


def baseline_for_recording(group: pd.DataFrame) -> pd.Series:
    idle = group[group["Force"] < BASELINE_FORCE_THRESHOLD]
    if idle.empty:
        n = max(1, math.ceil(len(group) * 0.1))
        idle = group.nsmallest(n, "Force")
    return idle[RAW_COLS].mean()


def process_unit(path: str, unit: int) -> pd.DataFrame:
    df = pd.read_csv(path)

    delta_frames = []
    for name, group in df.groupby("Recording Name", sort=False):
        baseline = baseline_for_recording(group)
        delta = group[RAW_COLS] - baseline
        delta.columns = [f"d{c}" for c in RAW_COLS]
        delta_frames.append(delta)
    deltas = pd.concat(delta_frames).sort_index()

    out = pd.DataFrame(index=df.index)
    out["Row Number"] = df["Row Number"]
    out["Recording Name"] = df["Recording Name"]
    out["Time"] = df["Time"]
    out["Unit"] = unit
    out["Theta"] = df["Theta"]
    out["Phi"] = df["Phi"]
    out["Force"] = df["Force"]

    for c in deltas.columns:
        out[c] = deltas[c]

    for s in SENSORS:
        out[f"{s}_mag"] = (
            deltas[f"d{s}X"] ** 2 + deltas[f"d{s}Y"] ** 2 + deltas[f"d{s}Z"] ** 2
        ) ** 0.5

    return out


def main():
    unit_frames = [process_unit(path, unit) for path, unit in SOURCES]
    combined = pd.concat(unit_frames, ignore_index=True)

    out_path = "model_v1/characterisation_delta_combined_backregion.csv"
    combined.to_csv(out_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Total rows: {len(combined)}")
    for unit in (1, 2, 3):
        sub = combined[combined["Unit"] == unit]
        print(f"  Unit {unit}: {len(sub)} rows, {sub['Recording Name'].nunique()} recordings")

    for unit in (1, 2, 3):
        sub = combined[combined["Unit"] == unit]
        thetas = sorted(sub["Theta"].unique())
        phis = sorted(sub["Phi"].unique())
        print(f"Unit {unit} Theta values present: {thetas}")
        print(f"Unit {unit} Phi values present: {phis}")


if __name__ == "__main__":
    main()
