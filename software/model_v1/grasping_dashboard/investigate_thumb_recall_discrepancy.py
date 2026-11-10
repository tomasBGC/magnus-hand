"""Investigates why the naive per-tick threshold sweep (sweep_finger_
thresholds.py) could not reproduce the recall numbers recorded for Thumb's
CONTACT_P_CONFIRM_PER_FINGER override (0.815->0.974) despite landing on the
identical 6 held-out reps. Hypothesis under test: the original figures were
computed against the LIVE, STATEFUL Schmitt-trigger + frame-hysteresis
confirmed_contact signal (sensing_gui.py's _process_one_finger, CONTACT_
CONFIRM_FRAMES=4 / CONTACT_RELEASE_FRAMES=5 + a dead zone + the
MIN_CONTACT_FORCE_N gate), NOT a memoryless per-tick probability>=threshold
comparison -- which is what the sweep script actually measured.

Simulates the EXACT state machine from sensing_gui.py's _process_one_finger
(confirm_signal/release_signal/dead-zone/streak-counters/confirmed_contact),
processed in tick order within each held-out rep (reset per rep), first
WITHOUT the MIN_CONTACT_FORCE_N Force gate (isolating the temporal-hysteresis
effect alone) and then WITH it (using the model's own regressor output for
Force, matching the live pipeline exactly -- NOT the load cell's ground-truth
force, which is a different column used only for the label).

Usage: uv run python model_v1/grasping_dashboard/investigate_thumb_recall_discrepancy.py
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from retrain_contact_with_load_cell_data import (
    ACTIVE_MODEL_PATH, INPUT_COLS, LOAD_CELL_CSV, _split_load_cell_dataset,
)

CONTACT_CONFIRM_FRAMES = 4
CONTACT_RELEASE_FRAMES = 5
MIN_CONTACT_FORCE_N = 0.1
DELTA_COLS = [c for c in INPUT_COLS if c.startswith("d")]


def _simulate_confirmed(probs: np.ndarray, forces: np.ndarray | None,
                         confirm_thr: float, release_thr: float) -> np.ndarray:
    """Byte-for-byte port of sensing_gui.py's _process_one_finger streak
    logic (CONTACT_PROB_SCHMITT_ENABLED=True path), run forward over one
    rep's ticks in order. `forces` is the per-tick predicted Force used for
    the MIN_CONTACT_FORCE_N gate -- pass None to skip that gate entirely
    (isolates the temporal-hysteresis effect on its own)."""
    n = len(probs)
    confirmed = np.zeros(n, dtype=bool)
    consecutive_contact_frames = 0
    consecutive_no_contact_frames = 0
    confirmed_contact = False
    for i in range(n):
        p = probs[i]
        force_ok = True if forces is None else (forces[i] > MIN_CONTACT_FORCE_N)
        confirm_signal = (p >= confirm_thr) and force_ok
        release_signal = p <= release_thr
        if confirm_signal:
            consecutive_contact_frames += 1
            consecutive_no_contact_frames = 0
        elif release_signal:
            consecutive_no_contact_frames += 1
            consecutive_contact_frames = 0
        # else: dead zone, both streaks hold (no change)

        if not confirmed_contact:
            if consecutive_contact_frames >= CONTACT_CONFIRM_FRAMES:
                confirmed_contact = True
        else:
            if consecutive_no_contact_frames >= CONTACT_RELEASE_FRAMES:
                confirmed_contact = False
        confirmed[i] = confirmed_contact
    return confirmed


def main() -> None:
    bundle = joblib.load(ACTIVE_MODEL_PATH)
    scaler = bundle["scaler"]
    clf = bundle["contact_classifier"]
    regressor = bundle["regressor"]
    regression_output_cols = bundle["regression_output_cols"]
    force_idx = regression_output_cols.index("Force")

    load_cell = pd.read_csv(LOAD_CELL_CSV)
    _, holdout = _split_load_cell_dataset(load_cell)
    thumb = holdout[holdout["finger"] == "Thumb"].copy()
    print(f"Thumb holdout: {len(thumb)} ticks, {thumb['source_file'].nunique()} reps "
          f"(same 6 reps as sweep_finger_thresholds.py / the original comment)")

    X = scaler.transform(thumb[INPUT_COLS].values)
    probs_all = clf.predict_proba(X)[:, 1]
    pred_force_all = np.maximum(0.0, regressor.predict(X)[:, force_idx])
    y_true_all = thumb["Contact"].values.astype(bool)

    print(f"\nModel's own predicted Force on these ticks: "
          f"min={pred_force_all.min():.3f} p25={np.percentile(pred_force_all,25):.3f} "
          f"median={np.median(pred_force_all):.3f} max={pred_force_all.max():.3f} "
          f"(MIN_CONTACT_FORCE_N gate = {MIN_CONTACT_FORCE_N})")
    print(f"Fraction of ticks with predicted Force > {MIN_CONTACT_FORCE_N}N: "
          f"{(pred_force_all > MIN_CONTACT_FORCE_N).mean():.3f}")

    for confirm_thr, release_thr, label in [(0.30, 0.15, "0.30/0.15 (current global)"),
                                              (0.10, 0.05, "0.10/0.05 (deployed Thumb override)")]:
        print(f"\n=== threshold {label} ===")
        for tag, use_force_gate in [("no force gate (pure hysteresis)", False),
                                     ("WITH MIN_CONTACT_FORCE_N gate (matches live exactly)", True)]:
            confirmed_all = np.zeros(len(thumb), dtype=bool)
            for source_file, group in thumb.groupby("source_file", sort=False):
                idx = group.sort_values("tick_index").index
                pos = thumb.index.get_indexer(idx)
                probs = probs_all[pos]
                forces = pred_force_all[pos] if use_force_gate else None
                confirmed = _simulate_confirmed(probs, forces, confirm_thr, release_thr)
                confirmed_all[pos] = confirmed

            tp = int((confirmed_all & y_true_all).sum())
            fn = int((~confirmed_all & y_true_all).sum())
            fp = int((confirmed_all & ~y_true_all).sum())
            tn = int((~confirmed_all & ~y_true_all).sum())
            recall = tp / (tp + fn) if (tp + fn) else float("nan")
            precision = tp / (tp + fp) if (tp + fp) else float("nan")
            print(f"  {tag}: recall={recall:.4f} precision={precision:.4f} "
                  f"(TP={tp} FN={fn} FP={fp} TN={tn})")

    print("\n--- reference: naive per-tick (no hysteresis, no force gate), from sweep_finger_thresholds.py ---")
    print("  0.30: recall=0.5692 precision=1.0000")
    print("  0.10: recall=0.7128 precision=1.0000")
    print("\n--- reference: originally-cited numbers (source unverified) ---")
    print("  0.30: recall=0.815")
    print("  0.10: recall=0.974")


if __name__ == "__main__":
    main()
