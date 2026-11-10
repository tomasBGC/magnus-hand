"""Computes Thumb's LIVE confirmed recall/false-positive rate from a
live_validate_thumb_threshold.py session, for direct comparison against the
offline holdout numbers that motivated CONTACT_P_CONFIRM_PER_FINGER["Thumb"]
=0.10 (sensing_gui.py, 2026-09-07): precision 1.000, recall 0.974 at zero
false positives (783 held-out ticks, 6 reps never used in training).

Ground-truth label matches retrain_contact_with_load_cell_data.py's own
convention exactly (Contact = Force >= CONTACT_FORCE_THRESHOLD, 0.3N) so the
live number here is apples-to-apples against the offline one, not a
differently-defined metric that merely looks comparable.

Two populations, scored separately against sensing.Thumb.contact (the live,
Schmitt-gated CONFIRMED decision -- what CONTACT_P_CONFIRM_PER_FINGER
actually controls) and sensing.Thumb.raw_contact (the classifier's
immediate, ungated output -- comparable to the offline script's raw
.predict() figure, i.e. the "53%" number):

  RECALL population: every tick from a NON-control rep with
  load_cell_orientation == "Thumb" (the load cell must be facing Thumb for
  its reading to be Thumb's own ground truth) AND load_cell_force >=
  CONTACT_FORCE_THRESHOLD (a genuine touch actually happened this tick).
  Recall = fraction of those ticks where Thumb's contact flag was True.

  FALSE-POSITIVE population: every tick from a CONTROL rep (load cell
  physically out of reach -- Thumb's true force is zero regardless of what
  the orientation dropdown happened to be left on that rep), PLUS every
  non-control, Thumb-oriented tick with load_cell_force < CONTACT_FORCE_
  THRESHOLD (approach-phase ticks before real contact). False-positive rate
  = fraction of those ticks where Thumb's contact flag was (incorrectly)
  True.

Usage:
    uv run python model_v1/grasping_dashboard/analyze_thumb_threshold_live_validation.py
    (reads every session_*_episode*.jsonl under grasping_dashboard_data/
    live_validation_thumb_threshold/ -- run live_validate_thumb_threshold.py
    and collect at least a few Thumb-facing reps plus some controls first)
"""
from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_V1_DIR = SCRIPT_DIR.parent
VALIDATION_DIR = MODEL_V1_DIR / "grasping_dashboard_data" / "live_validation_thumb_threshold"

CONTACT_FORCE_THRESHOLD = 0.3  # matches retrain_contact_with_load_cell_data.py exactly


def _load_ticks() -> list[dict]:
    ticks = []
    for path in sorted(VALIDATION_DIR.glob("session_*_episode*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ticks.append(json.loads(line))
    return ticks


def _confusion(pop: list[dict], contact_key: str) -> dict:
    tp = fp = tn = fn = 0
    for row in pop:
        true_positive = row["_ground_truth_touch"]
        pred_positive = bool(row["sensing"]["Thumb"][contact_key])
        if true_positive and pred_positive:
            tp += 1
        elif true_positive and not pred_positive:
            fn += 1
        elif not true_positive and pred_positive:
            fp += 1
        else:
            tn += 1
    n = tp + fp + tn + fn
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    fp_rate = fp / (fp + tn) if (fp + tn) else float("nan")
    accuracy = (tp + tn) / n if n else float("nan")
    return {"n": n, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "recall": recall, "precision": precision, "fp_rate": fp_rate, "accuracy": accuracy}


def main() -> None:
    if not VALIDATION_DIR.exists():
        raise SystemExit(
            f"{VALIDATION_DIR} does not exist -- run live_validate_thumb_threshold.py "
            "and collect at least a few Thumb-facing reps (target + control) first.")

    ticks = _load_ticks()
    if not ticks:
        raise SystemExit(f"No session_*_episode*.jsonl files found under {VALIDATION_DIR}.")

    recall_pop, fp_pop = [], []
    for row in ticks:
        meta = row["episode_meta"]
        lcf = row["load_cell_force"]
        is_control = meta["is_control"]
        orientation = meta["load_cell_orientation"]

        if is_control:
            row["_ground_truth_touch"] = False
            fp_pop.append(row)
        elif orientation == "Thumb":
            row["_ground_truth_touch"] = lcf >= CONTACT_FORCE_THRESHOLD
            if row["_ground_truth_touch"]:
                recall_pop.append(row)
            else:
                fp_pop.append(row)
        # non-control, non-Thumb-oriented ticks: no Thumb ground truth available, skip.

    print(f"Loaded {len(ticks)} ticks from {VALIDATION_DIR}")
    print(f"Recall population (genuine Thumb touch, load cell facing Thumb): n={len(recall_pop)}")
    print(f"False-positive population (control reps + Thumb-oriented no-touch ticks): n={len(fp_pop)}")
    if not recall_pop:
        print("\nWARNING: no genuine Thumb-touch ticks found -- collect reps with "
              "'Load cell facing: Thumb' and a non-control target force before analyzing.")
    if not fp_pop:
        print("\nWARNING: no false-positive-eligible ticks found -- collect some control reps too.")

    for label, contact_key in (("CONFIRMED (sensing.Thumb.contact -- the P_CONFIRM=0.10 threshold under test)", "contact"),
                                ("RAW (sensing.Thumb.raw_contact -- ungated classifier output, comparable to the offline 53% figure)", "raw_contact")):
        print(f"\n=== {label} ===")
        if recall_pop:
            r = _confusion(recall_pop, contact_key)
            print(f"  Recall population:  n={r['n']:4d}  recall={r['recall']:.3f}  "
                  f"(TP={r['TP']} FN={r['FN']})")
        if fp_pop:
            f = _confusion(fp_pop, contact_key)
            print(f"  FP population:      n={f['n']:4d}  false-positive rate={f['fp_rate']:.3f}  "
                  f"precision-context TN={f['TN']} FP={f['FP']}")

    print("\nOffline holdout reference (2026-09-07, 783 held-out ticks, P_CONFIRM=0.10): "
          "precision=1.000, recall=0.974, zero false positives.")


if __name__ == "__main__":
    main()
