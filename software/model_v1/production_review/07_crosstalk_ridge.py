"""Production-readiness review, step 7: a fourth crosstalk backend candidate
-- origin-anchored quadratic ridge regression on the directly-measured
4-finger lookup data (corrected target).

Why: comparison_report_corrected.md leaves Middle (43.4 best) and Ring
(35.0 best) poorly served by ALL current backends, and its own coverage
diagnostic found NN-distance does NOT predict error (far/near ratio 0.59x)
-- i.e. sparse coverage is not the bottleneck, so a smooth GLOBAL fit over
all 80 train combos should beat local lookup. The earlier trained-model
attempt (approach (c) in evaluate_approaches.py, gbm_shallow) was only ever
scored against the WRONG target (raw delta_from_reference, self-effect
conflated in); it was never re-evaluated against crosstalk_from_others.

Model shape, chosen for live-deployment properties (not just MAE):
  features per (affected finger, channel): the other 3 fingers' positions
  P1,P2,P3 (scaled /100), their squares, and pairwise products -- 9
  features, ridge with fit_intercept=False. All features are zero at the
  all-open reference position, so prediction there is EXACTLY 0 by
  construction -- the same structural origin guarantee whose absence made
  plain nearest_neighbor regress live (see sensing_gui.py's
  CROSSTALK_CORRECTION_SOURCE comment). Continuous everywhere -- no
  Voronoi-boundary jumps. The interaction terms are what pairwise_additive
  structurally cannot represent.

Scored on the same 80/20 combo split as validate_corrected_nn.py (imported,
seed-matched), against the same corrected target, alongside all three
existing backends. Read-only outside production_review/.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_V1 = SCRIPT_DIR.parent
for p in (MODEL_V1 / "crosstalk_lookup", MODEL_V1 / "crosstalk"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import csv
import json
import numpy as np
from sklearn.linear_model import Ridge

import crosstalk_common as cc
import crosstalk_model as cm
import crosstalk_lookup_common as clc
import nn_lookup_model as nn
from compute_crosstalk_from_others import CORRECTED_CSV
from evaluate_approaches import train_test_split_combos

OUT_MD = SCRIPT_DIR / "07_crosstalk_ridge_report.md"
OUT_JSON = SCRIPT_DIR / "candidate_crosstalk_ridge_v1.json"
POSITION_COL = {"Index": "index_pos", "Middle": "middle_pos", "Ring": "ring_pos", "Thumb": "thumb_pos"}
ALPHAS = [0.03, 0.1, 0.3, 1.0, 3.0]


def poly_features(positions: list[float]) -> list[float]:
    """positions: the 3 OTHER fingers' positions in alphabetical-by-label
    order, scaled to 0..1. 9 features, all zero at the origin."""
    p = [x / 100.0 for x in positions]
    return [p[0], p[1], p[2],
            p[0] ** 2, p[1] ** 2, p[2] ** 2,
            p[0] * p[1], p[0] * p[2], p[1] * p[2]]


with open(CORRECTED_CSV, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

combo_ids = sorted(set(int(r["combo_index"]) for r in rows))
train_ids, test_ids = train_test_split_combos(combo_ids)
train_set, test_set = set(train_ids), set(test_ids)

positions_by_combo: dict[int, dict[str, float]] = {}
actual: dict[tuple[int, str], dict[str, float]] = {}
for r in rows:
    cid = int(r["combo_index"])
    positions_by_combo.setdefault(cid, {lbl: float(r[POSITION_COL[lbl]]) for lbl in clc.FINGER_LABELS})
    actual.setdefault((cid, r["finger_label"]), {})[r["channel"]] = float(r["crosstalk_from_others"])

# Existing backends for comparison
nn_model = nn.build_model(rows, combo_ids=train_set, built_from_label=f"train_{len(train_ids)}")
a_model = cm.load_model(cc.MODEL_JSON)

# ---------------------------------------------------------------- fit ridge
# alpha selected per FINGER (not per channel -- 80 samples is thin enough
# already) by 5-fold CV over train combos.
ridge_model: dict[str, dict] = {"fingers": {}}
rng = np.random.default_rng(20260716)

for finger in clc.FINGER_LABELS:
    others = sorted(l for l in clc.FINGER_LABELS if l != finger)
    Xtr = np.array([poly_features([positions_by_combo[cid][o] for o in others])
                    for cid in train_ids])
    Ytr = np.array([[actual[(cid, finger)][ch] for ch in clc.RAW_COLS] for cid in train_ids])

    # 5-fold CV over combos for alpha
    order = rng.permutation(len(train_ids))
    folds = np.array_split(order, 5)
    best_alpha, best_cv = None, np.inf
    for alpha in ALPHAS:
        errs = []
        for k in range(5):
            val = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            m = Ridge(alpha=alpha, fit_intercept=False)
            m.fit(Xtr[tr], Ytr[tr])
            errs.append(np.abs(m.predict(Xtr[val]) - Ytr[val]).mean())
        cv = float(np.mean(errs))
        if cv < best_cv:
            best_cv, best_alpha = cv, alpha
    m = Ridge(alpha=best_alpha, fit_intercept=False)
    m.fit(Xtr, Ytr)
    ridge_model["fingers"][finger] = {
        "others": others, "alpha": best_alpha, "cv_mae": best_cv,
        "coef": m.coef_.tolist(),  # (15 channels x 9 features)
        "channels": list(clc.RAW_COLS),
    }
    print(f"{finger}: alpha={best_alpha} cv_mae={best_cv:.3f}")


def ridge_predict(finger: str, other_positions: dict[str, float]) -> dict[str, float]:
    entry = ridge_model["fingers"][finger]
    x = np.array(poly_features([other_positions[o] for o in entry["others"]]))
    shifts = np.array(entry["coef"]) @ x
    return {ch: float(s) for ch, s in zip(entry["channels"], shifts)}


# ---------------------------------------------------------------- evaluate
per_finger = {f: {"a": [], "b": [], "c": [], "ridge": []} for f in clc.FINGER_LABELS}
for cid in test_ids:
    positions = positions_by_combo[cid]
    for finger in clc.FINGER_LABELS:
        other_positions = {l: p for l, p in positions.items() if l != finger}
        preds = {
            "a": cm.predict_shift(a_model, finger, other_positions),
            "b": nn.predict_shift(nn_model, finger, other_positions),
            "c": nn.predict_shift_smoothed(nn_model, finger, other_positions),
            "ridge": ridge_predict(finger, other_positions),
        }
        act = actual[(cid, finger)]
        for tag, pred in preds.items():
            per_finger[finger][tag].extend(
                abs(pred.get(ch, 0.0) - act[ch]) for ch in clc.RAW_COLS)

lines = ["# Crosstalk backend candidate: origin-anchored quadratic ridge", ""]
add = lines.append
add(f"Fit on {len(train_ids)} train combos, scored on {len(test_ids)} held-out "
    f"combos against crosstalk_from_others (same split/target as "
    f"comparison_report_corrected.md).")
add("")
add("| Finger | (a) pairwise-additive | (b) NN raw | (c) NN smoothed | (d) ridge quad | ridge alpha |")
add("|---|---|---|---|---|---|")
overall = {t: [] for t in ("a", "b", "c", "ridge")}
for finger in clc.FINGER_LABELS:
    row = per_finger[finger]
    for t in overall:
        overall[t].extend(row[t])
    add(f"| {finger} | {np.mean(row['a']):.3f} | {np.mean(row['b']):.3f} | "
        f"{np.mean(row['c']):.3f} | **{np.mean(row['ridge']):.3f}** | "
        f"{ridge_model['fingers'][finger]['alpha']} |")
add(f"| **Overall** | {np.mean(overall['a']):.3f} | {np.mean(overall['b']):.3f} | "
    f"{np.mean(overall['c']):.3f} | **{np.mean(overall['ridge']):.3f}** | |")
add("")
add("Origin check: prediction at all-others-at-0% is exactly 0 by "
    "construction (all 9 features vanish).")
add("")

# Sanity: origin prediction
for finger in clc.FINGER_LABELS:
    zero = ridge_predict(finger, {l: 0.0 for l in clc.FINGER_LABELS if l != finger})
    assert all(abs(v) < 1e-12 for v in zero.values())
add("Verified numerically for all 4 fingers.")
add("")

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(ridge_model, f)
add(f"Candidate model saved to {OUT_JSON.name} (not wired into sensing_gui.py).")

OUT_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT_MD}")
