"""Origin-anchored quadratic-ridge crosstalk backend -- candidate 4th
backend for sensing_gui.py's CROSSTALK_CORRECTION_BACKENDS.

Implements the exact same contract as crosstalk_model.py and
nn_lookup_model.py:

    load_model(path) -> model dict
    predict_shift(model, affected_label, source_positions) -> {channel: shift}

Model produced by production_review/07_crosstalk_ridge.py
(candidate_crosstalk_ridge_v1.json): per affected finger, a (15 channels x
9 features) coefficient matrix over the OTHER 3 fingers' positions --
features [p1,p2,p3, p1^2,p2^2,p3^2, p1p2,p1p3,p2p3] with p=position/100,
fit_intercept=False. Properties that matter live (see 07's report):

  - Prediction at the all-open reference position is EXACTLY zero by
    construction (every feature vanishes) -- the same structural origin
    guarantee whose absence made plain nearest_neighbor regress live.
  - Continuous everywhere (no Voronoi-boundary jumps).
  - Fitted on all 100 directly-measured simultaneous-4-finger combinations
    (self-effect-corrected target), so no additive-combination assumption.

Validation: 07_crosstalk_ridge_report.md -- held-out MAE best or tied-best
for Index/Middle/Ring across a 5x repeated 5-fold CV; any live adoption
should still be gated on a counterbalanced realistic_session_diagnostic A/B
(offline MAE has mispredicted live behaviour once before in this project).
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "candidate_crosstalk_ridge_v1.json"


def load_model(path=DEFAULT_MODEL_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        model = json.load(f)
    if "fingers" not in model:
        raise ValueError(f"{path} does not look like a ridge crosstalk model")
    return model


def _features(p1: float, p2: float, p3: float) -> list[float]:
    a, b, c = p1 / 100.0, p2 / 100.0, p3 / 100.0
    return [a, b, c, a * a, b * b, c * c, a * b, a * c, b * c]


def predict_shift(model: dict, affected_label: str,
                  source_positions: dict[str, float]) -> dict[str, float]:
    """Predicted raw-channel shifts on `affected_label` caused by the OTHER
    fingers sitting at `source_positions` (0=open..100=closed). Missing
    source fingers default to 0% (the reference position, zero
    contribution), matching the other backends' behaviour."""
    entry = model["fingers"][affected_label]
    others = entry["others"]  # alphabetical, fixed at fit time
    x = _features(*(float(source_positions.get(o, 0.0)) for o in others))
    shifts = {}
    for ch, coef_row in zip(entry["channels"], entry["coef"]):
        shifts[ch] = float(sum(c * xi for c, xi in zip(coef_row, x)))
    return shifts


if __name__ == "__main__":
    m = load_model()
    for finger in m["fingers"]:
        zero = predict_shift(m, finger, {})
        assert all(abs(v) < 1e-12 for v in zero.values()), finger
        some = predict_shift(m, finger, {lbl: 50.0 for lbl in m["fingers"] if lbl != finger})
        mx = max(abs(v) for v in some.values())
        print(f"{finger}: origin=0 OK, max |shift| at all-50% = {mx:.2f}")
