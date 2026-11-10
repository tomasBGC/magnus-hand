"""Pairwise-additive crosstalk model: fitting, serialisation, prediction.

Model form
----------
For each affected finger F, each source finger S != F, and each raw
channel c, a polynomial in the source's position P is fitted, forced
through the origin (P=0 -> shift=0 by construction, since every channel's
target value is a delta relative to that same sweep's own P=0 recording --
see build_crosstalk_dataset.py). Degree (1 or 2) is chosen per channel by
leave-one-out cross-validation -- see train_crosstalk_model.py.

Additive assumption
--------------------
The total predicted baseline shift for finger F, given ALL 3 other
fingers' positions simultaneously, is the SUM of each pairwise function
evaluated independently:

    shift(F, channel) = sum_{S != F} f_{F,S,channel}(position_S)

This is an ASSUMPTION, not something derived from data: every recording
in raw/ moves exactly one finger at a time (see record_crosstalk.py), so
no real simultaneous-movement data exists to confirm additivity actually
holds. See evaluate_crosstalk_model.py's per-pair effect-size/noise-floor
flags and README.md for where this assumption is (and isn't) likely safe.

This module is a plain data/prediction library -- it is not imported by,
and does not import, anything in the live sensing pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def fit_through_origin(positions: list[float], values: list[float], degree: int) -> list[float]:
    """Least-squares fit of value = c1*P + c2*P^2 + ... + cN*P^N (no
    intercept/constant term -- the fit is structurally forced through the
    origin rather than relying on an observed (0, 0) data point)."""
    p = np.asarray(positions, dtype=float)
    y = np.asarray(values, dtype=float)
    basis = np.vstack([p ** d for d in range(1, degree + 1)]).T
    coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
    return coeffs.tolist()


def predict_polynomial(coeffs: list[float], position: float) -> float:
    return sum(c * position ** (i + 1) for i, c in enumerate(coeffs))


def save_model(model: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)


def load_model(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_shift(model: dict[str, Any], affected_label: str,
                   source_positions: dict[str, float]) -> dict[str, float]:
    """Additive prediction: sums each source finger's pairwise contribution
    for `affected_label`'s 15 raw channels.

    source_positions: {source_label: position_percent, ...}. A source
    finger omitted from this dict is treated as being at the reference
    position (0%), which contributes exactly 0 by construction -- so
    passing only the fingers that actually moved is equivalent to passing
    all of them with the stationary ones at 0.
    """
    pair_data = model["pairs"].get(affected_label, {})
    channels = model["channels"]
    total = {c: 0.0 for c in channels}
    for source_label, position in source_positions.items():
        source_data = pair_data.get(source_label)
        if source_data is None:
            continue
        for c in channels:
            coeffs = source_data["channels"][c]["coeffs"]
            total[c] += predict_polynomial(coeffs, position)
    return total
