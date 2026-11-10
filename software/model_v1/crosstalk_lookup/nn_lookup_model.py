"""Nearest-neighbour crosstalk-correction model -- no fitting, just a
distance search against directly-measured data (approach (b) from the
crosstalk_lookup 3-way comparison, now self-effect-corrected -- see
self_effect.py's module docstring for why the correction target is
`crosstalk_from_others`, not the raw `delta_from_reference`).

Deliberately mirrors model_v1/crosstalk/crosstalk_model.py's public API
(load_model() / predict_shift(model, affected_label, source_positions))
so sensing_gui.py can dispatch to either module through the SAME call
site, just swapping which model object it holds -- see that module's
CROSSTALK_CORRECTION_SOURCE toggle.

Per-(affected finger) table, NOT a single joint 4D table: each affected
finger's query key is the OTHER 3 fingers' positions only (never its own
-- the self-effect-corrected target is specifically "shift caused by
others", so its own position is not a meaningful part of the query, and
excluding it matches crosstalk_model.predict_shift()'s existing
source_positions contract exactly, where the affected finger's own label
is never present in source_positions).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import crosstalk_lookup_common as clc

MODEL_JSON = SCRIPT_DIR / "nn_lookup_model_v1.json"


def build_model(rows: list[dict], combo_ids: set[int] | None = None,
                 built_from_label: str = "all_100", add_origin_anchor: bool = True) -> dict:
    """rows: crosstalk_lookup_dataset_corrected.csv's rows (as dicts, one
    per (combo, finger, channel) -- see compute_crosstalk_from_others.py).
    combo_ids: restrict to these combo_index values only (e.g. an 80-combo
    train split for validation); None uses every combo present in `rows`
    (the deployed live model uses all 100, per integration decision).

    add_origin_anchor (2026-07, live diagnostic finding): crosstalk_model.py's
    pairwise-additive polynomials are FITTED WITH NO INTERCEPT TERM, forced
    through the origin BY CONSTRUCTION -- at the reference position (all
    OTHER fingers at 0%), predict_shift() returns exactly 0.0 for every
    channel, always, regardless of what data it was fitted on. This lookup
    table has NO such structural guarantee: the 100 sampled combinations
    were drawn broadly across the pose space and essentially never land
    exactly on (0%, 0%, 0%) (continuous LHS sampling, then rounded), so the
    "nearest" stored point to an exact-reference query can sit a genuine
    distance away with a non-trivial measured shift of its own -- live
    testing found this cost real accuracy: at Index's exact reference query,
    the nearest stored point was 8.9% away and predicted a -37 raw-unit
    shift on S3Z where the correct answer is exactly 0, and this was
    confirmed as the dominant cause of new false-contact episodes appearing
    at every reference-adjacent transition in realistic_session_diagnostic.py's
    counterbalanced A/B result (4 of 5 flagged sequence steps queried the
    exact origin). Since `crosstalk_from_others` is trivially/definitionally
    0.0 at the reference position (every combination's target was ITSELF
    measured as a delta from that exact reference recording -- see
    compute_crosstalk_from_others.py), this isn't new data or an assumption,
    it's the same analytically-certain fact the additive model already bakes
    into its own fit; the anchor just makes the lookup table encode it too.
    Added unconditionally (both to train-only validation tables and the
    all-100 deployed table) as a synthetic point (positions all 0%,
    combo_index=-1, shift 0.0 for every channel) so an exact/near-reference
    query resolves to it (distance 0, guaranteed nearest) instead of
    whatever real measured combination happens to be geometrically closest.
    """
    position_col = {"Index": "index_pos", "Middle": "middle_pos",
                     "Ring": "ring_pos", "Thumb": "thumb_pos"}
    affected: dict[str, dict] = {}
    for finger in clc.FINGER_LABELS:
        other_labels = [lbl for lbl in clc.FINGER_LABELS if lbl != finger]
        by_combo: dict[int, dict] = {}
        for row in rows:
            if row["finger_label"] != finger:
                continue
            combo_id = int(row["combo_index"])
            if combo_ids is not None and combo_id not in combo_ids:
                continue
            entry = by_combo.setdefault(combo_id, {
                "combo_index": combo_id,
                "positions": [float(row[position_col[lbl]]) for lbl in other_labels],
                "shift": {},
            })
            entry["shift"][row["channel"]] = float(row["crosstalk_from_others"])
        points = sorted(by_combo.values(), key=lambda p: p["combo_index"])
        if add_origin_anchor:
            points.append({
                "combo_index": -1,
                "positions": [0.0 for _ in other_labels],
                "shift": {col: 0.0 for col in clc.RAW_COLS},
                "is_origin_anchor": True,
            })
        affected[finger] = {"other_labels": other_labels, "points": points}
    return {
        "meta": {
            "built_from": built_from_label,
            "origin_anchor_added": add_origin_anchor,
            "n_points_per_finger": {f: len(affected[f]["points"]) for f in clc.FINGER_LABELS},
            "target": "crosstalk_from_others (self-effect-corrected, see self_effect.py)",
        },
        "affected": affected,
    }


def save_model(model: dict, path: Path = MODEL_JSON) -> None:
    path.write_text(json.dumps(model, indent=2), encoding="utf-8")


def load_model(path: Path = MODEL_JSON) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _distances(model: dict, affected_label: str, source_positions: dict[str, float]):
    """Returns [(point, distance), ...] for every stored point, unsorted."""
    entry = model["affected"][affected_label]
    other_labels = entry["other_labels"]
    query = [source_positions[lbl] for lbl in other_labels]
    return [
        (point, sum((q - p) ** 2 for q, p in zip(query, point["positions"])) ** 0.5)
        for point in entry["points"]
    ]


def _nearest(model: dict, affected_label: str, source_positions: dict[str, float]):
    best_point, best_dist = None, float("inf")
    for point, dist in _distances(model, affected_label, source_positions):
        if dist < best_dist:
            best_dist, best_point = dist, point
    return best_point, best_dist


def predict_shift(model: dict, affected_label: str,
                   source_positions: dict[str, float]) -> dict[str, float]:
    """Same signature/contract as crosstalk_model.predict_shift(): returns
    {channel: predicted_shift, ...} for `affected_label`'s raw channels,
    given the other fingers' CURRENT positions. Missing source fingers
    (not expected in live use -- sensing_gui.py always passes all 3
    others) fall back to 0% since that's the pipeline's own reference
    position, mirroring crosstalk_model.predict_shift()'s same fallback
    for an omitted source finger.

    Pure single-nearest-neighbour -- kept available (not replaced by
    predict_shift_smoothed below) as the direct A/B comparison point; see
    that function's docstring for why live testing moved the deployed
    default to the smoothed variant."""
    entry = model["affected"][affected_label]
    filled = {lbl: source_positions.get(lbl, 0.0) for lbl in entry["other_labels"]}
    point, _dist = _nearest(model, affected_label, filled)
    if point is None:
        return {}
    return dict(point["shift"])


# Number of nearest points blended together -- small enough to stay a
# genuinely LOCAL estimate (not smoothed all the way to a global average,
# which would throw away the whole point of a directly-measured lookup),
# large enough that crossing from "closest to point A" to "closest to point
# B" changes the blend gradually (A's weight fades out, B's fades in) rather
# than jumping straight from A's value to B's the instant the query crosses
# their Voronoi boundary -- exactly the discontinuity a pure 1-NN lookup has
# by construction and CROSSTALK_POSITION_DEADBAND_PCT was never able to fix
# for THIS model (that constant only ever smoothed the query's own position
# estimate, not the lookup function itself -- see sensing_gui.py's (reverted)
# comment on it).
SMOOTHING_K = 4

# Inverse-distance-squared weighting (a standard, simple choice -- steeper
# than plain inverse-distance, so nearby points still dominate the blend
# rather than being diluted too much by the k-1 farther ones).
SMOOTHING_POWER = 2.0

# Distances below this (percent) are treated as an exact match (weight
# effectively infinite) -- avoids a divide-by-near-zero blowing the
# weighting up numerically right at/very near a stored point (including the
# origin anchor above), where the answer should just BE that point's value,
# not some enormous-but-technically-finite weight swamping the others.
SMOOTHING_EXACT_MATCH_EPS = 1e-6


def predict_shift_smoothed(model: dict, affected_label: str, source_positions: dict[str, float],
                            k: int = SMOOTHING_K, power: float = SMOOTHING_POWER) -> dict[str, float]:
    """Distance-weighted k-nearest-neighbour blend (inverse-distance**power
    weighting, a.k.a. Shepard's method) instead of predict_shift()'s hard
    single-nearest jump. Same (model, affected_label, source_positions) ->
    {channel: shift} contract, so it's a drop-in alternative backend (see
    sensing_gui.py's CROSSTALK_CORRECTION_BACKENDS).

    Why: predict_shift() is a genuinely discontinuous function of the query
    position -- as the query crosses from being closest-to-point-A to
    closest-to-point-B, the returned correction jumps straight from A's
    measured value to B's, with no transition. realistic_session_diagnostic.py's
    counterbalanced live A/B (5+5 reps, order-balanced) found this
    corresponded to a real, consistent increase in Index false-contact
    episodes (5.00+/-0.00 vs pairwise_additive's 3.80+/-1.33, pooled across
    10 reps each) even though predict_shift() has BETTER steady-state
    accuracy than the polynomial model (corrected-target MAE 19.8 vs 35.8) --
    the live regression was a discontinuity/noise problem, not an accuracy
    problem. Blending several nearby points with smoothly-varying weights
    removes the hard jump at cell boundaries while still recovering
    (numerically, via SMOOTHING_EXACT_MATCH_EPS) predict_shift()'s exact
    value at/very near any stored point, including the origin anchor --
    i.e. still gets the accuracy win where the data is dense, degrades
    gracefully rather than discontinuously where it isn't.
    """
    entry = model["affected"][affected_label]
    filled = {lbl: source_positions.get(lbl, 0.0) for lbl in entry["other_labels"]}
    distances = _distances(model, affected_label, filled)
    if not distances:
        return {}

    for point, dist in distances:
        if dist <= SMOOTHING_EXACT_MATCH_EPS:
            return dict(point["shift"])

    nearest_k = sorted(distances, key=lambda pd: pd[1])[:k]
    weights = [1.0 / (dist ** power) for _point, dist in nearest_k]
    total_weight = sum(weights)
    channels = nearest_k[0][0]["shift"].keys()
    return {
        col: sum(w * point["shift"][col] for (point, _d), w in zip(nearest_k, weights)) / total_weight
        for col in channels
    }


def nearest_distance(model: dict, affected_label: str,
                      source_positions: dict[str, float]) -> float:
    """Euclidean distance (in the OTHER-3-fingers' position space, 0-100%
    per dimension) to the nearest stored measurement -- a coverage/
    confidence diagnostic: large distances mean this query fell in a part
    of the pose space none of the measured combinations sampled closely,
    so predict_shift()'s answer there is a bigger extrapolation than
    usual. Not used by predict_shift() itself, only by diagnostics."""
    entry = model["affected"][affected_label]
    filled = {lbl: source_positions.get(lbl, 0.0) for lbl in entry["other_labels"]}
    _point, dist = _nearest(model, affected_label, filled)
    return dist
