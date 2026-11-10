"""Shared constants/paths for the crosstalk_lookup pipeline (sample -> record
-> build -> evaluate).

Entirely separate from model_v1/crosstalk/'s own pairwise-additive-model
pipeline (crosstalk_common.py, build_crosstalk_dataset.py, etc.) -- nothing
there is modified by anything in this folder. Where it's a strict win to
reuse rather than duplicate (raw channel naming, finger labels, the
connection/servo/capture plumbing), this pipeline imports directly from
model_v1/crosstalk/ instead of copying -- see record_crosstalk_lookup.py's
own imports.

Why a separate folder at all, instead of extending model_v1/crosstalk/:
that folder's whole dataset (raw/*.json) was collected under a ONE-FINGER-
MOVES-AT-A-TIME protocol, which is exactly the assumption approach (a)
below (the existing pairwise-additive model) needs and approaches (b)/(c)
don't -- mixing the two datasets on disk would blur which recordings
support which assumption. Keeping this pipeline's ALL-4-FINGERS-MOVE-
TOGETHER dataset in its own folder keeps that boundary unambiguous.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw"
MANIFEST_JSON = SCRIPT_DIR / "combinations_manifest.json"
DATASET_CSV = SCRIPT_DIR / "crosstalk_lookup_dataset.csv"
COMPARISON_REPORT_MD = SCRIPT_DIR / "comparison_report.md"

# model_v1/crosstalk/ is imported (not copied) for RAW_COLS/FINGER_LABELS
# and the crosstalk_model.py library (approach (a) below) -- same pattern
# record_crosstalk_combined_round2.py already uses to import from
# record_crosstalk.py, just one folder further out.
_CROSSTALK_DIR = SCRIPT_DIR.parent / "crosstalk"
if str(_CROSSTALK_DIR) not in sys.path:
    sys.path.insert(0, str(_CROSSTALK_DIR))
import crosstalk_common as cc  # noqa: E402

RAW_COLS = cc.RAW_COLS
FINGER_LABELS = cc.FINGER_LABELS

# Reference pose every combination's "shift" is measured against -- the
# same all-open 0% pose model_v1/crosstalk/'s own delta_from_p0 uses, so
# approach (a)'s existing predict_shift() output (also a shift from 0%) is
# directly comparable to (b)/(c) on the same target quantity.
REFERENCE_POSITIONS = {label: 0 for label in FINGER_LABELS}

# --- Safety constraint (Index+Thumb collision) ------------------------------
# model_v1/crosstalk/record_crosstalk_combined_round2.py already established
# a hard collision cap for this exact pair (SAFETY_CAPS = {frozenset({
# "Index", "Thumb"}): 30}, confirmed hardware constraint: they crash into
# each other if both are closed past ~30% AT THE SAME TIME). This session
# uses a TIGHTER 20% cap, per explicit instruction for this data-collection
# run specifically -- broad random/LHS coverage of the full 4D pose space
# visits far more simultaneous-nonzero Index+Thumb combinations than the
# earlier targeted validation batches ever did (which only ever moved one
# flagged pair at a time, deliberately away from this corner), so a more
# conservative margin is warranted here. Both values are legitimate for
# their own script; this module intentionally does NOT import the 30%
# constant from record_crosstalk_combined_round2.py, to avoid any risk of
# silently inheriting the looser cap into this broader-coverage session.
INDEX_THUMB_SAFETY_CAP_PCT = 20

SAFETY_CAPS: dict[frozenset[str], int] = {
    frozenset({"Index", "Thumb"}): INDEX_THUMB_SAFETY_CAP_PCT,
}


def violates_safety_caps(positions: dict[str, float]) -> str | None:
    """Checks `positions` (label -> percent, 0=open..100=closed) against
    every pair in SAFETY_CAPS. Returns a human-readable reason string if
    any pair's cap is violated (both members of the pair simultaneously
    above the cap), else None.

    Pure/no side effects -- safe to call at sampling time (filtering
    candidate combinations, no hardware touched yet) and again at
    recording time (defensive re-check immediately before every single
    servo move actually sent) with the exact same function, so both call
    sites can never drift out of sync with each other.
    """
    for pair, cap in SAFETY_CAPS.items():
        a, b = sorted(pair)
        pos_a, pos_b = positions.get(a), positions.get(b)
        if pos_a is None or pos_b is None:
            continue
        if pos_a > cap and pos_b > cap:
            return (f"{a}={pos_a:.0f}% and {b}={pos_b:.0f}% are both above the "
                    f"{cap}% collision-risk cap for this pair simultaneously")
    return None
