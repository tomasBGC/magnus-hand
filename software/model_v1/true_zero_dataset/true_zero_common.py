"""Shared constants/paths for the true_zero_dataset pipeline (sample ->
record -> build -> analyze -> retrain).

Standalone, additive pipeline (same convention as model_v1/crosstalk_lookup/)
-- does not modify anything in model_v1/crosstalk/, model_v1/crosstalk_lookup/,
or the live sensing pipeline. Reuses model_v1/crosstalk/crosstalk_common.py
for RAW_COLS/FINGER_LABELS (import, not copy) since those are just naming
conventions, not data.

Why this exists, distinct from model_v1/crosstalk_lookup/: that pipeline
measures raw magnetic *crosstalk shift* at 100 simultaneous-4-finger
combinations, for fitting/comparing correction backends. This pipeline
measures the *live model's actual DISPLAYED output* (Force as the deployed
GUI would show it -- post baseline, post crosstalk correction, post Contact
classifier, post hysteresis, post the 2026-07-17 auto-rebaseline fix) at
500 simultaneous-4-finger combinations, specifically to find and then fix
remaining "no touch, should read 0N, but doesn't" cases across the full
position space -- a direct, measured-data continuation of the same class of
issue the self-closure fix and the auto-rebaseline fix both already
targeted, at much broader (genuinely simultaneous, 4-finger, 500-combination)
coverage than either of those.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw"
MANIFEST_JSON = SCRIPT_DIR / "combinations_manifest.json"
DATASET_CSV = SCRIPT_DIR / "true_zero_dataset.csv"
ANALYSIS_REPORT_MD = SCRIPT_DIR / "true_zero_report.md"

_CROSSTALK_DIR = SCRIPT_DIR.parent / "crosstalk"
if str(_CROSSTALK_DIR) not in sys.path:
    sys.path.insert(0, str(_CROSSTALK_DIR))
import crosstalk_common as cc  # noqa: E402

RAW_COLS = cc.RAW_COLS
FINGER_LABELS = cc.FINGER_LABELS

REFERENCE_POSITIONS = {label: 0.0 for label in FINGER_LABELS}

# --- Safety constraint (Index+Thumb collision) ------------------------------
# Per explicit instruction for THIS data-collection run: Index and Thumb can
# never simultaneously EXCEED 15% closure (strictly tighter than
# crosstalk_lookup's own 20% cap, and record_crosstalk_combined_round2.py's
# 30% cap -- each is that script's own considered choice for its own
# session; this module deliberately does NOT import either, to avoid any
# risk of silently inheriting a looser cap here). "Exceed" means strictly
# greater than 15 -- both fingers sitting AT exactly 15% simultaneously is
# allowed (the closest-without-touching case, deliberately oversampled by
# sample_combinations.py -- see its own docstring).
INDEX_THUMB_SAFETY_CAP_PCT = 15

SAFETY_CAPS: dict[frozenset[str], int] = {
    frozenset({"Index", "Thumb"}): INDEX_THUMB_SAFETY_CAP_PCT,
}


def violates_safety_caps(positions: dict[str, float]) -> str | None:
    """Checks `positions` (label -> percent, 0=open..100=closed) against
    every pair in SAFETY_CAPS. Returns a human-readable reason string if
    any pair's cap is violated (both members strictly above the cap
    simultaneously), else None.

    Pure/no side effects -- safe to call at sampling time (no hardware
    touched yet) and again at recording time (defensive re-check
    immediately before every servo move actually sent) with the exact same
    function, so both call sites can never drift out of sync with each
    other -- same pattern as crosstalk_lookup_common.violates_safety_caps.
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
