"""Step 1: generates exactly 500 valid 4-finger position combinations to
record, via Latin Hypercube Sampling (scipy.stats.qmc) over the full
[0, 100]^4 (Index, Middle, Ring, Thumb) pose space, filtered against the
15% Index+Thumb safety cap (see true_zero_common.SAFETY_CAPS) -- PLUS a
deliberately-injected block of "boundary" combinations where Index and
Thumb both sit at EXACTLY 15% simultaneously (the closest-without-touching
case, and the most informative one for a model to learn "this is still
zero-force" from -- LHS alone would only very rarely land exactly on this
boundary by chance).

Pure software -- no hardware touched, no serial port opened. Run this
first; record_true_zero_dataset.py then reads its output
(combinations_manifest.json) rather than sampling inline, so a recording
session that gets interrupted partway through can be resumed/rerun against
the exact same, already-inspectable combination list instead of a fresh
random draw. Same two-stage design as model_v1/crosstalk_lookup/
sample_combinations.py, extended with the boundary-case injection.

Usage:
    python model_v1/true_zero_dataset/sample_combinations.py
        [--n 500] [--n-boundary 15] [--seed 20260717]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
from scipy.stats import qmc

import true_zero_common as tzc

N_COMBINATIONS_DEFAULT = 500
N_BOUNDARY_DEFAULT = 15  # within the 10-20 requested range
RANDOM_SEED_DEFAULT = 20260717

# Same reasoning as crosstalk_lookup/sample_combinations.py's own choice --
# the safety filter removes a large chunk of the raw 4D volume, so batches
# are oversized relative to the current shortfall rather than guessing one
# fixed factor up front.
OVERSAMPLE_FACTOR = 3.5


def sample_batch(n: int, seed: int) -> list[dict[str, float]]:
    """One space-filling LHS batch of size `n` over [0, 100]^4, positions
    rounded to the nearest whole percent."""
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    unit_samples = sampler.random(n=n)
    scaled = unit_samples * 100.0
    return [
        {label: round(float(val)) for label, val in zip(tzc.FINGER_LABELS, row)}
        for row in scaled
    ]


def generate_general_combinations(n_target: int, seed: int,
                                   exclude_keys: set) -> tuple[list[dict[str, float]], list[dict]]:
    """LHS-sampled combinations over the full space, safety-filtered.
    exclude_keys lets the boundary-case combinations (generated separately)
    avoid an (extremely unlikely but possible) exact duplicate here."""
    accepted: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set(exclude_keys)
    rejected_log: list[dict] = []
    batch_seed = seed
    batch_index = 0

    while len(accepted) < n_target:
        remaining = n_target - len(accepted)
        batch_n = max(remaining, int(remaining * OVERSAMPLE_FACTOR))
        batch = sample_batch(batch_n, seed=batch_seed)
        batch_index += 1
        for combo in batch:
            if len(accepted) >= n_target:
                break
            key = tuple(combo[label] for label in tzc.FINGER_LABELS)
            if key in seen:
                continue  # duplicate draw (or collides with a boundary combo) -- skip silently
            seen.add(key)
            reason = tzc.violates_safety_caps(combo)
            if reason is not None:
                rejected_log.append({
                    "batch": batch_index,
                    "combination": combo,
                    "reason": reason,
                    "source": "general_lhs",
                })
                continue
            accepted.append(combo)
        batch_seed += 1

    return accepted, rejected_log


def generate_boundary_combinations(n_boundary: int, seed: int) -> list[dict[str, float]]:
    """Index=Thumb=15 exactly (the closest-without-touching case, allowed
    since the safety cap is "strictly exceed", not "at or above"), with
    Middle/Ring spread across [0, 100] via a small independent LHS draw so
    this block isn't just one repeated point -- still gives the model
    varied Middle/Ring crosstalk context at the one Index/Thumb position
    that matters most here. All trivially pass the safety filter (15 is not
    > 15), but still run through violates_safety_caps() below as a
    defensive check rather than assumed.
    """
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    unit_samples = sampler.random(n=n_boundary)
    scaled = unit_samples * 100.0
    combos = []
    for row in scaled:
        combo = {
            "Index": 15.0,
            "Thumb": 15.0,
            "Middle": round(float(row[0])),
            "Ring": round(float(row[1])),
        }
        reason = tzc.violates_safety_caps(combo)
        if reason is not None:
            # Should be structurally impossible (15 is never > 15) -- but
            # checked explicitly rather than assumed, matching this
            # project's "never silently trust a safety-relevant assumption"
            # convention.
            raise SystemExit(f"INTERNAL ERROR: boundary combination unexpectedly "
                              f"violates the safety cap: {combo} -- {reason}")
        combos.append(combo)
    return combos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate N safety-filtered 4-finger position combinations (LHS + "
                    "deliberate Index=Thumb=15%% boundary block) for the true_zero_dataset pipeline")
    parser.add_argument("--n", type=int, default=N_COMBINATIONS_DEFAULT,
                         help="Total combinations (general + boundary)")
    parser.add_argument("--n-boundary", type=int, default=N_BOUNDARY_DEFAULT,
                         help="How many of --n are the Index=Thumb=15%% boundary case")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT)
    parser.add_argument("--out", default=str(tzc.MANIFEST_JSON))
    args = parser.parse_args()

    if args.n_boundary >= args.n:
        raise SystemExit(f"--n-boundary ({args.n_boundary}) must be less than --n ({args.n})")

    boundary_combos = generate_boundary_combinations(args.n_boundary, seed=args.seed + 1)
    boundary_keys = {tuple(c[label] for label in tzc.FINGER_LABELS) for c in boundary_combos}

    n_general = args.n - args.n_boundary
    general_combos, rejected_log = generate_general_combinations(
        n_general, seed=args.seed, exclude_keys=boundary_keys)

    for r in rejected_log:
        c = r["combination"]
        print(f"  [REJECTED] batch {r['batch']}: Index={c['Index']:.0f}% Middle={c['Middle']:.0f}% "
              f"Ring={c['Ring']:.0f}% Thumb={c['Thumb']:.0f}% -- {r['reason']}")

    # Interleave: boundary combos spread evenly through the session (not
    # all at the start or end) so any slow session-level drift affects them
    # the same way it affects the general combos, rather than confounding
    # "boundary case" with "early/late in the session".
    all_combos = list(general_combos)
    if len(all_combos) > 0:
        stride = max(1, len(all_combos) // max(1, len(boundary_combos)))
        for i, bc in enumerate(boundary_combos):
            insert_at = min(len(all_combos), (i + 1) * stride)
            all_combos.insert(insert_at, bc)
    else:
        all_combos = boundary_combos

    is_boundary = [tuple(c[l] for l in tzc.FINGER_LABELS) in boundary_keys for c in all_combos]

    # Defensive re-verification of the FULL accepted set, before writing
    # anything out -- belt-and-braces against a logic error above.
    still_bad = [c for c in all_combos if tzc.violates_safety_caps(c) is not None]
    if still_bad:
        raise SystemExit(f"INTERNAL ERROR: {len(still_bad)} accepted combination(s) still violate "
                          f"the safety cap -- refusing to write a manifest. First: {still_bad[0]}")

    manifest = {
        "generated_at_iso": datetime.now(timezone.utc).isoformat(),
        "tool": "sample_combinations.py (true_zero_dataset)",
        "method": "Latin Hypercube Sampling over [0,100]^4, positions rounded to nearest "
                  "whole percent, plus a dedicated Index=Thumb=15%% boundary block "
                  "(Middle/Ring LHS-varied) interleaved through the session",
        "random_seed": args.seed,
        "n_requested": args.n,
        "n_boundary_requested": args.n_boundary,
        "n_accepted": len(all_combos),
        "n_boundary_accepted": sum(is_boundary),
        "n_rejected": len(rejected_log),
        "safety_cap_pct": tzc.INDEX_THUMB_SAFETY_CAP_PCT,
        "reference_position": tzc.REFERENCE_POSITIONS,
        "rejected_log": rejected_log,
        "combinations": [
            {"combo_index": i, "is_boundary_case": bool(is_boundary[i]), **combo}
            for i, combo in enumerate(all_combos)
        ],
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{len(all_combos)} valid combinations accepted "
          f"({sum(is_boundary)} boundary + {len(all_combos) - sum(is_boundary)} general), "
          f"{len(rejected_log)} rejected by the {tzc.INDEX_THUMB_SAFETY_CAP_PCT}% "
          f"Index+Thumb safety cap.")
    print(f"Manifest written -> {out_path}")


if __name__ == "__main__":
    main()
