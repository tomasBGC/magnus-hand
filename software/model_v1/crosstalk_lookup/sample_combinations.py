"""Step 1: generates exactly 100 valid 4-finger position combinations to
record, via Latin Hypercube Sampling (scipy.stats.qmc) over the full
[0, 100]^4 (Index, Middle, Ring, Thumb) pose space, filtered against the
Index+Thumb safety cap (see crosstalk_lookup_common.SAFETY_CAPS).

Pure software -- no hardware touched, no serial port opened. Run this
first; record_crosstalk_lookup.py then reads its output
(combinations_manifest.json) rather than sampling inline, so a recording
session that gets interrupted partway through can be resumed/rerun against
the exact same, already-inspectable combination list instead of a fresh
random draw.

Why LHS + rejection rather than a smaller direct/stratified design: the
safety filter removes P(Index>20%)*P(Thumb>20%) = 0.8*0.8 = 64% of the raw
4D volume (only one of Index/Thumb may exceed 20% at a time), so roughly
2.8x oversampling is needed on average to land 100 valid combinations.
Rather than guess a single oversample factor, this draws successive LHS
batches (each internally space-filling) with independent sub-seeds until
enough valid combinations accumulate, then takes exactly the first 100 in
draw order -- deterministic given RANDOM_SEED, and every rejected
combination is logged (both to console and into the manifest itself) per
the "log every rejected/skipped combination attempt" requirement.

Usage:
    python model_v1/crosstalk_lookup/sample_combinations.py [--n 100] [--seed 20260715]
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

import crosstalk_lookup_common as clc

N_COMBINATIONS_DEFAULT = 100
RANDOM_SEED_DEFAULT = 20260715  # fixed for reproducibility -- see module docstring

# Each LHS batch is oversized relative to what's still needed, since ~64%
# of any batch will be rejected on average (see module docstring) -- sizing
# each batch off the CURRENT shortfall (rather than one huge fixed draw)
# keeps the final combination count close to exactly what was requested
# without a large excess of unused, already-rejected draws sitting in one
# giant first batch.
OVERSAMPLE_FACTOR = 3.5


def sample_batch(n: int, seed: int) -> list[dict[str, float]]:
    """One space-filling LHS batch of size `n` over [0, 100]^4, positions
    rounded to the nearest whole percent (matches the existing pipeline's
    integer-percent convention -- see crosstalk_common.POSITIONS)."""
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    unit_samples = sampler.random(n=n)
    scaled = unit_samples * 100.0
    return [
        {label: round(float(val)) for label, val in zip(clc.FINGER_LABELS, row)}
        for row in scaled
    ]


def generate_combinations(n_target: int, seed: int) -> tuple[list[dict[str, float]], list[dict]]:
    """Returns (accepted, rejected_log). accepted has exactly n_target
    entries (in draw order); rejected_log has one entry per combination
    that failed the safety filter, in the order it was drawn, each with
    the reason it was rejected."""
    accepted: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()
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
            key = tuple(combo[label] for label in clc.FINGER_LABELS)
            if key in seen:
                continue  # duplicate draw (possible after rounding) -- skip silently, not a safety rejection
            seen.add(key)
            reason = clc.violates_safety_caps(combo)
            if reason is not None:
                rejected_log.append({
                    "batch": batch_index,
                    "combination": combo,
                    "reason": reason,
                })
                continue
            accepted.append(combo)
        batch_seed += 1  # next batch gets a fresh sub-seed, still deterministic overall

    return accepted, rejected_log


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate N safety-filtered 4-finger position combinations via Latin Hypercube Sampling")
    parser.add_argument("--n", type=int, default=N_COMBINATIONS_DEFAULT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT)
    parser.add_argument("--out", default=str(clc.MANIFEST_JSON))
    args = parser.parse_args()

    accepted, rejected_log = generate_combinations(args.n, args.seed)

    for r in rejected_log:
        c = r["combination"]
        print(f"  [REJECTED] batch {r['batch']}: Index={c['Index']:.0f}% Middle={c['Middle']:.0f}% "
              f"Ring={c['Ring']:.0f}% Thumb={c['Thumb']:.0f}% -- {r['reason']}")

    # Defensive re-verification of the accepted set itself, before writing
    # anything out -- belt-and-braces against a logic error in the filter
    # above (this pipeline's safety requirement is "enforced in code", so
    # the manifest this feeds into a live hardware session must never
    # silently carry a violating combination even if generate_combinations()
    # itself had a bug).
    still_bad = [c for c in accepted if clc.violates_safety_caps(c) is not None]
    if still_bad:
        raise SystemExit(f"INTERNAL ERROR: {len(still_bad)} accepted combination(s) still violate "
                          f"the safety cap -- refusing to write a manifest. First: {still_bad[0]}")

    manifest = {
        "generated_at_iso": datetime.now(timezone.utc).isoformat(),
        "tool": "sample_combinations.py",
        "method": "Latin Hypercube Sampling (scipy.stats.qmc.LatinHypercube), 4D over "
                  "[0,100]% per finger, positions rounded to nearest whole percent",
        "random_seed": args.seed,
        "n_requested": args.n,
        "n_accepted": len(accepted),
        "n_rejected": len(rejected_log),
        "safety_cap_pct": clc.INDEX_THUMB_SAFETY_CAP_PCT,
        "reference_position": clc.REFERENCE_POSITIONS,
        "rejected_log": rejected_log,
        "combinations": [
            {"combo_index": i, **combo} for i, combo in enumerate(accepted)
        ],
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{len(accepted)} valid combinations accepted, {len(rejected_log)} rejected by the "
          f"{clc.INDEX_THUMB_SAFETY_CAP_PCT}% Index+Thumb safety cap.")
    print(f"Manifest written -> {out_path}")


if __name__ == "__main__":
    main()
