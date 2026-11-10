"""Validates the additive crosstalk-model assumption against the
combined-movement recordings in raw_combined/ (see
record_crosstalk_combined.py), which moved 2 fingers SIMULTANEOUSLY --
something none of the single-finger sweeps that fitted
crosstalk_model_v1.json ever did.

For each combined recording (2 source fingers at a matching position P,
measured on 1 affected finger, 15 channels each), computes:

    additive_prediction = predict_shift(model, affected, {A: P, B: P})[channel]
                         = f_{affected,A}(P) + f_{affected,B}(P)

using crosstalk_model.py's actual predict_shift() (the same function a
future integration would call), and compares it against
combined_dataset.csv's delta_from_reference (the ACTUALLY measured
combined-movement baseline shift).

The raw discrepancy alone is hard to judge in isolation -- crosstalk_model_v1.json's
own per-channel fits are not perfect even for the single-finger case (see
LOOCV MAE in crosstalk_model_report.md's per-pair table). So each
discrepancy is also compared against an "expected noise-only" scale:
sqrt(loocv_mae_A^2 + loocv_mae_B^2), i.e. how much combined-prediction
error would be expected from JUST the two individual pairwise models'
own fit uncertainty, if additivity held exactly and the two models' errors
were independent. A ratio near/below 1 means the discrepancy is
explained by ordinary model noise; a ratio well above 1 means there is
error beyond what either individual model's own uncertainty accounts for
-- evidence of a real interaction effect, not just noise.

Run after build_combined_dataset.py. Appends/replaces an "Additivity
validation" section in crosstalk_model_report.md -- idempotent (reruns
replace the existing section by heading match rather than duplicating
it), and does not touch anything above that section (the single-finger
fit-quality report evaluate_crosstalk_model.py writes).
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import crosstalk_common as cc
import crosstalk_model as cm

COMBINED_CSV = SCRIPT_DIR / "combined_dataset.csv"
SECTION_HEADING = "## Additivity validation (combined-movement test)"

# Above this mean noise-adjusted ratio, a (affected, source-pair) triple is
# judged to show a real interaction effect beyond individual-model noise --
# see module docstring. Below it, the additive assumption is judged to
# hold well enough for that triple. A simple, documented threshold (like
# SIGNIFICANT_SNR_THRESHOLD elsewhere in this pipeline), not a formal test.
BREAKDOWN_RATIO_THRESHOLD = 2.0


def load_combined_rows() -> list[dict]:
    with open(COMBINED_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def expected_noise(model: dict, affected: str, source_a: str, source_b: str, channel: str) -> float:
    mae_a = model["pairs"][affected][source_a]["channels"][channel]["loocv_mae"]
    mae_b = model["pairs"][affected][source_b]["channels"][channel]["loocv_mae"]
    return (mae_a ** 2 + mae_b ** 2) ** 0.5


def main() -> None:
    model = cm.load_model(cc.MODEL_JSON)
    rows = load_combined_rows()

    by_triple: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["affected_label"], row["source_labels"], int(row["position_percent"]))
        by_triple[key].append(row)

    triple_summary_lines = []
    pair_ratios: dict[tuple[str, str], list[float]] = defaultdict(list)
    pair_abs_errors: dict[tuple[str, str], list[float]] = defaultdict(list)
    pair_abs_actuals: dict[tuple[str, str], list[float]] = defaultdict(list)
    channel_detail_lines = []

    for (affected, source_labels, position), triple_rows in sorted(by_triple.items()):
        source_a, source_b = source_labels.split("+")
        prediction = cm.predict_shift(model, affected, {source_a: position, source_b: position})

        ratios = []
        for row in triple_rows:
            channel = row["channel"]
            actual = float(row["delta_from_reference"])
            predicted = prediction[channel]
            err = predicted - actual
            noise = expected_noise(model, affected, source_a, source_b, channel)
            ratio = abs(err) / noise if noise > 1e-9 else (0.0 if abs(err) < 1e-9 else float("inf"))
            ratios.append(ratio)

            pair_key = (affected, source_labels)
            pair_ratios[pair_key].append(ratio)
            pair_abs_errors[pair_key].append(abs(err))
            pair_abs_actuals[pair_key].append(abs(actual))

            channel_detail_lines.append(
                f"| {affected} | {source_labels} | {position} | {channel} | "
                f"{actual:+.2f} | {predicted:+.2f} | {err:+.2f} | {noise:.2f} | {ratio:.1f} |"
            )

        finite_ratios = [r for r in ratios if r != float("inf")]
        mean_ratio = statistics.fmean(finite_ratios) if finite_ratios else float("inf")
        mae = statistics.fmean(abs(float(r["delta_from_reference"])
                                    - prediction[r["channel"]]) for r in triple_rows)
        mean_actual_abs = statistics.fmean(abs(float(r["delta_from_reference"])) for r in triple_rows)
        triple_summary_lines.append(
            f"| {affected} | {source_labels} | {position} | {mae:.2f} | "
            f"{mean_actual_abs:.2f} | {mean_ratio:.1f} |"
        )

    pair_summary_lines = []
    verdicts = {}
    for pair_key in sorted(pair_ratios):
        affected, source_labels = pair_key
        finite = [r for r in pair_ratios[pair_key] if r != float("inf")]
        mean_ratio = statistics.fmean(finite) if finite else float("inf")
        mae = statistics.fmean(pair_abs_errors[pair_key])
        mean_actual = statistics.fmean(pair_abs_actuals[pair_key])
        pct_of_signal = (mae / mean_actual * 100.0) if mean_actual > 1e-9 else float("nan")
        holds = mean_ratio <= BREAKDOWN_RATIO_THRESHOLD
        verdicts[pair_key] = holds
        verdict_text = "holds (within model noise)" if holds else "**breaks down**"
        pair_summary_lines.append(
            f"| {affected} | {source_labels} | {mae:.2f} | {mean_actual:.2f} | "
            f"{pct_of_signal:.0f}% | {mean_ratio:.1f} | {verdict_text} |"
        )

    def describe(pair_key: tuple[str, str]) -> str:
        affected, source_labels = pair_key
        return f"{affected} affected by {source_labels}"

    n_hold = sum(1 for v in verdicts.values() if v)
    n_total = len(verdicts)

    # --- Coverage vs. the original per-finger CAUTION flags -----------------
    # evaluate_crosstalk_model.py flagged an affected finger CAUTION when 2+
    # of its 3 possible source fingers were individually "significant" (see
    # crosstalk_common.SIGNIFICANT_SNR_THRESHOLD). This validation set only
    # ever covers 2 combinations (whatever record_crosstalk_combined.py's
    # COMBINED_PAIRS chose to move together), so each tested pair is
    # categorised against that SAME affected finger's own flagged-
    # significant sources, and the narrative below is built entirely from
    # that categorisation -- not hand-written per finger -- so it stays
    # correct if COMBINED_PAIRS/the underlying data ever changes.
    match_category: dict[tuple[str, str], str] = {}
    coverage_lines = []
    uncovered_caution: list[str] = []
    for affected in cc.FINGER_LABELS:
        sig_sources = {s for s in cc.FINGER_LABELS
                        if s != affected and model["pairs"][affected][s]["significant"]}
        tested = {key for key in verdicts if key[0] == affected}
        if not tested:
            if len(sig_sources) >= 2:
                uncovered_caution.append(affected)
                coverage_lines.append(
                    f"- **{affected}**: originally flagged CAUTION (sources: "
                    f"{', '.join(sorted(sig_sources))}) -- **not covered** by this "
                    f"validation set (never appears as the affected finger in "
                    f"either combined recording).")
            continue
        is_caution = len(sig_sources) >= 2
        for key in tested:
            _affected, source_labels = key
            tested_sources = set(source_labels.split("+"))
            if not is_caution:
                # This affected finger was never flagged CAUTION to begin
                # with (0 or 1 significant sources) -- "partial match to a
                # flagged pair" language would misleadingly imply a CAUTION
                # risk existed here. It didn't; this test is still useful
                # (it shows how 2 near-noise-floor effects combine) but
                # isn't validating an originally-flagged risk.
                category = "not_flagged"
                sig_desc = ', '.join(sorted(sig_sources)) if sig_sources else "none"
                match = (f"not originally flagged CAUTION (own significant source(s): "
                         f"{sig_desc}) -- this test still exercises {', '.join(sorted(tested_sources))} "
                         f"together, informative about combining small/near-noise effects but not "
                         f"validating a flagged risk")
            elif tested_sources == sig_sources:
                category = "exact"
                match = "**exact match** to its own flagged CAUTION pair"
            elif tested_sources & sig_sources:
                category = "partial"
                match = (f"**partial match** (only {', '.join(sorted(tested_sources & sig_sources))} "
                         f"of the flagged {', '.join(sorted(sig_sources))} was included; "
                         f"{', '.join(sorted(tested_sources - sig_sources))} substituted a "
                         f"non-flagged source)")
            else:
                category = "none"
                match = (f"**no overlap** with the flagged sources ({', '.join(sorted(sig_sources))}) "
                         f"-- this test used 2 sources neither individually flagged significant "
                         f"for {affected}")
            match_category[key] = category
            verdict_word = "held" if verdicts[key] else "broke down"
            coverage_lines.append(
                f"- **{affected}** tested against {source_labels}: {match} -- result **{verdict_word}**.")

    exact = [k for k in verdicts if match_category.get(k) == "exact"]
    partial_or_none = [k for k in verdicts if match_category.get(k) in ("partial", "none")]
    not_flagged = [k for k in verdicts if match_category.get(k) == "not_flagged"]

    narrative = [f"Additivity held within {BREAKDOWN_RATIO_THRESHOLD:.1f}x noise for "
                 f"{n_hold}/{n_total} tested combinations overall."]

    if exact:
        exact_broke = [k for k in exact if not verdicts[k]]
        if exact_broke:
            narrative.append(
                f"Of the {len(exact)} combination(s) that exactly match an originally-flagged "
                f"CAUTION pair (both of that finger's own flagged-significant sources moved "
                f"together), {len(exact_broke)} broke down: {'; '.join(describe(k) for k in exact_broke)} "
                f"-- this IS evidence of a real interaction effect for that pair, not just noise.")
        else:
            narrative.append(
                f"All {len(exact)} combination(s) that exactly match an originally-flagged "
                f"CAUTION pair held within noise: {'; '.join(describe(k) for k in exact)} -- "
                f"additivity appears safe for that specific pair.")
    else:
        narrative.append(
            "None of the tested combinations exactly match an originally-flagged CAUTION pair "
            "(see the coverage note below for exactly what was and wasn't covered).")

    if partial_or_none:
        po_broke = [k for k in partial_or_none if not verdicts[k]]
        narrative.append(
            f"{len(partial_or_none)} tested combination(s) partially cover a flagged CAUTION "
            f"finger (one flagged source substituted for a non-flagged one); "
            f"{len(po_broke) if po_broke else 'none'} broke down"
            + (f": {'; '.join(describe(k) for k in po_broke)}." if po_broke else "."))

    if not_flagged:
        nf_broke = [k for k in not_flagged if not verdicts[k]]
        if nf_broke:
            narrative.append(
                f"{len(not_flagged)} tested combination(s) involved an affected finger that was "
                f"never flagged CAUTION in the first place (0 or 1 significant sources on its "
                f"own); {len(nf_broke)} of those broke down: "
                f"{'; '.join(describe(k) for k in nf_broke)}. Check the 'Mean │actual│' column in "
                f"the per-pair table for these -- a small absolute effect size there means the "
                f"breakdown more likely reflects noise-level prediction of an already-tiny shift "
                f"than a large physical interaction, and doesn't itself indicate the additive "
                f"model is unreliable where it actually matters.")
        else:
            narrative.append(
                f"{len(not_flagged)} tested combination(s) involved an affected finger that was "
                f"never flagged CAUTION in the first place; all held within noise.")

    if uncovered_caution:
        narrative.append(
            f"Not covered by this validation set at all: {', '.join(uncovered_caution)} "
            f"(originally flagged CAUTION, but never the affected finger in either combined "
            f"recording here) -- still an open question.")

    overall = " ".join(narrative)

    section = f"""{SECTION_HEADING}

Generated by `validate_additivity.py` from `combined_dataset.csv`
(6 recordings in `raw_combined/`: Index+Middle and Middle+Ring each moved
to matching positions {{25,50,75}}% SIMULTANEOUSLY, testing whether their
individually-fitted crosstalk effects actually sum the way
`crosstalk_model.py`'s `predict_shift()` assumes).

### Conclusion

{overall}

### Coverage vs. the original CAUTION flags

crosstalk_model_report.md's single-finger section flags an affected finger
CAUTION when 2+ of its 3 possible source fingers were individually
"significant" (SNR >= {cc.SIGNIFICANT_SNR_THRESHOLD:.1f}). This validation
set only exercises whichever finger pairs
`record_crosstalk_combined.py`'s `COMBINED_PAIRS` moved together, so this
maps each thing actually tested back against what it does/doesn't cover:

{chr(10).join(coverage_lines)}

### Per (affected finger, source pair) summary

Averaged over 3 positions x 15 channels = 45 data points per row.
"Mean │actual│" is the average magnitude of the real measured shift (how
big the effect actually is); "MAE" is the average |prediction - actual|
in the same raw units; "% of signal" is MAE as a percentage of mean
│actual│ (a large percentage means the prediction error is comparable to
the effect itself, not a small correction on top of it). "Noise-adjusted
ratio" is MAE divided by the expected error scale from the two individual
pairwise models' own LOOCV fit noise (see module docstring) -- ratio <=
{BREAKDOWN_RATIO_THRESHOLD:.1f} is judged "holds", above is "breaks down".

| Affected | Sources (both moved) | MAE | Mean │actual│ | % of signal | Noise-adjusted ratio | Verdict |
|---|---|---|---|---|---|---|
{chr(10).join(pair_summary_lines)}

### Per (affected, sources, position) detail

| Affected | Sources | Position | MAE (15 ch.) | Mean │actual│ (15 ch.) | Mean noise-adjusted ratio |
|---|---|---|---|---|---|
{chr(10).join(triple_summary_lines)}

### Per-channel detail (all {len(rows)} rows)

| Affected | Sources | Pos | Channel | Actual Δ | Predicted Δ | Error | Expected noise | Ratio |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(channel_detail_lines)}
"""

    existing = cc.REPORT_MD.read_text(encoding="utf-8") if cc.REPORT_MD.exists() else ""
    idx = existing.find(SECTION_HEADING)
    if idx == -1:
        updated = existing.rstrip("\n") + "\n\n" + section
    else:
        updated = existing[:idx].rstrip("\n") + "\n\n" + section

    cc.REPORT_MD.write_text(updated, encoding="utf-8")
    print(f"Updated {cc.REPORT_MD} with additivity validation section.")
    print(f"Overall verdict: {n_hold}/{n_total} (affected, source-pair) combinations "
          f"held within {BREAKDOWN_RATIO_THRESHOLD:.1f}x noise.")


if __name__ == "__main__":
    main()
