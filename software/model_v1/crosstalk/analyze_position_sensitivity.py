"""Investigates whether the crosstalk correction is meaningfully sensitive
to small errors in its POSITION input -- relevant because the finger-to-
palm joints have mechanical backlash/play, so a commanded position (e.g.
100%) doesn't always correspond to the exact same true physical position
across repetitions. The correction relies on COMMANDED position as a
proxy for TRUE position (see sensing_gui.py's set_finger_position), so if
crosstalk_model.py's fitted polynomials are highly sensitive to small
position errors, backlash-scale uncertainty (assumed here to be a few
percent -- a mechanical fix isn't practical right now, so this is
investigating a SOFTWARE mitigation) could itself be injecting residual
error, separate from anything already investigated (Index's crosstalk fit
quality, the transit-timing mismatch, self-motion, or the stale-baseline
cascade).

Method: for every (affected, source) pair and every one of the 15 raw
channels, evaluates crosstalk_model.py's ACTUAL predict_polynomial()
(imported, not reimplemented) at every integer position 0-100%, and at
each position P, computes |f(P + backlash) - f(P)| for several assumed
backlash magnitudes -- the worst-case correction SWING a commanded
position P could produce if the true physical position were actually
P +/- backlash. Compared against that channel's own noise_floor_std_at_p0
(already stored in the fitted model, from build_crosstalk_dataset.py's
within-recording standard deviation at the reference position) as a
ratio: swing / noise_floor. A ratio >> 1 means position uncertainty alone
could produce a correction error larger than the sensor's own idle noise
-- i.e. a real, independent contributor to residual error, not something
already "priced in" to what the model's noise-floor comparisons already
account for.

Run standalone (no hardware needed -- reads the already-fitted
crosstalk_model_v1.json only).
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import crosstalk_common as cc
import crosstalk_model as cm

REPORT_MD = SCRIPT_DIR / "position_sensitivity_report.md"

# Plausible backlash magnitudes to test, in position-percent units -- "a
# few percent" per the mechanical description given (finger-to-palm joint
# play), tested across a small range since the true magnitude isn't
# precisely known without measuring the physical joints directly.
BACKLASH_MAGNITUDES_PCT = [2.0, 3.0, 5.0, 8.0]

# A swing/noise-floor ratio at or above this is treated as "comparable to
# a real crosstalk effect" -- reuses crosstalk_common's own existing
# significance bar (SIGNIFICANT_SNR_THRESHOLD) for consistency, rather
# than inventing a new threshold for this different-but-related question.
FLAG_RATIO = cc.SIGNIFICANT_SNR_THRESHOLD

POSITION_STEP = 1  # integer percent granularity to scan


def max_swing_for_channel(coeffs: list[float], backlash: float) -> tuple[float, float]:
    """Returns (max_swing, position_of_max_swing) -- the largest
    |f(P+backlash) - f(P)| over the full 0-100% range (clamped so
    P+backlash / P-backlash both stay in-range), checking both directions
    since backlash could push the true position either side of commanded."""
    best_swing = 0.0
    best_p = 0.0
    for p in range(0, 101, POSITION_STEP):
        f_p = cm.predict_polynomial(coeffs, p)
        for direction in (+1, -1):
            p2 = p + direction * backlash
            if p2 < 0 or p2 > 100:
                continue
            f_p2 = cm.predict_polynomial(coeffs, p2)
            swing = abs(f_p2 - f_p)
            if swing > best_swing:
                best_swing = swing
                best_p = p
    return best_swing, best_p


def worst_case_deadband_error(coeffs: list[float], commanded_p: float,
                               deadband_pct: float, backlash_pct: float) -> float:
    """Models the deadband/quantization step itself, not just a raw
    perturbation -- distinct from max_swing_for_channel() above, which
    only measures how much the CONTINUOUS polynomial swings between two
    perturbed positions. This answers a different, more direct question:
    given a COMMANDED position (what the deadband actually rounds), how
    far can the resulting correction be from the TRUE correction the
    (unobservable) true physical position actually needed?

    deadband_pct=0 means no quantization -- the estimate IS commanded_p,
    matching pre-deadband behaviour, so this doubles as the "no deadband"
    baseline for comparison.

    The deadband quantizes the ESTIMATE (built from commanded_p, all the
    software ever knows) -- it does NOT and CANNOT observe the true
    physical position, so this cannot fix any BIAS from backlash, only
    change how much the quantization itself adds ON TOP of that
    irreducible bias. This function measures the combination of both,
    which is what actually determines whether widening the deadband helps
    or not for a given commanded position.
    """
    if deadband_pct > 0:
        estimate = round(commanded_p / deadband_pct) * deadband_pct
    else:
        estimate = commanded_p
    computed_correction = cm.predict_polynomial(coeffs, estimate)

    worst = 0.0
    steps = 21
    for i in range(steps):
        true_p = commanded_p - backlash_pct + (2 * backlash_pct) * i / (steps - 1)
        true_p = max(0.0, min(100.0, true_p))
        true_correction = cm.predict_polynomial(coeffs, true_p)
        worst = max(worst, abs(computed_correction - true_correction))
    return worst


def deadband_sweep_for_pair(model: dict, affected: str, source: str,
                             deadband_candidates: list[float],
                             backlash_pct: float) -> dict[float, tuple[float, float, str]]:
    """For one (affected, source) pair, returns {deadband_pct: (worst_error,
    mean_error, worst_channel)} across ALL 15 channels and the full 0-100%
    commanded-position range, for each candidate deadband grain -- lets
    candidates be compared directly against each other AND against
    deadband=0 (no quantization at all)."""
    pair = model["pairs"][affected][source]
    results = {}
    for deadband in deadband_candidates:
        worst_overall = 0.0
        worst_channel = None
        all_errors = []
        for channel, chdata in pair["channels"].items():
            coeffs = chdata["coeffs"]
            channel_worst = 0.0
            for p in range(0, 101, POSITION_STEP):
                err = worst_case_deadband_error(coeffs, p, deadband, backlash_pct)
                channel_worst = max(channel_worst, err)
                all_errors.append(err)
            if channel_worst > worst_overall:
                worst_overall = channel_worst
                worst_channel = channel
        mean_error = sum(all_errors) / len(all_errors)
        results[deadband] = (worst_overall, mean_error, worst_channel)
    return results


def main() -> None:
    model = cm.load_model(cc.MODEL_JSON)

    lines: list[str] = []
    lines.append("# Crosstalk correction position-sensitivity report\n")
    lines.append("Generated by `analyze_position_sensitivity.py`. Investigates whether "
                 "small position errors (from finger-to-palm joint backlash/play) could "
                 "themselves be a meaningful source of residual crosstalk-correction error, "
                 "separate from the correction's own fit quality or the transit-timing "
                 "mismatch already fixed. See `README.md` for the conclusion and any "
                 "resulting mitigation.\n")

    lines.append(f"For each channel, the worst-case correction SWING from a +/-N% position "
                 f"error is compared against that channel's own sensor noise floor "
                 f"(`noise_floor_std_at_p0`) as a ratio. Ratio >= {FLAG_RATIO:.1f} is flagged "
                 f"as comparable to what this pipeline already calls a significant crosstalk "
                 f"effect (`crosstalk_common.SIGNIFICANT_SNR_THRESHOLD`).\n")

    # ---- Per-pair worst-channel summary, one row per backlash magnitude ----
    for backlash in BACKLASH_MAGNITUDES_PCT:
        lines.append(f"\n## Backlash = +/-{backlash:.0f}%\n")
        lines.append("| Affected | Source | Significant? | Worst channel | Swing | "
                     "Noise floor | **Swing/noise ratio** | @ position |")
        lines.append("|---|---|---|---|---|---|---|---|")

        for affected in cc.FINGER_LABELS:
            for source in cc.FINGER_LABELS:
                if source == affected:
                    continue
                pair = model["pairs"][affected][source]
                worst_ratio = -1.0
                worst_channel = None
                worst_swing = 0.0
                worst_noise = 0.0
                worst_p = 0.0
                for channel, chdata in pair["channels"].items():
                    swing, p = max_swing_for_channel(chdata["coeffs"], backlash)
                    noise = chdata["noise_floor_std_at_p0"]
                    ratio = swing / noise if noise > 1e-9 else float("inf")
                    if ratio > worst_ratio:
                        worst_ratio = ratio
                        worst_channel = channel
                        worst_swing = swing
                        worst_noise = noise
                        worst_p = p

                flag = "**YES**" if pair["significant"] else "no"
                ratio_str = f"**{worst_ratio:.1f}**" if worst_ratio >= FLAG_RATIO else f"{worst_ratio:.1f}"
                lines.append(
                    f"| {affected} | {source} | {flag} | {worst_channel} | "
                    f"{worst_swing:.2f} | {worst_noise:.2f} | {ratio_str} | {worst_p:.0f}% |"
                )

    # ---- Detailed sensitivity curve for the single worst pair (Index<-Middle,
    # already known to be the noisiest-fit pair) at each backlash magnitude,
    # broken out by position, to see WHERE across the range it's worst -----
    lines.append("\n## Detail: Index <- Middle (the pair with the worst crosstalk-fit "
                 "residual error, per crosstalk_model_report.md) across the position range\n")
    lines.append("Shows the worst-channel swing/noise ratio at each of several positions, "
                 "for each backlash magnitude, so the SHAPE of the sensitivity (not just its "
                 "single worst point) is visible.\n")
    pair = model["pairs"]["Index"]["Middle"]
    lines.append("| Position | " + " | ".join(f"+/-{b:.0f}%" for b in BACKLASH_MAGNITUDES_PCT) + " |")
    lines.append("|---|" + "---|" * len(BACKLASH_MAGNITUDES_PCT))
    for p in range(0, 101, 10):
        row = [f"{p}%"]
        for backlash in BACKLASH_MAGNITUDES_PCT:
            worst_ratio = 0.0
            for channel, chdata in pair["channels"].items():
                f_p = cm.predict_polynomial(chdata["coeffs"], p)
                for direction in (+1, -1):
                    p2 = p + direction * backlash
                    if p2 < 0 or p2 > 100:
                        continue
                    swing = abs(cm.predict_polynomial(chdata["coeffs"], p2) - f_p)
                    noise = chdata["noise_floor_std_at_p0"]
                    ratio = swing / noise if noise > 1e-9 else float("inf")
                    worst_ratio = max(worst_ratio, ratio)
            row.append(f"{worst_ratio:.1f}")
        lines.append("| " + " | ".join(row) + " |")

    # ---- Overall conclusion ------------------------------------------------
    lines.append("\n## Conclusion\n")
    significant_pairs = [
        (affected, source)
        for affected in cc.FINGER_LABELS for source in cc.FINGER_LABELS
        if source != affected and model["pairs"][affected][source]["significant"]
    ]
    for backlash in BACKLASH_MAGNITUDES_PCT:
        flagged = []
        for affected, source in significant_pairs:
            pair = model["pairs"][affected][source]
            worst_ratio = max(
                max_swing_for_channel(chdata["coeffs"], backlash)[0] / chdata["noise_floor_std_at_p0"]
                for chdata in pair["channels"].values()
                if chdata["noise_floor_std_at_p0"] > 1e-9
            )
            if worst_ratio >= FLAG_RATIO:
                flagged.append((affected, source, worst_ratio))
        lines.append(f"- Backlash +/-{backlash:.0f}%: {len(flagged)}/{len(significant_pairs)} "
                     f"significant pairs flagged (ratio >= {FLAG_RATIO:.1f}): "
                     + (", ".join(f"{a}<-{s} ({r:.1f}x)" for a, s, r in flagged) if flagged else "none"))

    # ---- Deadband widening effectiveness, specifically for Index's own
    # crosstalk sources (Middle, Thumb) -- does WIDENING
    # CROSSTALK_POSITION_DEADBAND_PCT (currently 5%) reduce Index's
    # remaining false-contact sensitivity, or does a wider bucket's larger
    # gap between adjacent bucket centers cancel out the benefit of fewer
    # boundary crossings? Uses worst_case_deadband_error(), which models
    # the quantization step itself (not just a raw perturbation), so this
    # answers the ACTUAL question rather than assuming "wider = safer."
    # ---------------------------------------------------------------------
    lines.append("\n## Deadband widening effectiveness (Index's own crosstalk sources)\n")
    lines.append("Does widening `CROSSTALK_POSITION_DEADBAND_PCT` beyond the current 5% "
                 "reduce Index's remaining false-contact sensitivity? Unlike the swing-based "
                 "tables above (which measure raw polynomial sensitivity), this models the "
                 "DEADBAND ITSELF: for a commanded position, quantizes it to each candidate "
                 "grain, then finds the worst-case gap between the resulting correction and "
                 "the TRUE correction needed for any physically-plausible true position within "
                 "the backlash range. A wider bucket means fewer boundary crossings, but each "
                 "crossing swings further (adjacent bucket centers are farther apart) -- this "
                 "is a genuine trade-off, not something that can be assumed to favour either "
                 "direction without checking.\n")

    deadband_candidates = [0.0, 5.0, 10.0]
    index_pairs = [("Index", "Middle"), ("Index", "Thumb")]

    for backlash in BACKLASH_MAGNITUDES_PCT:
        lines.append(f"\n### Backlash = +/-{backlash:.0f}%\n")
        lines.append("| Pair | Deadband | Worst-case error (any commanded position) | "
                     "Mean error (across full 0-100% range) | Worst channel |")
        lines.append("|---|---|---|---|---|")
        for affected, source in index_pairs:
            sweep = deadband_sweep_for_pair(model, affected, source, deadband_candidates, backlash)
            best_deadband = min(sweep, key=lambda d: sweep[d][0])
            for deadband in deadband_candidates:
                worst_err, mean_err, worst_ch = sweep[deadband]
                label = "no deadband" if deadband == 0.0 else f"{deadband:.0f}%"
                marker = " **<- best**" if deadband == best_deadband else ""
                lines.append(f"| {affected}<-{source} | {label} | {worst_err:.2f} | "
                             f"{mean_err:.2f} | {worst_ch}{marker} |")

    lines.append("\n### Conclusion (deadband widening)\n")
    any_meaningful_improvement = False
    for affected, source in index_pairs:
        for backlash in BACKLASH_MAGNITUDES_PCT:
            sweep = deadband_sweep_for_pair(model, affected, source, deadband_candidates, backlash)
            worst_at_5 = sweep[5.0][0]
            worst_at_10 = sweep[10.0][0]
            worst_at_0 = sweep[0.0][0]
            improvement_5_to_10 = (worst_at_5 - worst_at_10) / worst_at_5 * 100 if worst_at_5 > 1e-9 else 0.0
            if improvement_5_to_10 >= 15.0:  # a genuinely meaningful reduction, not noise
                any_meaningful_improvement = True
            lines.append(
                f"- {affected}<-{source} @ +/-{backlash:.0f}% backlash: worst-case error "
                f"no-deadband={worst_at_0:.2f}, 5%={worst_at_5:.2f}, 10%={worst_at_10:.2f} "
                f"({'IMPROVES' if worst_at_10 < worst_at_5 else 'WORSE or unchanged'} "
                f"going from 5% to 10%, {improvement_5_to_10:+.0f}%)"
            )
    lines.append(f"\n**Overall: widening to 10% {'DOES' if any_meaningful_improvement else 'does NOT'} "
                 f"meaningfully reduce worst-case error for Index's crosstalk sources.** "
                 + ("See the per-backlash detail above for which specific cases improve." if any_meaningful_improvement
                    else "The current 5% deadband already captures most of the achievable benefit from "
                         "quantization alone -- if Index still shows residual false-contact noise, it is "
                         "more likely dominated by the underlying crosstalk fit's own residual error "
                         "(Index<-Middle's fit was already identified as the noisiest among the major "
                         "pairs, see crosstalk_model_report.md) than by position-rounding jitter, and "
                         "further deadband tuning alone should not be expected to resolve it."))

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
