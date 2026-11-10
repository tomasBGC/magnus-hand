"""Step 3/4: evaluates the fitted additive crosstalk model
(crosstalk_model_v1.json) against crosstalk_dataset.csv, and writes a
human-readable report, crosstalk_model_report.md, summarising:

  - an end-to-end sanity check of every dataset row through the SAME
    crosstalk_model.predict_shift() additive-combination code path a
    future integration would use (only one source finger active per row,
    since that's all the data contains -- see its docstring below for
    exactly what this does and doesn't validate)
  - per-(affected, source) pair fit quality: in-sample R^2 and LOOCV MAE,
    averaged over the 15 raw channels (already computed per channel by
    train_crosstalk_model.py)
  - per-pair effect size vs noise floor (SNR) and the "significant" flag
  - a per-affected-finger additivity-safety summary: whether more than
    one of its 3 possible source fingers has a "significant" effect --
    exactly the situation the additive assumption is UNTESTED for (see
    README.md)

Run after train_crosstalk_model.py.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import crosstalk_common as cc
import crosstalk_model as cm


def load_dataset_rows() -> list[dict]:
    with open(cc.DATASET_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_additive_predictions(model: dict, rows: list[dict]) -> dict:
    """Re-predicts every dataset row's delta via predict_shift() and
    compares to the actual measured delta. Since every recording moved
    exactly one finger, only one term of the additive sum is ever
    non-zero per row -- this confirms the pairwise functions are wired
    correctly end-to-end through the additive-prediction code path, NOT
    that the additive COMBINATION of 2+ simultaneous sources is correct
    (no data exists here to test that -- see README.md).
    """
    abs_errors = []
    for row in rows:
        affected = row["affected_label"]
        source = row["source_label"]
        position = float(row["position_percent"])
        channel = row["channel"]
        actual = float(row["delta_from_p0"])
        predicted = cm.predict_shift(model, affected, {source: position})[channel]
        abs_errors.append(abs(predicted - actual))
    return {
        "n": len(abs_errors),
        "mae": sum(abs_errors) / len(abs_errors) if abs_errors else float("nan"),
        "max": max(abs_errors) if abs_errors else float("nan"),
    }


def format_pair_table(model: dict) -> str:
    lines = [
        "| Affected | Source | Mean R² (w/ signal) | Mean LOOCV MAE | Effect ch. | "
        "Max│Δ│ | Noise σ (P=0) | SNR | Significant? |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for affected in cc.FINGER_LABELS:
        for source in cc.FINGER_LABELS:
            if source == affected:
                continue
            pair = model["pairs"][affected][source]
            r2s = [cf["in_sample_r2"] for cf in pair["channels"].values() if cf["in_sample_r2"] is not None]
            maes = [cf["loocv_mae"] for cf in pair["channels"].values()]
            n_channels = len(pair["channels"])
            mean_r2 = sum(r2s) / len(r2s) if r2s else float("nan")
            mean_mae = sum(maes) / len(maes)
            flag = "**YES**" if pair["significant"] else "no"
            lines.append(
                f"| {affected} | {source} | {mean_r2:.3f} ({len(r2s)}/{n_channels} ch.) "
                f"| {mean_mae:.2f} | {pair['effect_channel']} | {pair['effect_size_max_abs_delta']:.2f} "
                f"| {pair['noise_floor_std_at_p0']:.2f} | {pair['snr']:.1f} | {flag} |"
            )
    return "\n".join(lines)


def additivity_summary(model: dict) -> str:
    lines = []
    for affected in cc.FINGER_LABELS:
        sources = [s for s in cc.FINGER_LABELS if s != affected]
        sig = [s for s in sources if model["pairs"][affected][s]["significant"]]
        if len(sig) >= 2:
            flag = (f"**CAUTION** -- {len(sig)} significant sources ({', '.join(sig)}). "
                    f"The additive assumption combining these is UNTESTED: no recording "
                    f"here ever moved more than one finger at once. Real simultaneous-"
                    f"movement data is needed before trusting the summed prediction for "
                    f"{affected} when 2+ of its neighbours move together.")
        elif len(sig) == 1:
            flag = (f"OK -- only one significant source ({sig[0]}); there is nothing "
                    f"non-negligible for it to additively combine WITH in practice (the "
                    f"other 2 sources' effects are within the noise floor), so additivity "
                    f"is effectively a non-issue for {affected} today.")
        else:
            flag = f"OK -- no source finger produces an effect above the noise floor for {affected}."
        lines.append(f"- **{affected}**: {flag}")
    return "\n".join(lines)


def main() -> None:
    model = cm.load_model(cc.MODEL_JSON)
    rows = load_dataset_rows()

    pointwise = check_additive_predictions(model, rows)
    n_pairs = len(cc.FINGER_LABELS) * (len(cc.FINGER_LABELS) - 1)

    report = f"""# Crosstalk model report

Generated by `evaluate_crosstalk_model.py` from `{cc.MODEL_JSON.name}` and
`{cc.DATASET_CSV.name}`. See `README.md` for the additive-model assumption
and its limitations in full; this file is regenerated every run of the
pipeline, not hand-edited.

## Data

- {len(cc.FINGER_LABELS)} fingers x {len(cc.POSITIONS)} positions x 1 recording each = 20 recordings.
- {len(rows)} dataset rows ({len(cc.FINGER_LABELS)} fingers x {len(cc.FINGER_LABELS) - 1} affected x {len(cc.POSITIONS)} positions x {len(cc.RAW_COLS)} channels).
- Every recording moved exactly ONE finger; the other 3 were held at the reference (0%, open) position throughout, idle (no contact).

## End-to-end additive-prediction check (single-source rows only)

Re-predicting every dataset row's measured delta via the SAME
`crosstalk_model.predict_shift()` additive-combination code path a future
integration would use (with only one source finger active per row, since
that is all the data contains):

- n = {pointwise['n']} (fingers x positions x channels)
- Mean absolute error = {pointwise['mae']:.2f} (raw sensor units)
- Max absolute error = {pointwise['max']:.2f}

This is expected to match each channel's own in-sample fit essentially
exactly (it IS the same fitted function, just re-evaluated) -- it confirms
the pairwise functions are wired correctly through the additive-sum code
path. It does **not** validate the additive COMBINATION of 2+ simultaneous
moving fingers, since no recording here ever moved more than one finger.
See the per-finger summary below for that gap.

## Per-pair fit quality and effect size

{n_pairs} (affected, source) finger pairs, each averaged over 15 raw
channels. "Mean R²" only averages channels with detectable signal
(non-flat target); channels with no real variation are excluded from that
average and counted in parentheses, since R² is not a meaningful
statistic for a flat target. "Mean LOOCV MAE" leaves out one of the 4
non-zero measured positions at a time, refits on the rest, and predicts
the held-out point -- a genuine held-out check (not in-sample), though
noisy with only 4 points per channel.

{format_pair_table(model)}

"Effect ch." / "Max│Δ│" is the single raw channel (of 15) with the
largest measured shift for that pair -- a conservative one-number summary
of that pair's overall effect size. "Noise σ (P=0)" is that same
channel's within-recording standard deviation during the P=0 (reference)
capture, i.e. pure sensor/idle noise with no crosstalk. SNR = Max│Δ│
/ Noise σ; pairs with SNR ≥ {cc.SIGNIFICANT_SNR_THRESHOLD:.1f} are
flagged "Significant" -- a simple screening heuristic, not a formal
significance test.

## Additivity-assumption safety, per affected finger

{additivity_summary(model)}
"""

    cc.REPORT_MD.write_text(report, encoding="utf-8")
    print(f"Wrote report -> {cc.REPORT_MD}")
    print(f"Pointwise additive-prediction check: n={pointwise['n']}  "
          f"MAE={pointwise['mae']:.2f}  max={pointwise['max']:.2f}")
    # The report itself contains non-ASCII characters (sigma, superscript 2,
    # box-drawing bars) that Windows' default cp1252 console encoding can't
    # represent -- dumping it straight to stdout raises UnicodeEncodeError
    # on a plain `python script.py` run there. The file (already written
    # above, as UTF-8) is the actual deliverable; this is a best-effort
    # console echo only, so encoding failures here are swallowed rather
    # than crashing a run that already succeeded.
    try:
        print()
        print(report)
    except UnicodeEncodeError:
        print(f"(Report contains characters this console can't display -- see {cc.REPORT_MD})")


if __name__ == "__main__":
    main()
