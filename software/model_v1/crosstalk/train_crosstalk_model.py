"""Step 2/3: fits the additive pairwise crosstalk model from
crosstalk_dataset.csv and writes crosstalk_model_v1.json.

For each (affected finger, source finger, raw channel) triple, fits a
polynomial-through-origin (see crosstalk_model.py) predicting that
channel's baseline SHIFT (delta from the source's own P=0 reference) as a
function of the source finger's position. Degree (1 vs 2) is chosen per
channel by leave-one-out cross-validation over the 4 non-zero measured
positions {25,50,75,100}: whichever degree has the lower LOOCV mean
absolute error wins; ties favour degree 1 (simpler, less prone to
overfitting a 2-parameter model to 4 points). This mirrors model_v1's own
precedent of an auto-decision rule that documents its own choice (see
train_model_forcefix.py's split-head winner selection) rather than a
single hand-picked degree applied everywhere regardless of fit.

Also computes, per (affected, source) pair, an effect-size-vs-noise-floor
signal-to-noise ratio (SNR) and a "significant" flag -- see
crosstalk_common.SIGNIFICANT_SNR_THRESHOLD and README.md's additive-
assumption section for what this flag means and doesn't mean.

Run after build_crosstalk_dataset.py.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import crosstalk_common as cc
import crosstalk_model as cm


def load_dataset() -> dict[tuple[str, str, str], dict[int, dict]]:
    """Returns {(source, affected, channel): {position: {mean, std, delta}}}."""
    data: dict[tuple[str, str, str], dict[int, dict]] = defaultdict(dict)
    with open(cc.DATASET_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["source_label"], row["affected_label"], row["channel"])
            pos = int(row["position_percent"])
            data[key][pos] = {
                "mean": float(row["mean_value"]),
                "std": float(row["std_value"]),
                "delta": float(row["delta_from_p0"]),
            }
    return data


def loocv_mae(positions: list[int], values: list[float], degree: int) -> float:
    """Leave-one-out over the 4 non-zero measured positions: fit on the
    other 3 (through origin -- the implicit (0, 0) point needs no separate
    inclusion, see crosstalk_model.fit_through_origin), predict the held-
    out point, average the absolute residual across all 4 folds. A
    genuine held-out check (not in-sample), though noisy with only 4
    points per channel -- see README.md's limitations section."""
    residuals = []
    for i in range(len(positions)):
        train_p = positions[:i] + positions[i + 1:]
        train_y = values[:i] + values[i + 1:]
        coeffs = cm.fit_through_origin(train_p, train_y, degree)
        pred = cm.predict_polynomial(coeffs, positions[i])
        residuals.append(abs(pred - values[i]))
    return sum(residuals) / len(residuals)


def in_sample_r2(positions: list[int], values: list[float], coeffs: list[float]) -> float | None:
    """R^2 over all 5 measured points (0,25,50,75,100), including the
    trivial (0, 0) point. None if the true values have ~zero spread
    (flat/no real signal on this channel), where R^2 is not a meaningful
    or numerically stable statistic (near-zero denominator)."""
    all_p = [0] + positions
    all_y = [0.0] + values
    y_mean = sum(all_y) / len(all_y)
    ss_tot = sum((y - y_mean) ** 2 for y in all_y)
    if ss_tot < 1e-9:
        return None
    ss_res = sum((y - cm.predict_polynomial(coeffs, p)) ** 2 for p, y in zip(all_p, all_y))
    return 1.0 - ss_res / ss_tot


def fit_pair_channel(pos_data: dict[int, dict]) -> dict:
    positions = cc.NONZERO_POSITIONS
    values = [pos_data[p]["delta"] for p in positions]

    mae_by_degree = {d: loocv_mae(positions, values, d) for d in (1, 2)}
    degree = min(mae_by_degree, key=lambda d: (mae_by_degree[d], d))
    coeffs = cm.fit_through_origin(positions, values, degree)
    r2 = in_sample_r2(positions, values, coeffs)

    return {
        "degree": degree,
        "coeffs": coeffs,
        "loocv_mae": mae_by_degree[degree],
        "loocv_mae_degree1": mae_by_degree[1],
        "loocv_mae_degree2": mae_by_degree[2],
        "in_sample_r2": r2,
        "max_abs_delta": max(abs(v) for v in values),
        "noise_floor_std_at_p0": pos_data[0]["std"],
    }


def main() -> None:
    dataset = load_dataset()

    pairs: dict[str, dict[str, dict]] = {affected: {} for affected in cc.FINGER_LABELS}

    for affected in cc.FINGER_LABELS:
        for source in cc.FINGER_LABELS:
            if source == affected:
                continue
            channel_fits = {}
            for channel in cc.RAW_COLS:
                key = (source, affected, channel)
                if key not in dataset:
                    raise SystemExit(f"Missing data for {key} -- rerun build_crosstalk_dataset.py")
                channel_fits[channel] = fit_pair_channel(dataset[key])

            max_abs = max(cf["max_abs_delta"] for cf in channel_fits.values())
            effect_channel = max(channel_fits, key=lambda c: channel_fits[c]["max_abs_delta"])
            noise = channel_fits[effect_channel]["noise_floor_std_at_p0"]
            snr = max_abs / noise if noise > 1e-9 else float("inf")

            pairs[affected][source] = {
                "channels": channel_fits,
                "effect_size_max_abs_delta": max_abs,
                "effect_channel": effect_channel,
                "noise_floor_std_at_p0": noise,
                "snr": snr,
                "significant": snr >= cc.SIGNIFICANT_SNR_THRESHOLD,
            }

    model = {
        "model_type": "additive_pairwise_polynomial_through_origin",
        "created_at_iso": datetime.now(timezone.utc).isoformat(),
        "channels": cc.RAW_COLS,
        "positions_used": cc.POSITIONS,
        "significant_snr_threshold": cc.SIGNIFICANT_SNR_THRESHOLD,
        "reference_definition": (
            "delta = mean(channel | source finger at position P, idle capture) "
            "- mean(channel | source finger at P=0), computed within the SAME "
            "single-finger sweep (see record_crosstalk.py) -- the affected "
            "finger and the other 2 non-source fingers were held at the "
            "reference (0%, open) position throughout, idle, no contact."
        ),
        "additive_assumption": (
            "Total predicted baseline shift for finger F, given multiple "
            "other fingers moving simultaneously, is the SUM of each "
            "finger-pair's fitted function evaluated independently. NOT "
            "validated against real simultaneous-movement data -- every "
            "recording used to fit this model moved exactly one finger at a "
            "time. See crosstalk_model_report.md's per-pair 'significant' "
            "flag and README.md for where this assumption is least safe."
        ),
        "pairs": pairs,
    }

    cm.save_model(model, cc.MODEL_JSON)
    n_pairs = len(cc.FINGER_LABELS) * (len(cc.FINGER_LABELS) - 1)
    print(f"Fitted {n_pairs} finger pairs x {len(cc.RAW_COLS)} channels -> {cc.MODEL_JSON}")


if __name__ == "__main__":
    main()
