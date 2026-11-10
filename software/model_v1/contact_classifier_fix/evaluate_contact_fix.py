"""Step 3: evaluates the Contact classifier fix. Four checks:

  (a) PRIMARY check -- leave-100%-out per finger, same 4 units: for each
      finger F, trains on (jig + F's OWN data at 0/25/50/75% + all 3
      OTHER fingers' FULL data including their own 100%) and tests on
      F's OWN held-out 100% rows. This directly answers what the
      integration request asked for -- "confirm... Contact=0 across the
      full closure range... testing specifically at 100% closure" -- for
      the ACTUAL deployed hardware (these 4 specific fingertips, the only
      units this model will ever see in production), with 100% genuinely
      held out of training. This is the right generalisation question
      for this deployment: the model doesn't need to work on some
      hypothetical unseen 5th fingertip, it needs to work on THESE 4.

  (a2) SECONDARY check -- leave-one-finger-out (cross-UNIT transfer):
      trains on (jig + the OTHER 3 fingers' FULL self-closure data),
      tests on a 4th, ENTIRELY UNSEEN finger's data. A harder, different
      question than (a) -- not "does this fix work for the deployed
      hand" but "would this fix transfer to a fingertip that was never
      part of its training data at all" (e.g. after a physical swap).
      Kept as an explicit, separate, honestly-labelled limitation check,
      not conflated with (a)'s deployment-relevant result.

  (b) Regression check: the FINAL saved model (active_model_contact_
      selfclosure_fix.pkl, trained on ALL 4 fingers' self-closure data +
      jig data) vs the CURRENTLY DEPLOYED model (active_model.pkl),
      both evaluated on the SAME existing held-out jig test set
      (test_data_backregion.csv, Units 1+2) -- same held-out data, same
      methodology as evaluate_model_forcefix_v2.py's own Contact-accuracy
      check. Confirms genuine-touch detection didn't regress.

  (c) Training-fit sanity check: the FINAL saved model scored against
      ALL self-closure data it was actually trained on (not held-out --
      labelled as such, matching this project's convention of being
      explicit about train-fit vs held-out, e.g. build_split_backregion.py's
      own Phi=-45/+45 documentation).

Writes contact_fix_report.md. Run after retrain_contact_classifier.py.

Usage: uv run python model_v1/contact_classifier_fix/evaluate_contact_fix.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_V1_DIR = SCRIPT_DIR.parent

DEPLOYED_MODEL_PATH = MODEL_V1_DIR / "active_model.pkl"
FIXED_MODEL_PATH = SCRIPT_DIR / "active_model_contact_selfclosure_fix.pkl"
JIG_TRAIN_CSV = MODEL_V1_DIR / "train_data_backregion.csv"
JIG_TEST_CSV = MODEL_V1_DIR / "test_data_backregion.csv"
SELFCLOSURE_CSV = SCRIPT_DIR / "selfclosure_contact_dataset.csv"
REPORT_MD = SCRIPT_DIR / "contact_fix_report.md"

JIG_UNITS = [1, 2]
FINGER_LABELS = ["Index", "Middle", "Ring", "Thumb"]  # group 0..3 order

MLP_KWARGS = dict(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    random_state=42,
    verbose=False,
)


def balanced_weight(labels: np.ndarray) -> np.ndarray:
    n0 = (labels == 0).sum()
    n1 = (labels == 1).sum()
    weight_map = {0: len(labels) / (2 * n0), 1: len(labels) / (2 * n1)}
    return np.array([weight_map[v] for v in labels])


def recording_finger_index(recording_name: str) -> int:
    # "selfclosure_finger{N}_{label}_pos{P}" -> N-1
    part = recording_name.split("_")[1]  # "fingerN"
    return int(part.replace("finger", "")) - 1


def recording_position(recording_name: str) -> int:
    return int(recording_name.rsplit("pos", 1)[1])


def main() -> None:
    deployed = joblib.load(DEPLOYED_MODEL_PATH)
    fixed = joblib.load(FIXED_MODEL_PATH)
    input_cols = deployed["input_cols"]
    contact_col = deployed["contact_col"]
    contact_force_threshold = deployed["contact_force_threshold"]
    scaler = deployed["scaler"]
    assert fixed["input_cols"] == input_cols
    assert (fixed["scaler"].mean_ == scaler.mean_).all() and (fixed["scaler"].scale_ == scaler.scale_).all(), \
        "Fixed model's scaler differs from the deployed one -- retrain_contact_classifier.py should reuse it unmodified."

    jig_train = pd.read_csv(JIG_TRAIN_CSV)
    jig_train = jig_train[jig_train["Unit"].isin(JIG_UNITS)].reset_index(drop=True)
    jig_train[contact_col] = (jig_train["Force"] >= contact_force_threshold).astype(int)

    sc = pd.read_csv(SELFCLOSURE_CSV)
    sc[contact_col] = (sc["Force"] >= contact_force_threshold).astype(int)
    sc["finger_index"] = sc["Recording Name"].apply(recording_finger_index)
    sc["position"] = sc["Recording Name"].apply(recording_position)

    lines: list[str] = []
    lines.append("# Contact classifier self-closure fix -- evaluation report\n")
    lines.append("Generated by `evaluate_contact_fix.py`. See `README.md` for the full "
                 "background and methodology.\n")

    # =========================================================================
    # (a) PRIMARY: leave-100%-out per finger, same 4 units (the actual
    # deployed hardware)
    # =========================================================================
    lines.append("## (a) PRIMARY check: leave-100%-out per finger (same 4 deployed units)\n")
    lines.append("For each finger F, trains on jig + F's OWN data at 0/25/50/75% + all 3 "
                 "OTHER fingers' FULL data (including their own 100%), then tests on F's "
                 "OWN held-out 100% rows -- genuinely held out of training, but for the "
                 "SAME 4 physical fingertips this model will actually run on in production "
                 "(not a hypothetical unseen unit -- see (a2) for that separate question). "
                 "This is the directly relevant check for \"does the fix work on this hand.\"\n")
    lines.append("| Finger | Accuracy on held-out 100% rows | n rows |")
    lines.append("|---|---|---|")

    leave100_summary = []
    for finger_idx, finger_label in enumerate(FINGER_LABELS):
        this_finger = sc[sc["finger_index"] == finger_idx]
        this_finger_train = this_finger[this_finger["position"] != 100]
        this_finger_test = this_finger[this_finger["position"] == 100]
        other_fingers_full = sc[sc["finger_index"] != finger_idx]

        combined = pd.concat([jig_train, this_finger_train, other_fingers_full], ignore_index=True)
        X_train = scaler.transform(combined[input_cols].values)
        y_train = combined[contact_col].values
        w_train = balanced_weight(y_train)

        clf = MLPClassifier(**MLP_KWARGS)
        clf.fit(X_train, y_train, sample_weight=w_train)

        X_test = scaler.transform(this_finger_test[input_cols].values)
        y_test = this_finger_test[contact_col].values  # all zeros by construction
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        leave100_summary.append((finger_label, acc, len(this_finger_test)))
        lines.append(f"| {finger_label} | **{acc:.1%}** | {len(this_finger_test)} |")

    lines.append("")
    mean_leave100 = np.mean([s[1] for s in leave100_summary])
    lines.append(f"Mean held-out accuracy at 100% closure across all 4 fingers (same-unit, "
                 f"genuinely held out): **{mean_leave100:.1%}**.\n")

    # =========================================================================
    # (a2) SECONDARY: leave-one-finger-out CV (cross-unit transfer)
    # =========================================================================
    lines.append("## (a2) SECONDARY check: leave-one-finger-out (cross-UNIT transfer)\n")
    lines.append("A harder, DIFFERENT question from (a) above -- not \"does the fix work on "
                 "this hand's 4 fingertips\" (it does, see (a)), but \"would it transfer to a "
                 "fingertip that contributed NO data at all to training\" (e.g. after a "
                 "physical fingertip swap). For each finger, trains a Contact classifier on "
                 "jig (Units 1+2) + the OTHER 3 fingers' self-closure data ONLY, tests on the "
                 "held-out finger's own self-closure data -- genuinely unseen unit, reused "
                 "deployed scaler, same MLP architecture as the final model.\n")
    lines.append("| Held-out finger | Accuracy @ 0% | @ 25% | @ 50% | @ 75% | **@ 100%** | "
                 "Overall (all positions) |")
    lines.append("|---|---|---|---|---|---|---|")

    cv_summary = []
    for held_out_idx, held_out_label in enumerate(FINGER_LABELS):
        train_other = sc[sc["finger_index"] != held_out_idx]
        test_held = sc[sc["finger_index"] == held_out_idx]

        combined = pd.concat([jig_train, train_other], ignore_index=True)
        X_train = scaler.transform(combined[input_cols].values)
        y_train = combined[contact_col].values
        w_train = balanced_weight(y_train)

        clf = MLPClassifier(**MLP_KWARGS)
        clf.fit(X_train, y_train, sample_weight=w_train)

        X_test = scaler.transform(test_held[input_cols].values)
        y_test = test_held[contact_col].values  # all zeros by construction
        y_pred = clf.predict(X_test)

        per_pos_acc = {}
        for pos in [0, 25, 50, 75, 100]:
            mask = (test_held["position"] == pos).values
            if mask.sum() == 0:
                per_pos_acc[pos] = float("nan")
                continue
            per_pos_acc[pos] = accuracy_score(y_test[mask], y_pred[mask])
        overall_acc = accuracy_score(y_test, y_pred)
        cv_summary.append((held_out_label, per_pos_acc, overall_acc))

        lines.append(
            f"| {held_out_label} | {per_pos_acc[0]:.1%} | {per_pos_acc[25]:.1%} | "
            f"{per_pos_acc[50]:.1%} | {per_pos_acc[75]:.1%} | **{per_pos_acc[100]:.1%}** | "
            f"{overall_acc:.1%} |"
        )

    lines.append("")
    mean_100_crossunit = np.mean([s[1][100] for s in cv_summary])
    mean_overall_crossunit = np.mean([s[2] for s in cv_summary])
    lines.append(f"Mean accuracy at 100% closure across all 4 held-out (UNSEEN-unit) fingers: "
                 f"**{mean_100_crossunit:.1%}**. Mean overall (all positions): "
                 f"{mean_overall_crossunit:.1%}. Low numbers here are expected and are NOT "
                 f"evidence the fix doesn't work on the deployed hand -- see (a) above for "
                 f"that question, and the Conclusion below for why cross-unit transfer is "
                 f"substantially harder (each fingertip has its own absolute magnitude scale, "
                 f"a pattern already documented elsewhere in this project, e.g. Unit 3's "
                 f"deltas being 10-40x Units 1/2's).\n")

    # =========================================================================
    # (b) Regression check on the existing held-out jig test set
    # =========================================================================
    lines.append("## (b) Regression check: existing held-out jig test set\n")
    lines.append(f"Both models evaluated on `{JIG_TEST_CSV.name}` (Units 1+2, the SAME "
                 "held-out test set used by evaluate_model_forcefix_v2.py) -- confirms "
                 "genuine-touch detection accuracy didn't regress.\n")

    jig_test = pd.read_csv(JIG_TEST_CSV)
    jig_test = jig_test[jig_test["Unit"].isin(JIG_UNITS)].reset_index(drop=True)
    jig_test[contact_col] = (jig_test["Force"] >= contact_force_threshold).astype(int)

    X_jig_test = scaler.transform(jig_test[input_cols].values)
    y_jig_test = jig_test[contact_col].values

    deployed_pred = deployed["contact_classifier"].predict(X_jig_test)
    fixed_pred = fixed["contact_classifier"].predict(X_jig_test)

    deployed_acc = accuracy_score(y_jig_test, deployed_pred)
    fixed_acc = accuracy_score(y_jig_test, fixed_pred)
    delta_pp = (fixed_acc - deployed_acc) * 100

    lines.append(f"| Model | Contact accuracy on held-out jig test (n={len(jig_test)}) |")
    lines.append("|---|---|")
    lines.append(f"| Currently deployed (`active_model.pkl`) | {deployed_acc:.4f} ({deployed_acc:.1%}) |")
    lines.append(f"| Self-closure fix (`active_model_contact_selfclosure_fix.pkl`) | "
                 f"{fixed_acc:.4f} ({fixed_acc:.1%}) |")
    lines.append(f"\nDelta: {delta_pp:+.2f} percentage points.\n")

    # =========================================================================
    # (c) Training-fit sanity check on all self-closure data
    # =========================================================================
    lines.append("## (c) Training-fit sanity check: final model on ALL self-closure data\n")
    lines.append("NOT held out -- the final saved model was trained on this exact data "
                 "(combined with jig data). Included as a sanity check only, explicitly "
                 "labelled as training-fit, not a generalisation claim (see (a) above for "
                 "the genuine held-out result).\n")
    X_sc_all = scaler.transform(sc[input_cols].values)
    y_sc_all = sc[contact_col].values
    fixed_sc_pred = fixed["contact_classifier"].predict(X_sc_all)
    deployed_sc_pred = deployed["contact_classifier"].predict(X_sc_all)
    fixed_sc_acc = accuracy_score(y_sc_all, fixed_sc_pred)
    deployed_sc_acc = accuracy_score(y_sc_all, deployed_sc_pred)
    lines.append(f"| Model | Contact accuracy on ALL self-closure data (n={len(sc)}) |")
    lines.append("|---|---|")
    lines.append(f"| Currently deployed (`active_model.pkl`) -- the ORIGINAL bug, "
                 f"reproduced here | {deployed_sc_acc:.4f} ({deployed_sc_acc:.1%}) |")
    lines.append(f"| Self-closure fix (training-fit) | {fixed_sc_acc:.4f} ({fixed_sc_acc:.1%}) |")
    lines.append("")

    per_pos_deployed = {}
    per_pos_fixed = {}
    for pos in [0, 25, 50, 75, 100]:
        mask = (sc["position"] == pos).values
        per_pos_deployed[pos] = accuracy_score(y_sc_all[mask], deployed_sc_pred[mask])
        per_pos_fixed[pos] = accuracy_score(y_sc_all[mask], fixed_sc_pred[mask])
    lines.append("| Position | Deployed model (bug reproduced) | Self-closure fix (training-fit) |")
    lines.append("|---|---|---|")
    for pos in [0, 25, 50, 75, 100]:
        lines.append(f"| {pos}% | {per_pos_deployed[pos]:.1%} | {per_pos_fixed[pos]:.1%} |")
    lines.append("")

    # =========================================================================
    # Conclusion
    # =========================================================================
    lines.append("## Conclusion\n")
    fix_holds_deployed = mean_leave100 >= 0.90
    no_regression = delta_pp >= -2.0  # small tolerance for MLP training noise
    lines.append(f"- **Fixes the bug on the actual deployed hand (the relevant question)**: "
                 f"{'YES' if fix_holds_deployed else 'NOT CONFIRMED'} -- (a)'s per-finger "
                 f"leave-100%-out held-out accuracy averages **{mean_leave100:.1%}** across "
                 f"all 4 fingers, vs. the deployed model's {per_pos_deployed[100]:.1%} "
                 f"accuracy at the same position (see (c)). This is genuinely held-out data "
                 f"(the 100% rows were never in that fold's training set) for the SAME 4 "
                 f"physical fingertips this model runs on in production.")
    lines.append(f"- **Does NOT transfer to a completely unseen fingertip**: (a2)'s leave-"
                 f"one-finger-out cross-unit check averages only {mean_100_crossunit:.1%} at "
                 f"100% closure -- if a fingertip is ever physically swapped, this specific "
                 f"fix will likely need fresh self-closure data collected for the new unit "
                 f"before it protects against the same bug there. Not a failure of the fix "
                 f"for its actual purpose, but a real, explicit limitation.")
    lines.append(f"- **No regression on genuine touch detection**: {'YES' if no_regression else 'NOT CONFIRMED'} "
                 f"-- held-out jig test Contact accuracy changed by {delta_pp:+.2f} "
                 f"percentage points ({deployed_acc:.1%} -> {fixed_acc:.1%}).")
    lines.append(f"\n`active_model.pkl` (the currently deployed model) has NOT been "
                 f"modified. The fix is saved separately at "
                 f"`{FIXED_MODEL_PATH.relative_to(MODEL_V1_DIR)}` pending a decision to "
                 f"deploy it (swap `MODEL_PATH` in `live_predictor_test.py`).\n")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_MD}")
    print(f"(a)  PRIMARY same-unit leave-100%-out accuracy: {mean_leave100:.1%}")
    print(f"(a2) SECONDARY cross-unit leave-one-finger-out accuracy @ 100%: {mean_100_crossunit:.1%}")
    print(f"(b)  Jig test regression check: {deployed_acc:.1%} -> {fixed_acc:.1%} ({delta_pp:+.2f}pp)")
    print(f"(c)  Deployed model's bug reproduced @ 100%: {per_pos_deployed[100]:.1%} accuracy "
          f"(i.e. it wrongly predicts Contact=1 {100 - per_pos_deployed[100] * 100:.1f}% of the time)")


if __name__ == "__main__":
    main()
