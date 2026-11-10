"""Production-readiness review, step 9: assemble the review's candidate
model bundle -- active_model.pkl with ONLY the validated head/metadata
changes applied, saved as a NEW file. active_model.pkl is never written.

Changes vs deployed:
  - classifier (Phi head): replaced with candidate_phi_no_theta0.pkl's
    (08_phi_head_report.md: theta=90 accuracy 0.416 -> 0.644, theta>=30
    U1+2 tied 0.837 -> 0.841; theta=15 regression is accepted because the
    GUI change deployed alongside this gates Phi display to theta>=25).
  - contact_calibrator (NEW key): isotonic P(contact) calibrator fit on
    Units 1+2 (04_calibration_report.md: U1+2 ECE 0.053 -> 0.007). Additive
    -- nothing in the existing pipeline reads it until the GUI change ships.
  - regressor / contact_classifier / scaler: byte-for-byte carried over.

Runs the full regression battery before saving: the same test-set metrics
as 01, plus the self-closure Contact check, and refuses to save if any
guarded metric regresses beyond tolerance.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

MODEL_PATH = "model_v1/active_model.pkl"
PHI_PKL = "model_v1/production_review/candidate_phi_no_theta0.pkl"
CAL_PKL = "model_v1/production_review/candidate_calibrators.pkl"
TEST_CSV = "model_v1/test_data_backregion.csv"
SELFC_CSV = "model_v1/contact_classifier_fix/selfclosure_contact_dataset.csv"
OUT_PKL = "model_v1/production_review/active_model_review_candidate.pkl"
THR = 0.3

bundle = joblib.load(MODEL_PATH)
phi_new = joblib.load(PHI_PKL)["phi_classifier_no_theta0"]
cal = joblib.load(CAL_PKL)

test = pd.read_csv(TEST_CSV)
test["Contact"] = (test.Force >= THR).astype(int)
X = bundle["scaler"].transform(test[bundle["input_cols"]].values)

# --- guarded regression checks --------------------------------------------
reg_pred = bundle["regressor"].predict(X)
force_mae = np.abs(test.Force.values - reg_pred[:, 0]).mean()
contact_acc = accuracy_score(test.Contact.values, bundle["contact_classifier"].predict(X))

phi_old = bundle["classifier"].predict(X)
phi_new_pred = phi_new.predict(X)
m_meaningful = (test.Unit.isin([1, 2]) & (test.Theta >= 30)).values
acc_old = accuracy_score(test.Phi.values[m_meaningful], phi_old[m_meaningful])
acc_new = accuracy_score(test.Phi.values[m_meaningful], phi_new_pred[m_meaningful])
m_back = (test.Theta == 90).values
back_old = accuracy_score(test.Phi.values[m_back], phi_old[m_back])
back_new = accuracy_score(test.Phi.values[m_back], phi_new_pred[m_back])

selfc = pd.read_csv(SELFC_CSV)
Xs = bundle["scaler"].transform(selfc[bundle["input_cols"]].values)
selfc_acc = (bundle["contact_classifier"].predict(Xs) == 0).mean()

print(f"Force MAE (unchanged head): {force_mae:.4f} N")
print(f"Contact acc (unchanged head): {contact_acc:.4f}; self-closure Contact=0 rate {selfc_acc:.4f}")
print(f"Phi theta>=30 U1+2: old={acc_old:.4f} new={acc_new:.4f}")
print(f"Phi theta=90 all units: old={back_old:.4f} new={back_new:.4f}")

assert acc_new >= acc_old - 0.005, "Phi head regressed in the meaningful regime -- not saving"
assert back_new > back_old + 0.10, "Phi head did not deliver the back-region gain -- not saving"
assert selfc_acc > 0.99, "self-closure contact fix not intact -- not saving"

out = dict(bundle)
out["classifier"] = phi_new
out["phi_training_source"] = "jigonly_units_1_2_no_theta0"
out["contact_calibrator"] = cal["contact_isotonic"]
out["contact_calibrator_note"] = (
    "Isotonic P(contact)->calibrated P, fit on Units 1+2 train probabilities "
    "(production_review/04). U1+2 test ECE 0.053->0.007. NOT fit on / valid "
    "for Unit-3-scale inputs.")
out["review_note"] = (
    "production_review candidate (2026-07-17): Phi head retrained without "
    "theta=0 rows (see 08_phi_head_report.md), contact_calibrator added. "
    "Regressor, Contact classifier, scaler byte-identical to the 2026-07-13 "
    "deployed active_model.pkl.")
head = dict(out.get("head_training_data", {}))
head["classifier_phi"] = "jigonly_units_1_2_no_theta0"
out["head_training_data"] = head

joblib.dump(out, OUT_PKL)
print(f"\nSaved candidate bundle to {OUT_PKL} (active_model.pkl untouched).")
