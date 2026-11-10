"""Production-readiness review, step 4: confidence calibration.

Does the deployed model know when it doesn't know? Step 1's baseline report
says no: Contact ECE 0.174 (rows with P(contact)<0.1 are actually 53%
contact), Phi accuracy at 95-99% confidence is ~51%.

This step fits post-hoc calibrators (isotonic regression) and quantifies
before/after ECE on held-out data:

  - Contact: isotonic on P(contact). Fit on train-set probabilities
    (caveat: the MLP saw those rows during training, so its train probs are
    mildly optimistic -- stated, not hidden), evaluated on the untouched
    test set, per unit.
  - Phi: isotonic mapping max-prob -> P(correct). Same fit/eval protocol.

Also evaluates the practical live-pipeline lever: replacing the hard
predict() in the Contact hysteresis with a calibrated-probability Schmitt
trigger (confirm above P_hi, release below P_lo), scored on idle/touch
error rates at several threshold pairs.

Read-only outside production_review/. Saves calibrators to
production_review/candidate_calibrators.pkl (not deployed).
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.isotonic import IsotonicRegression

TRAIN_CSV = "model_v1/train_data_backregion.csv"
TEST_CSV = "model_v1/test_data_backregion.csv"
MODEL_PATH = "model_v1/active_model.pkl"
OUT_MD = "model_v1/production_review/04_calibration_report.md"
OUT_PKL = "model_v1/production_review/candidate_calibrators.pkl"
THR = 0.3

bundle = joblib.load(MODEL_PATH)
scaler = bundle["scaler"]
contact_clf = bundle["contact_classifier"]
phi_clf = bundle["classifier"]
INPUT_COLS = bundle["input_cols"]

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
for df in (train, test):
    df["Contact"] = (df.Force >= THR).astype(int)

c1 = list(contact_clf.classes_).index(1)


def contact_p1(df):
    X = scaler.transform(df[INPUT_COLS].values)
    return contact_clf.predict_proba(X)[:, c1]


def phi_conf_correct(df):
    X = scaler.transform(df[INPUT_COLS].values)
    proba = phi_clf.predict_proba(X)
    pred = phi_clf.classes_[proba.argmax(axis=1)]
    return proba.max(axis=1), (pred == df.Phi.values).astype(int)


def ece(p, y, nbins=10):
    bins = np.linspace(0, 1, nbins + 1)
    total = 0.0
    for i in range(nbins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < nbins - 1 else p <= 1.0)
        if m.sum() == 0:
            continue
        total += m.mean() * abs(p[m].mean() - y[m].mean())
    return total


lines = ["# Confidence calibration report", ""]
add = lines.append

# ------------------------------------------------------------- Contact
# Fit on Units 1+2 ONLY: including Unit 3 poisons the low-probability
# region (Unit 3's genuine-contact rows all get near-zero raw P, so pooled
# isotonic pulls the whole low end up to ~0.30 -- a floor above any usable
# release threshold, i.e. confirmed contact could never release; caught in
# review, see report note).
tr12 = train[train.Unit.isin([1, 2])]
p_tr = contact_p1(tr12)
p_te = contact_p1(test)
iso_contact = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso_contact.fit(p_tr, tr12.Contact.values)
p_te_cal = iso_contact.predict(p_te)

add("## Contact classifier")
add("")
add("Isotonic calibrator fit on train-set P(contact) (caveat: train probs "
    "are mildly optimistic since the MLP saw those rows), evaluated on the "
    "held-out test set.")
add("")
add("| Slice | n | ECE raw | ECE calibrated |")
add("|---|---|---|---|")
for label, m in [("All test", np.ones(len(test), bool)),
                 ("Units 1+2", test.Unit.isin([1, 2]).values),
                 ("Unit 3", (test.Unit == 3).values),
                 ("theta=90", (test.Theta == 90).values)]:
    add(f"| {label} | {m.sum()} | {ece(p_te[m], test.Contact.values[m]):.4f} | "
        f"{ece(p_te_cal[m], test.Contact.values[m]):.4f} |")
add("")

# Schmitt-trigger policy comparison (per-frame proxy: what fraction of idle
# frames sit above the confirm threshold / touch frames below the release
# threshold -- the streak logic then multiplies these error rates down).
add("### Probability-Schmitt policy (Units 1+2 rows)")
add("")
add("Current live rule: hard predict()==1 AND Force>0.1N counts toward the "
    "confirm streak. Candidate rule: calibrated P(contact) >= P_hi to "
    "confirm, <= P_lo to release.")
add("")
u12 = test.Unit.isin([1, 2]).values
idle = u12 & (test.Contact.values == 0)
touch = u12 & (test.Contact.values == 1)
X12 = scaler.transform(test[INPUT_COLS].values)
force_pred = bundle["regressor"].predict(X12)[:, 0]
hard = contact_clf.predict(X12)
cur_confirm_sig = (hard == 1) & (force_pred > 0.1)
add("Three health metrics per policy: idle-confirm rate (drives FALSE "
    "contact -- lower is better), touch-release rate (drives dropped/"
    "flickering contact during a real press -- lower is better), and "
    "idle-release rate (must be HIGH, or a once-confirmed contact can "
    "never release).")
add("")
add(f"- Current rule: idle-confirm {cur_confirm_sig[idle].mean():.4f}; "
    f"touch-release {(~cur_confirm_sig[touch]).mean():.4f}; "
    f"idle-release {(~cur_confirm_sig[idle]).mean():.4f}")
for tag, probs in [("calibrated", p_te_cal), ("raw", p_te)]:
    for p_hi, p_lo in [(0.6, 0.3), (0.7, 0.3), (0.8, 0.2), (0.8, 0.3), (0.9, 0.5)]:
        conf = (probs >= p_hi) & (force_pred > 0.1)
        rel = probs <= p_lo
        add(f"- Schmitt({tag}) P_hi={p_hi}/P_lo={p_lo}: idle-confirm "
            f"{conf[idle].mean():.4f}; touch-release {rel[touch].mean():.4f}; "
            f"idle-release {rel[idle].mean():.4f}")
add("")

# ------------------------------------------------------------- Phi
conf_tr, corr_tr = phi_conf_correct(train[train.Unit.isin([1, 2])])
conf_te, corr_te = phi_conf_correct(test)
iso_phi = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso_phi.fit(conf_tr, corr_tr)
conf_te_cal = iso_phi.predict(conf_te)

add("## Phi classifier (confidence -> P(correct))")
add("")
add("| Slice | n | ECE raw | ECE calibrated |")
add("|---|---|---|---|")
for label, m in [("All test", np.ones(len(test), bool)),
                 ("Units 1+2, theta>=30", (test.Unit.isin([1, 2]) & (test.Theta >= 30)).values),
                 ("theta=90", (test.Theta == 90).values),
                 ("theta<30", (test.Theta < 30).values)]:
    add(f"| {label} | {m.sum()} | {ece(conf_te[m], corr_te[m]):.4f} | "
        f"{ece(conf_te_cal[m], corr_te[m]):.4f} |")
add("")
add("Interpretation note: even a perfect calibrator cannot fix Phi at "
    "theta<30 -- the azimuth is physically degenerate there and the honest "
    "output is a LOW confidence, which is what calibration should produce.")
add("")

joblib.dump({"contact_isotonic": iso_contact, "phi_conf_isotonic": iso_phi,
             "fitted_against": "active_model.pkl train-set probabilities"},
            OUT_PKL)
add(f"Calibrators saved to {OUT_PKL} (not deployed).")

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))
print(f"Wrote {OUT_MD}")
