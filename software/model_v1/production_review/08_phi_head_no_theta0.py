"""Production-readiness review, step 8: candidate Phi head retrained
WITHOUT theta=0 rows.

At theta=0 (straight-down press) the azimuth Phi is physically undefined --
the stimulus is identical whatever the recording's nominal Phi label says.
Those rows are therefore label noise for the Phi classifier (baseline
report: Phi accuracy at theta=0 is 15%, i.e. below the 33% chance floor --
the classifier is actively misled there). 02's GBM variant showed dropping
them helps GBM Phi; this tests the same cut on the DEPLOYED architecture
(MLPClassifier, identical hyperparameters, deployed scaler reused
unmodified -- the exact single-head-swap mechanism contact_classifier_fix
already validated).

Evaluated on the untouched test set: overall Phi acc, U1+2 theta>=30 acc
(the physically-meaningful regime), per-theta breakdown, and confidence
calibration in the meaningful regime.

Read-only outside production_review/. Saves the candidate head to
candidate_phi_no_theta0.pkl (not deployed).
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

TRAIN_CSV = "model_v1/train_data_backregion.csv"
TEST_CSV = "model_v1/test_data_backregion.csv"
MODEL_PATH = "model_v1/active_model.pkl"
OUT_MD = "model_v1/production_review/08_phi_head_report.md"
OUT_PKL = "model_v1/production_review/candidate_phi_no_theta0.pkl"
THR = 0.3

MLP_KWARGS = dict(hidden_layer_sizes=(64, 32, 16), activation="relu",
                  solver="adam", alpha=1e-4, learning_rate_init=1e-3,
                  max_iter=500, early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=15, random_state=42, verbose=False)

bundle = joblib.load(MODEL_PATH)
scaler = bundle["scaler"]
INPUT_COLS = bundle["input_cols"]

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
for df in (train, test):
    df["Contact"] = (df.Force >= THR).astype(int)

jig = train[train.Unit.isin([1, 2])]
jig_nz = jig[jig.Theta != 0]
print(f"Jig rows: {len(jig)}, after dropping theta=0: {len(jig_nz)}")


def w_bal(y):
    n0 = (y == 0).sum(); n1 = (y == 1).sum()
    return np.where(y == 0, len(y) / (2 * n0), len(y) / (2 * n1))


X = scaler.transform(jig_nz[INPUT_COLS].values)
w = w_bal(jig_nz.Contact.values)
clf = MLPClassifier(**MLP_KWARGS)
clf.fit(X, jig_nz.Phi.values, sample_weight=w)

Xte = scaler.transform(test[INPUT_COLS].values)
pred_new = clf.predict(Xte)
proba_new = clf.predict_proba(Xte).max(axis=1)
pred_old = bundle["classifier"].predict(Xte)
proba_old = bundle["classifier"].predict_proba(Xte).max(axis=1)

lines = ["# Candidate Phi head: MLP retrained without theta=0 rows", ""]
add = lines.append
add("| Slice | n | Deployed Phi acc | Candidate Phi acc |")
add("|---|---|---|---|")
slices = [
    ("All test", np.ones(len(test), bool)),
    ("Units 1+2", test.Unit.isin([1, 2]).values),
    ("Units 1+2, theta>=30", (test.Unit.isin([1, 2]) & (test.Theta >= 30)).values),
    ("Units 1+2, theta in {15,30}", (test.Unit.isin([1, 2]) & test.Theta.isin([15, 30])).values),
    ("theta=90 (all units)", (test.Theta == 90).values),
    ("theta=0 (degenerate labels; expect ~chance for both)", (test.Theta == 0).values),
]
for label, m in slices:
    add(f"| {label} | {m.sum()} | {accuracy_score(test.Phi.values[m], pred_old[m]):.4f} | "
        f"{accuracy_score(test.Phi.values[m], pred_new[m]):.4f} |")
add("")

add("Per-theta accuracy (Units 1+2):")
add("")
add("| Theta | n | Deployed | Candidate |")
add("|---|---|---|---|")
u12 = test.Unit.isin([1, 2]).values
for th in sorted(test.Theta.unique()):
    m = u12 & (test.Theta == th).values
    if not m.sum():
        continue
    add(f"| {th} | {m.sum()} | {accuracy_score(test.Phi.values[m], pred_old[m]):.4f} | "
        f"{accuracy_score(test.Phi.values[m], pred_new[m]):.4f} |")
add("")

# Confidence usefulness in the meaningful regime: accuracy above confidence
# thresholds (what a GUI confidence gate would show).
add("Confidence gating (Units 1+2, theta>=30): accuracy and coverage above "
    "a max-prob threshold:")
add("")
add("| Threshold | Deployed acc @ coverage | Candidate acc @ coverage |")
add("|---|---|---|")
m = (test.Unit.isin([1, 2]) & (test.Theta >= 30)).values
for t in (0.5, 0.7, 0.9):
    mo = m & (proba_old >= t)
    mn = m & (proba_new >= t)
    add(f"| {t} | {accuracy_score(test.Phi.values[mo], pred_old[mo]):.4f} @ {mo.sum()/m.sum():.1%} | "
        f"{accuracy_score(test.Phi.values[mn], pred_new[mn]):.4f} @ {mn.sum()/m.sum():.1%} |")
add("")

joblib.dump({"phi_classifier_no_theta0": clf,
             "note": "MLP, deployed scaler reused, jig U1+2 rows with Theta!=0"},
            OUT_PKL)
add(f"Candidate head saved to {OUT_PKL} (not deployed).")

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))
print(f"Wrote {OUT_MD}")
