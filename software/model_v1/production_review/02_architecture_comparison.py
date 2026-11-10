"""Production-readiness review, step 2: architecture comparison.

Trains candidate replacements for each head of active_model.pkl on the SAME
train/test recording-level split (train_data_backregion.csv /
test_data_backregion.csv) and compares against the deployed model:

  - HistGradientBoosting (GBM) heads on RAW (unscaled) features. Trees are
    scale-invariant, so this architecture needs NO StandardScaler at all --
    it structurally cannot suffer the scaler-dilution failure class that
    produced forcefix/forcefix_v2, and per-unit scale differences (Unit 3's
    10-40x deltas) stop being a modelling hazard.
  - A Phi variant trained WITHOUT theta=0 rows: at theta=0 (straight-down
    press) the azimuth Phi is physically degenerate -- the stimulus is
    identical whatever the recording's nominal Phi label says -- so those
    rows are label noise for the classifier. Evaluated on all rows AND on
    the theta>=30 subset where Phi is physically meaningful.
  - Contact GBM trained with the same data recipe as the deployed fix
    (jig Units 1+2 + the self-closure dataset), so the self-closure bug fix
    is preserved by construction.

Additive/read-only: writes only candidate .pkl files inside
production_review/ and 02_architecture_report.md. active_model.pkl is never
touched.
"""
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

TRAIN_CSV = "model_v1/train_data_backregion.csv"
TEST_CSV = "model_v1/test_data_backregion.csv"
SELFCLOSURE_CSV = "model_v1/contact_classifier_fix/selfclosure_contact_dataset.csv"
MODEL_PATH = "model_v1/active_model.pkl"
OUT_MD = "model_v1/production_review/02_architecture_report.md"
OUT_PKL = "model_v1/production_review/candidate_gbm_heads.pkl"

INPUT_COLS = ['dS1X', 'dS1Y', 'dS1Z', 'dS2X', 'dS2Y', 'dS2Z', 'dS3X', 'dS3Y',
              'dS3Z', 'dS4X', 'dS4Y', 'dS4Z', 'dS5X', 'dS5Y', 'dS5Z',
              'S1_mag', 'S2_mag', 'S3_mag', 'S4_mag', 'S5_mag']
THR = 0.3
RANDOM_STATE = 42

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
train["Contact"] = (train.Force >= THR).astype(int)
test["Contact"] = (test.Force >= THR).astype(int)
selfc = pd.read_csv(SELFCLOSURE_CSV)
selfc["Contact"] = (selfc.Force >= THR).astype(int)  # all 0 by construction

Xtr = train[INPUT_COLS].values
Xte = test[INPUT_COLS].values


def contact_balanced_w(df):
    n0 = (df.Contact == 0).sum(); n1 = (df.Contact == 1).sum()
    return df.Contact.map({0: len(df) / (2 * n0), 1: len(df) / (2 * n1)}).values


# ---------------------------------------------------------------- deployed
bundle = joblib.load(MODEL_PATH)
Xte_s = bundle["scaler"].transform(Xte)
dep_reg = bundle["regressor"].predict(Xte_s)
dep_phi = bundle["classifier"].predict(Xte_s)
dep_contact = bundle["contact_classifier"].predict(Xte_s)

# ---------------------------------------------------------------- GBM heads
GBM_REG_KW = dict(max_iter=300, learning_rate=0.1, max_depth=None,
                  max_leaf_nodes=31, early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=20, random_state=RANDOM_STATE)
GBM_CLF_KW = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                  early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=20, random_state=RANDOM_STATE)

results = {}
timings = {}

w_reg = contact_balanced_w(train)

t0 = time.time()
gbm_force = HistGradientBoostingRegressor(**GBM_REG_KW)
gbm_force.fit(Xtr, train.Force.values, sample_weight=w_reg)
timings["gbm_force"] = time.time() - t0

t0 = time.time()
gbm_theta = HistGradientBoostingRegressor(**GBM_REG_KW)
gbm_theta.fit(Xtr, train.Theta.values, sample_weight=w_reg)
timings["gbm_theta"] = time.time() - t0

# Phi: jig-only Units 1+2, classes {-90,0,90} -- same recipe as deployed.
jig = train[train.Unit.isin([1, 2])]
w_jig = contact_balanced_w(jig)
t0 = time.time()
gbm_phi = HistGradientBoostingClassifier(**GBM_CLF_KW)
gbm_phi.fit(jig[INPUT_COLS].values, jig.Phi.values, sample_weight=w_jig)
timings["gbm_phi"] = time.time() - t0

# Phi variant: drop theta=0 rows (physically degenerate azimuth labels).
jig_nz = jig[jig.Theta != 0]
w_jig_nz = contact_balanced_w(jig_nz)
t0 = time.time()
gbm_phi_nz = HistGradientBoostingClassifier(**GBM_CLF_KW)
gbm_phi_nz.fit(jig_nz[INPUT_COLS].values, jig_nz.Phi.values, sample_weight=w_jig_nz)
timings["gbm_phi_nz"] = time.time() - t0

# Contact: jig Units 1+2 + self-closure rows (deployed fix's exact recipe).
contact_train = pd.concat([
    jig[INPUT_COLS + ["Contact"]],
    selfc[INPUT_COLS + ["Contact"]],
], ignore_index=True)
w_contact = contact_balanced_w(contact_train)
t0 = time.time()
gbm_contact = HistGradientBoostingClassifier(**GBM_CLF_KW)
gbm_contact.fit(contact_train[INPUT_COLS].values, contact_train.Contact.values,
                sample_weight=w_contact)
timings["gbm_contact"] = time.time() - t0

# ---------------------------------------------------------------- predict
gbm_force_pred = gbm_force.predict(Xte)
gbm_theta_pred = gbm_theta.predict(Xte)
gbm_phi_pred = gbm_phi.predict(Xte)
gbm_phi_nz_pred = gbm_phi_nz.predict(Xte)
gbm_contact_pred = gbm_contact.predict(Xte)

# Self-closure regression check: GBM contact on the self-closure dataset
# (training-fit, same caveat as contact_fix_report.md's check (c)) plus a
# proper leave-100%-out check per finger.
selfc_pred = gbm_contact.predict(selfc[INPUT_COLS].values)
selfc_acc = (selfc_pred == 0).mean()

# Finger/position are encoded in Recording Name:
# "selfclosure_finger{N}_{label}_pos{PPP}"
name_parts = selfc["Recording Name"].str.extract(r"selfclosure_(finger\d+_\w+?)_pos(\d+)")
selfc["_finger"] = name_parts[0]
selfc["_pos"] = name_parts[1].astype(float)
loo100 = {}
if selfc["_finger"].notna().all():
    for f in selfc["_finger"].unique():
        m_hold = (selfc["_finger"] == f) & (selfc["_pos"] == 100)
        tr2 = pd.concat([jig[INPUT_COLS + ["Contact"]],
                         selfc[~m_hold][INPUT_COLS + ["Contact"]]], ignore_index=True)
        w2 = contact_balanced_w(tr2)
        clf2 = HistGradientBoostingClassifier(**GBM_CLF_KW)
        clf2.fit(tr2[INPUT_COLS].values, tr2.Contact.values, sample_weight=w2)
        pred2 = clf2.predict(selfc[m_hold][INPUT_COLS].values)
        loo100[str(f)] = float((pred2 == 0).mean())

# ---------------------------------------------------------------- report
lines = ["# Architecture comparison: deployed MLP heads vs GBM heads", ""]
add = lines.append
add(f"Same split as evaluate_model_forcefix_v2.py: train={len(train)} rows, "
    f"test={len(test)} rows (12 held-out recordings). GBM heads use RAW "
    f"features -- no scaler exists in the GBM path at all.")
add("")
add("Training times: " + ", ".join(f"{k}={v:.1f}s" for k, v in timings.items()))
add("")


def block(mask, label):
    y_f = test.Force.values[mask]; y_t = test.Theta.values[mask]
    y_p = test.Phi.values[mask]; y_c = test.Contact.values[mask]
    rows = []
    rows.append(f"### {label} (n={mask.sum()})")
    rows.append("")
    rows.append("| Metric | Deployed (MLP) | GBM | GBM Phi (no theta=0) |")
    rows.append("|---|---|---|---|")
    rows.append(f"| Force MAE (N) | {np.abs(y_f - dep_reg[mask,0]).mean():.4f} | "
                f"{np.abs(y_f - gbm_force_pred[mask]).mean():.4f} | -- |")
    rows.append(f"| Theta MAE (deg) | {np.abs(y_t - dep_reg[mask,1]).mean():.4f} | "
                f"{np.abs(y_t - gbm_theta_pred[mask]).mean():.4f} | -- |")
    rows.append(f"| Phi acc | {accuracy_score(y_p, dep_phi[mask]):.4f} | "
                f"{accuracy_score(y_p, gbm_phi_pred[mask]):.4f} | "
                f"{accuracy_score(y_p, gbm_phi_nz_pred[mask]):.4f} |")
    rows.append(f"| Contact acc | {accuracy_score(y_c, dep_contact[mask]):.4f} | "
                f"{accuracy_score(y_c, gbm_contact_pred[mask]):.4f} | -- |")
    rows.append("")
    return rows


lines += block(np.ones(len(test), dtype=bool), "All test rows")
for u in (1, 2, 3):
    lines += block((test.Unit == u).values, f"Unit {u}")
lines += block((test.Theta == 90).values, "Back region (theta=90, all units)")
lines += block((test.Theta >= 30).values, "Theta >= 30 (Phi physically meaningful)")
lines += block((test.Unit.isin([1, 2]) & (test.Theta >= 30)).values,
               "Units 1+2, theta >= 30")

idle12 = (test.Unit.isin([1, 2]) & (test.Contact == 0)).values
add("### Contact false-positive rate on idle rows, Units 1+2 (drives live "
    "false-contact noise)")
add("")
add(f"- Deployed: {(dep_contact[idle12] == 1).mean():.4f}")
add(f"- GBM:      {(gbm_contact_pred[idle12] == 1).mean():.4f}")
add("")
touch12 = (test.Unit.isin([1, 2]) & (test.Contact == 1)).values
add("### Contact false-negative rate on touch rows, Units 1+2")
add("")
add(f"- Deployed: {(dep_contact[touch12] == 0).mean():.4f}")
add(f"- GBM:      {(gbm_contact_pred[touch12] == 0).mean():.4f}")
add("")
add("### Self-closure regression check (GBM contact head)")
add("")
add(f"- Training-fit accuracy on ALL self-closure rows (Contact=0 expected): "
    f"{selfc_acc:.4f} (deployed fix scored 0.9967 training-fit)")
if loo100:
    add(f"- Leave-100%-out per finger (held out of training): " +
        ", ".join(f"{k}={v:.3f}" for k, v in loo100.items()))
add("")

# Force noise on idle rows (what drives resting Force jitter in the GUI)
add("### Idle-row predicted-Force distribution, Units 1+2 (true Force < 0.3N)")
add("")
i12 = idle12
for name, pred in [("Deployed", dep_reg[:, 0]), ("GBM", gbm_force_pred)]:
    p = pred[i12]
    add(f"- {name}: mean={p.mean():+.3f}N  p50={np.percentile(p,50):+.3f}  "
        f"p90={np.percentile(p,90):+.3f}  p99={np.percentile(p,99):+.3f}")
add("")

joblib.dump({
    "gbm_force": gbm_force, "gbm_theta": gbm_theta, "gbm_phi": gbm_phi,
    "gbm_phi_no_theta0": gbm_phi_nz, "gbm_contact": gbm_contact,
    "input_cols": INPUT_COLS, "note": "production_review candidates, raw features, no scaler",
}, OUT_PKL)
add(f"Candidate heads saved to {OUT_PKL} (not deployed).")

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))
print(f"Wrote {OUT_MD}")
