"""Production-readiness review, step 1: quantify the CURRENTLY DEPLOYED
model's baseline performance on the existing held-out test set, with much
finer breakdowns than evaluate_model_forcefix_v2.py reports -- per unit, per
theta (incl. the theta=90 back region), per phi, per force band -- plus
Contact/Phi probability calibration (does the model know when it doesn't
know?).

Read-only: loads model_v1/active_model.pkl and test/train CSVs, writes
nothing except its own report (production_review/01_baseline_report.md).
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

MODEL_PATH = "model_v1/active_model.pkl"
TEST_CSV = "model_v1/test_data_backregion.csv"
TRAIN_CSV = "model_v1/train_data_backregion.csv"
OUT_MD = "model_v1/production_review/01_baseline_report.md"

bundle = joblib.load(MODEL_PATH)
scaler = bundle["scaler"]
reg = bundle["regressor"]
phi_clf = bundle["classifier"]
contact_clf = bundle["contact_classifier"]
INPUT_COLS = bundle["input_cols"]
THR = bundle["contact_force_threshold"]

df = pd.read_csv(TEST_CSV)
df["Contact"] = (df["Force"] >= THR).astype(int)
X = scaler.transform(df[INPUT_COLS].values)
reg_pred = reg.predict(X)
df["Force_pred"] = reg_pred[:, 0]
df["Theta_pred"] = reg_pred[:, 1]
df["Phi_pred"] = phi_clf.predict(X)
phi_proba = phi_clf.predict_proba(X)
df["Phi_conf"] = phi_proba.max(axis=1)
contact_proba = contact_clf.predict_proba(X)
contact_cls1_idx = list(contact_clf.classes_).index(1)
df["Contact_p1"] = contact_proba[:, contact_cls1_idx]
df["Contact_pred"] = contact_clf.predict(X)

lines = ["# Baseline evaluation of deployed active_model.pkl", ""]
lines.append(f"- Test set: {TEST_CSV}, n={len(df)} rows, "
             f"{df['Recording Name'].nunique()} held-out recordings")
lines.append("")


def add(s=""):
    lines.append(s)


# ---------------------------------------------------------------- overall
add("## Overall (all test rows)")
add()
fmae = np.abs(df.Force - df.Force_pred).mean()
tmae = np.abs(df.Theta - df.Theta_pred).mean()
add(f"- Force MAE: {fmae:.4f} N")
add(f"- Theta MAE: {tmae:.4f} deg")
add(f"- Phi accuracy (rows with Phi in [-90,0,90]): "
    f"{accuracy_score(df.Phi, df.Phi_pred):.4f}")
add(f"- Contact accuracy: {accuracy_score(df.Contact, df.Contact_pred):.4f}")
add()

# ---------------------------------------------------------------- per unit
add("## Per unit")
add()
add("| Unit | n | Force MAE (N) | Theta MAE (deg) | Phi acc | Contact acc | Contact FP rate (idle rows misread as contact) | Contact FN rate |")
add("|---|---|---|---|---|---|---|---|")
for u in sorted(df.Unit.unique()):
    s = df[df.Unit == u]
    idle = s[s.Contact == 0]
    touch = s[s.Contact == 1]
    fp = (idle.Contact_pred == 1).mean() if len(idle) else float("nan")
    fn = (touch.Contact_pred == 0).mean() if len(touch) else float("nan")
    add(f"| {u} | {len(s)} | {np.abs(s.Force - s.Force_pred).mean():.4f} | "
        f"{np.abs(s.Theta - s.Theta_pred).mean():.4f} | "
        f"{accuracy_score(s.Phi, s.Phi_pred):.4f} | "
        f"{accuracy_score(s.Contact, s.Contact_pred):.4f} | {fp:.4f} | {fn:.4f} |")
add()

# ---------------------------------------------------------------- per theta
add("## Per theta (all units pooled)")
add()
add("| Theta | n | Force MAE | Theta MAE | Phi acc | Contact acc |")
add("|---|---|---|---|---|---|")
for th in sorted(df.Theta.unique()):
    s = df[df.Theta == th]
    add(f"| {th} | {len(s)} | {np.abs(s.Force - s.Force_pred).mean():.4f} | "
        f"{np.abs(s.Theta - s.Theta_pred).mean():.4f} | "
        f"{accuracy_score(s.Phi, s.Phi_pred):.4f} | "
        f"{accuracy_score(s.Contact, s.Contact_pred):.4f} |")
add()

# ------------------------------------------------- per (unit, theta, phi)
add("## Per held-out recording (the real granularity of the split)")
add()
add("| Recording | Unit | Theta | Phi | n | Force MAE | Theta MAE | Phi acc | Contact acc |")
add("|---|---|---|---|---|---|---|---|---|")
for name, s in df.groupby("Recording Name"):
    add(f"| {name} | {s.Unit.iloc[0]} | {s.Theta.iloc[0]} | {s.Phi.iloc[0]} | {len(s)} | "
        f"{np.abs(s.Force - s.Force_pred).mean():.3f} | "
        f"{np.abs(s.Theta - s.Theta_pred).mean():.2f} | "
        f"{accuracy_score(s.Phi, s.Phi_pred):.3f} | "
        f"{accuracy_score(s.Contact, s.Contact_pred):.3f} |")
add()

# ---------------------------------------------------------------- force bands
add("## Force accuracy by band (Units 1+2 -- deployed-hand regime)")
add()
add("| Band (N) | n | MAE (N) | Bias (N) | rel MAE |")
add("|---|---|---|---|---|")
u12 = df[df.Unit.isin([1, 2])]
for lo, hi in [(0.0, 0.3), (0.3, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 100.0)]:
    s = u12[(u12.Force >= lo) & (u12.Force < hi)]
    if not len(s):
        continue
    mae = np.abs(s.Force - s.Force_pred).mean()
    bias = (s.Force_pred - s.Force).mean()
    rel = mae / max(s.Force.mean(), 1e-9)
    add(f"| [{lo},{hi}) | {len(s)} | {mae:.4f} | {bias:+.4f} | {rel:.1%} |")
add()

# ---------------------------------------------------------------- calibration
add("## Contact classifier probability calibration (test set)")
add()
add("Reliability table: rows bucketed by predicted P(contact); a well-"
    "calibrated classifier's empirical contact fraction matches the bucket "
    "center. ECE = expected calibration error (bucket-weighted |gap|).")
add()
add("| P(contact) bucket | n | Mean predicted P | Empirical contact fraction | Gap |")
add("|---|---|---|---|---|")
bins = np.linspace(0, 1, 11)
ece = 0.0
for i in range(10):
    m = (df.Contact_p1 >= bins[i]) & (df.Contact_p1 < bins[i + 1] if i < 9 else df.Contact_p1 <= 1.0)
    s = df[m]
    if not len(s):
        continue
    mp = s.Contact_p1.mean()
    emp = s.Contact.mean()
    ece += len(s) / len(df) * abs(mp - emp)
    add(f"| [{bins[i]:.1f},{bins[i+1]:.1f}) | {len(s)} | {mp:.3f} | {emp:.3f} | {emp - mp:+.3f} |")
add()
add(f"**Contact ECE: {ece:.4f}**")
add()

add("## Phi classifier confidence calibration (test set)")
add()
add("| Max-prob bucket | n | Mean confidence | Empirical accuracy | Gap |")
add("|---|---|---|---|---|")
bins = np.array([0.33, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001])
phi_ece = 0.0
correct = (df.Phi == df.Phi_pred).values
for i in range(len(bins) - 1):
    m = (df.Phi_conf >= bins[i]) & (df.Phi_conf < bins[i + 1])
    s = df[m]
    if not len(s):
        continue
    mc = s.Phi_conf.mean()
    acc = correct[m.values].mean()
    phi_ece += len(s) / len(df) * abs(mc - acc)
    add(f"| [{bins[i]:.2f},{bins[i+1]:.2f}) | {len(s)} | {mc:.3f} | {acc:.3f} | {acc - mc:+.3f} |")
add()
add(f"**Phi ECE (over max-prob buckets): {phi_ece:.4f}**")
add()

# ---------------------------------------------------------------- coverage
add("## Training data coverage (train_data_backregion.csv)")
add()
tr = pd.read_csv(TRAIN_CSV)
tr["Contact"] = (tr.Force >= THR).astype(int)
add(f"- Total train rows: {len(tr)}, recordings: {tr['Recording Name'].nunique()}")
add(f"- Contact balance: idle={int((tr.Contact==0).sum())} ({(tr.Contact==0).mean():.1%}), "
    f"touch={int((tr.Contact==1).sum())} ({(tr.Contact==1).mean():.1%})")
add()
add("Rows and recordings per (Unit, Theta, Phi) cell -- blank cells are "
    "combinations that were NEVER recorded:")
add()
cell = tr.groupby(["Unit", "Theta", "Phi"]).agg(
    n=("Force", "size"), recs=("Recording Name", "nunique"),
    max_force=("Force", "max")).reset_index()
add("| Unit | Theta | Phi | rows | recordings | max Force (N) |")
add("|---|---|---|---|---|---|")
for _, r in cell.iterrows():
    add(f"| {int(r.Unit)} | {int(r.Theta)} | {int(r.Phi)} | {int(r.n)} | {int(r.recs)} | {r.max_force:.2f} |")
add()
# Missing combos across the notional full grid
thetas = sorted(tr.Theta.unique())
phis = [-90, -45, 0, 45, 90]
missing = []
for u in sorted(tr.Unit.unique()):
    for th in thetas:
        for ph in phis:
            if not len(cell[(cell.Unit == u) & (cell.Theta == th) & (cell.Phi == ph)]):
                missing.append((u, th, ph))
add(f"Missing (Unit, Theta, Phi) combinations vs. the full 3x7x5 grid: "
    f"{len(missing)} of {3*len(thetas)*len(phis)}")
add()
add("Phi=-45/+45 coverage: " + str(cell[cell.Phi.isin([-45, 45])][["Unit", "Theta", "Phi", "n", "recs"]].to_dict("records")))
add()

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))
print(f"Wrote {OUT_MD}")
print(f"Headline: Force MAE={fmae:.4f}N  Theta MAE={tmae:.4f}deg  "
      f"Phi acc={accuracy_score(df.Phi, df.Phi_pred):.4f}  "
      f"Contact acc={accuracy_score(df.Contact, df.Contact_pred):.4f}  "
      f"Contact ECE={ece:.4f}")
