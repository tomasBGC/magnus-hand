"""Production-readiness review, step 3: scale-invariant features for
cross-unit generalisation.

Known, documented weak spot (contact_fix_report.md (a2), forcefix_v2's
scaler saga, baseline report's Unit-3 Contact FN rate of 99.5%): every unit
has its own absolute magnetic scale, so classifiers keyed to absolute delta
magnitudes do not transfer across units. But the DIRECTION PATTERN of the
15-channel delta vector (which sensors move, in which ratio) is set by
contact geometry, not by the unit's absolute sensitivity.

Hypothesis: features = (unit-normalised delta direction) + (log magnitude)
transfer across units far better for Contact and Phi than raw deltas.

Test protocol (leave-one-unit-out, the honest cross-unit check):
  - Contact: train on one unit's train rows, test on the OTHER units' test
    rows. Raw features vs scale-invariant features, same GBM classifier.
  - Phi: same, restricted to theta>=30 rows (azimuth is physically
    degenerate near theta=0, see 02's report).

Also scores the within-distribution case (train U1+2 -> test U1+2) to check
invariant features don't COST accuracy where the current model is already
good.

Read-only outside production_review/.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

TRAIN_CSV = "model_v1/train_data_backregion.csv"
TEST_CSV = "model_v1/test_data_backregion.csv"
OUT_MD = "model_v1/production_review/03_scale_invariant_report.md"

DELTA_COLS = ['dS1X', 'dS1Y', 'dS1Z', 'dS2X', 'dS2Y', 'dS2Z', 'dS3X', 'dS3Y',
              'dS3Z', 'dS4X', 'dS4Y', 'dS4Z', 'dS5X', 'dS5Y', 'dS5Z']
MAG_COLS = ['S1_mag', 'S2_mag', 'S3_mag', 'S4_mag', 'S5_mag']
RAW_FEATURES = DELTA_COLS + MAG_COLS
THR = 0.3
GBM_KW = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
              early_stopping=True, validation_fraction=0.1,
              n_iter_no_change=15, random_state=42)

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
for df in (train, test):
    df["Contact"] = (df.Force >= THR).astype(int)


def invariant_features(df):
    """15 direction components (delta / total L2 norm) + log1p(total norm)
    + 5 per-sensor magnitude fractions. All computable live from the same
    raw_delta dict the pipeline already builds."""
    D = df[DELTA_COLS].values
    total = np.linalg.norm(D, axis=1, keepdims=True)
    total_safe = np.where(total < 1e-9, 1.0, total)
    direction = D / total_safe
    mags = df[MAG_COLS].values
    mag_frac = mags / np.where(total < 1e-9, 1.0, total)
    logmag = np.log1p(total)
    return np.hstack([direction, mag_frac, logmag])


def w_balanced(y):
    n0 = (y == 0).sum(); n1 = (y == 1).sum()
    if n0 == 0 or n1 == 0:
        return np.ones(len(y))
    return np.where(y == 0, len(y) / (2 * n0), len(y) / (2 * n1))


def w_balanced_multi(y):
    vals, counts = np.unique(y, return_counts=True)
    wmap = {v: len(y) / (len(vals) * c) for v, c in zip(vals, counts)}
    return np.array([wmap[v] for v in y])


lines = ["# Scale-invariant features: cross-unit transfer test", ""]
add = lines.append

# ---------------------------------------------------------------- Contact
add("## Contact classifier -- leave-one-unit-out")
add("")
add("| Train unit(s) | Test unit (test-set rows) | Raw-feature acc | Invariant-feature acc |")
add("|---|---|---|---|")

combos = [([1], 2), ([1], 3), ([2], 1), ([2], 3), ([1, 2], 3)]
for train_units, test_unit in combos:
    tr = train[train.Unit.isin(train_units)]
    te = test[test.Unit == test_unit]
    ytr, yte = tr.Contact.values, te.Contact.values
    w = w_balanced(ytr)

    m_raw = HistGradientBoostingClassifier(**GBM_KW)
    m_raw.fit(tr[RAW_FEATURES].values, ytr, sample_weight=w)
    acc_raw = accuracy_score(yte, m_raw.predict(te[RAW_FEATURES].values))

    m_inv = HistGradientBoostingClassifier(**GBM_KW)
    m_inv.fit(invariant_features(tr), ytr, sample_weight=w)
    acc_inv = accuracy_score(yte, m_inv.predict(invariant_features(te)))

    add(f"| {train_units} | {test_unit} | {acc_raw:.4f} | {acc_inv:.4f} |")
add("")

# Within-distribution check: does invariance cost in-domain accuracy?
tr = train[train.Unit.isin([1, 2])]
te = test[test.Unit.isin([1, 2])]
w = w_balanced(tr.Contact.values)
m_raw = HistGradientBoostingClassifier(**GBM_KW)
m_raw.fit(tr[RAW_FEATURES].values, tr.Contact.values, sample_weight=w)
acc_raw = accuracy_score(te.Contact.values, m_raw.predict(te[RAW_FEATURES].values))
m_inv = HistGradientBoostingClassifier(**GBM_KW)
m_inv.fit(invariant_features(tr), tr.Contact.values, sample_weight=w)
acc_inv = accuracy_score(te.Contact.values, m_inv.predict(invariant_features(te)))
add(f"Within-distribution (train U1+2 -> test U1+2): raw={acc_raw:.4f}, "
    f"invariant={acc_inv:.4f}")
add("")

# ---------------------------------------------------------------- Phi
add("## Phi classifier (theta>=30 rows only) -- leave-one-unit-out")
add("")
add("Unit 3 has no theta>=30 rows with Phi in {-90,0,90} in test except "
    "theta=90 -- Unit 3's test recordings ARE theta=90, so this is exactly "
    "the back-region transfer question.")
add("")
add("| Train unit(s) | Test unit | Raw acc | Invariant acc | n test |")
add("|---|---|---|---|---|")

for train_units, test_unit in combos:
    tr = train[(train.Unit.isin(train_units)) & (train.Theta >= 30)
               & (train.Phi.isin([-90, 0, 90]))]
    te = test[(test.Unit == test_unit) & (test.Theta >= 30)
              & (test.Phi.isin([-90, 0, 90]))]
    if not len(te) or tr.Phi.nunique() < 2:
        continue
    w = w_balanced_multi(tr.Phi.values)
    m_raw = HistGradientBoostingClassifier(**GBM_KW)
    m_raw.fit(tr[RAW_FEATURES].values, tr.Phi.values, sample_weight=w)
    acc_raw = accuracy_score(te.Phi.values, m_raw.predict(te[RAW_FEATURES].values))
    m_inv = HistGradientBoostingClassifier(**GBM_KW)
    m_inv.fit(invariant_features(tr), tr.Phi.values, sample_weight=w)
    acc_inv = accuracy_score(te.Phi.values, m_inv.predict(invariant_features(te)))
    add(f"| {train_units} | {test_unit} | {acc_raw:.4f} | {acc_inv:.4f} | {len(te)} |")
add("")

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))
print(f"Wrote {OUT_MD}")
