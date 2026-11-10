"""Production-readiness review, step 6: temporal levers.

Two separately-testable questions, both replayed at the LIVE cadence (test
recordings are ~150Hz; the GUI polls at 20Hz, so recordings are subsampled
to ~50ms spacing first -- results at the wrong cadence would be misleading):

  (a) Rate-of-change features: per-channel first differences (delta[t] -
      delta[t-1] at 20Hz) appended to the feature vector. Trained/evaluated
      with GBM heads on the same recording-level split. The live pipeline
      could compute these from one retained previous frame.
  (b) Feature-level EMA smoothing (pipeline lever, no retraining): EMA
      applied to the 15 delta channels BEFORE the deployed model's
      prediction. Metrics: idle predicted-Force std (jitter), touch Force
      MAE (accuracy cost), and release lag (extra ticks for predicted
      Force to fall below 0.3N after true Force does).

Read-only outside production_review/.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

TRAIN_CSV = "model_v1/train_data_backregion.csv"
TEST_CSV = "model_v1/test_data_backregion.csv"
MODEL_PATH = "model_v1/active_model.pkl"
OUT_MD = "model_v1/production_review/06_temporal_report.md"
THR = 0.3
LIVE_DT = 0.05

DELTA_COLS = ['dS1X', 'dS1Y', 'dS1Z', 'dS2X', 'dS2Y', 'dS2Z', 'dS3X', 'dS3Y',
              'dS3Z', 'dS4X', 'dS4Y', 'dS4Z', 'dS5X', 'dS5Y', 'dS5Z']
MAG_COLS = ['S1_mag', 'S2_mag', 'S3_mag', 'S4_mag', 'S5_mag']
INPUT_COLS = DELTA_COLS + MAG_COLS
GBM_KW = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
              early_stopping=True, validation_fraction=0.1,
              n_iter_no_change=15, random_state=42)


def subsample_to_live(df):
    """Keep ~1 row per LIVE_DT seconds within each recording, preserving
    chronological order."""
    out = []
    for _, g in df.groupby("Recording Name", sort=False):
        g = g.sort_values("Time")
        t = g.Time.values
        keep = [0]
        for i in range(1, len(t)):
            if t[i] - t[keep[-1]] >= LIVE_DT:
                keep.append(i)
        out.append(g.iloc[keep])
    return pd.concat(out, ignore_index=True)


print("Loading + subsampling to live 20Hz cadence...")
train = subsample_to_live(pd.read_csv(TRAIN_CSV))
test = subsample_to_live(pd.read_csv(TEST_CSV))
for df in (train, test):
    df["Contact"] = (df.Force >= THR).astype(int)
print(f"train {len(train)} rows, test {len(test)} rows after subsampling")


def add_rate_features(df):
    rate = np.zeros((len(df), 15))
    pos = 0
    for _, g in df.groupby("Recording Name", sort=False):
        D = g[DELTA_COLS].values
        r = np.vstack([np.zeros((1, 15)), np.diff(D, axis=0)])
        rate[pos:pos + len(g)] = r
        pos += len(g)
    return rate


def w_bal(y):
    n0 = (y == 0).sum(); n1 = (y == 1).sum()
    return np.where(y == 0, len(y) / (2 * n0), len(y) / (2 * n1))


lines = ["# Temporal levers: rate features + feature-level EMA", ""]
add = lines.append
add(f"All at the live 20Hz cadence (subsampled): train n={len(train)}, "
    f"test n={len(test)}.")
add("")

# ------------------------------------------------------------- (a) rate feats
add("## (a) Rate-of-change features (GBM heads, same split)")
add("")
Xtr_base = train[INPUT_COLS].values
Xte_base = test[INPUT_COLS].values
Xtr_rate = np.hstack([Xtr_base, add_rate_features(train)])
Xte_rate = np.hstack([Xte_base, add_rate_features(test)])

w = w_bal(train.Contact.values)
res = {}
for tag, Xtr, Xte in [("base", Xtr_base, Xte_base), ("base+rate", Xtr_rate, Xte_rate)]:
    rf = HistGradientBoostingRegressor(**GBM_KW)
    rf.fit(Xtr, train.Force.values, sample_weight=w)
    fmae = np.abs(test.Force.values - rf.predict(Xte)).mean()

    jig_m = train.Unit.isin([1, 2]).values
    wj = w_bal(train.Contact.values[jig_m])
    cf = HistGradientBoostingClassifier(**GBM_KW)
    cf.fit(Xtr[jig_m], train.Contact.values[jig_m], sample_weight=wj)
    u12 = test.Unit.isin([1, 2]).values
    cacc = accuracy_score(test.Contact.values[u12], cf.predict(Xte[u12]))
    res[tag] = (fmae, cacc)
    add(f"- {tag}: Force MAE={fmae:.4f}N (all test), Contact acc (U1+2)={cacc:.4f}")
add("")
add("(If base+rate is not clearly better, rate features are not worth the "
    "live-pipeline statefulness they'd add.)")
add("")

# ------------------------------------------------------------- (b) EMA sweep
add("## (b) Feature-level EMA before the deployed model (no retraining)")
add("")
bundle = joblib.load(MODEL_PATH)
scaler = bundle["scaler"]
reg = bundle["regressor"]


def predict_force_ema(df, alpha):
    preds = np.zeros(len(df))
    pos = 0
    for _, g in df.groupby("Recording Name", sort=False):
        D = g[DELTA_COLS].values
        E = np.zeros_like(D)
        acc = D[0]
        for i in range(len(D)):
            acc = alpha * D[i] + (1 - alpha) * acc
            E[i] = acc
        mags = np.zeros((len(D), 5))
        for s in range(5):
            mags[:, s] = np.linalg.norm(E[:, 3 * s:3 * s + 3], axis=1)
        X = scaler.transform(np.hstack([E, mags]))
        preds[pos:pos + len(g)] = reg.predict(X)[:, 0]
        pos += len(g)
    return preds


u12 = test.Unit.isin([1, 2]).values
idle_m = u12 & (test.Contact.values == 0)
touch_m = u12 & (test.Contact.values == 1)


def release_lag_ticks(df, force_pred):
    """Mean extra ticks (vs truth) for predicted force to drop below THR
    after each true release edge."""
    lags = []
    pos = 0
    for _, g in df.groupby("Recording Name", sort=False):
        n = len(g)
        f_true = g.Force.values
        f_pred = force_pred[pos:pos + n]
        above = f_true >= THR
        for i in range(1, n):
            if above[i - 1] and not above[i]:
                j = i
                while j < min(i + 40, n) and f_pred[j] >= THR:
                    j += 1
                lags.append(j - i)
        pos += n
    return float(np.mean(lags)) if lags else float("nan")


add("| alpha (1.0 = no smoothing) | idle Force std (N) | idle p90 |Force| (N) | touch Force MAE (N) | mean release lag (ticks @20Hz) |")
add("|---|---|---|---|---|")
for alpha in [1.0, 0.6, 0.4, 0.25, 0.15]:
    fp = predict_force_ema(test, alpha)
    idle_std = fp[idle_m].std()
    idle_p90 = np.percentile(np.abs(fp[idle_m]), 90)
    mae_t = np.abs(test.Force.values[touch_m] - fp[touch_m]).mean()
    lag = release_lag_ticks(test, fp)
    add(f"| {alpha} | {idle_std:.4f} | {idle_p90:.4f} | {mae_t:.4f} | {lag:.2f} |")
add("")
add("EMA on the delta features is mathematically upstream of the existing "
    "FORCE_SMOOTHING_WINDOW display average -- if adopted it should replace "
    "(not stack on) part of that, and the release-lag column is the cost "
    "side of the trade.")

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))
print(f"Wrote {OUT_MD}")
