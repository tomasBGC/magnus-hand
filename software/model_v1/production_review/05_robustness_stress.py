"""Production-readiness review, step 5: robustness / failure-mode stress
tests, run offline against the deployed model and the GBM candidate heads.

Scenarios (all synthetic perturbations of REAL test-set frames):

  1. Sub-guard glitches: single-channel spikes BELOW SANE_DELTA_MAX=2000
     (300..1900 raw units) injected into genuinely idle Units-1+2 frames.
     The live glitch guard only rejects |delta|>2000, so all of these reach
     the model. Metric: fraction of frames whose (Contact==1 AND
     Force>MIN_CONTACT_FORCE_N=0.1) would feed the live confirm streak, and
     the Force magnitude that would be displayed if confirmed.
  2. Scale sweep (OOD extrapolation): a real median-force touch frame's
     delta vector scaled x0.25..x30. A production model should saturate or
     degrade gracefully, not explode. MLPs extrapolate linearly without
     bound; GBMs saturate at the training envelope by construction.
  3. Noise gain: N(0, sigma) added to idle and touch frames (sigma=2.5 raw
     units, the measured sensor noise floor from crosstalk_model_report.md's
     noise sigma column). Metric: std of predicted Force across 50 noise
     draws -- the model's own contribution to displayed jitter.
  4. Exact-zero input (perfectly clean baseline): predicted Force/Contact.

Read-only outside production_review/.
"""
import numpy as np
import pandas as pd
import joblib

TEST_CSV = "model_v1/test_data_backregion.csv"
MODEL_PATH = "model_v1/active_model.pkl"
GBM_PKL = "model_v1/production_review/candidate_gbm_heads.pkl"
OUT_MD = "model_v1/production_review/05_robustness_report.md"
THR = 0.3
MIN_CONTACT_FORCE_N = 0.1
SANE_DELTA_MAX = 2000.0
RNG = np.random.default_rng(42)

DELTA_COLS = ['dS1X', 'dS1Y', 'dS1Z', 'dS2X', 'dS2Y', 'dS2Z', 'dS3X', 'dS3Y',
              'dS3Z', 'dS4X', 'dS4Y', 'dS4Z', 'dS5X', 'dS5Y', 'dS5Z']
MAG_COLS = ['S1_mag', 'S2_mag', 'S3_mag', 'S4_mag', 'S5_mag']
INPUT_COLS = DELTA_COLS + MAG_COLS

bundle = joblib.load(MODEL_PATH)
gbm = joblib.load(GBM_PKL)


def recompute_mags(D):
    """D: (n,15) delta matrix -> (n,5) per-sensor magnitudes. The live
    pipeline recomputes magnitudes from the (possibly corrupted) deltas, so
    the stress input must too."""
    out = np.zeros((len(D), 5))
    for s in range(5):
        out[:, s] = np.linalg.norm(D[:, 3 * s:3 * s + 3], axis=1)
    return out


def predict_deployed(D):
    X = np.hstack([D, recompute_mags(D)])
    Xs = bundle["scaler"].transform(X)
    f = bundle["regressor"].predict(Xs)[:, 0]
    c = bundle["contact_classifier"].predict(Xs)
    return f, c


def predict_gbm(D):
    X = np.hstack([D, recompute_mags(D)])
    f = gbm["gbm_force"].predict(X)
    c = gbm["gbm_contact"].predict(X)
    return f, c


test = pd.read_csv(TEST_CSV)
test["Contact"] = (test.Force >= THR).astype(int)
u12 = test[test.Unit.isin([1, 2])]
idle = u12[u12.Contact == 0]
touch = u12[u12.Contact == 1]

idle_D = idle[DELTA_COLS].values
n_idle = min(3000, len(idle_D))
idx = RNG.choice(len(idle_D), n_idle, replace=False)
idle_D = idle_D[idx]

lines = ["# Robustness stress-test report", ""]
add = lines.append
add(f"Base frames: {n_idle} idle Units-1+2 test frames (true Force<0.3N), "
    f"plus a median-force real touch frame for the scale sweep.")
add("")

# ---------------------------------------------------------- 1. sub-guard glitch
add("## 1. Single-channel glitches below the SANE_DELTA_MAX=2000 guard")
add("")
add("Spike added to one random channel per frame (sign random). Frames "
    "whose corrupted deltas exceed 2000 are excluded (the live guard "
    "catches those). 'confirm-feed rate' = fraction with Contact==1 AND "
    "Force>0.1N, i.e. frames that would push the live confirm streak; 5 "
    "consecutive such frames = a false CONTACT on screen.")
add("")
add("| Spike (raw units) | Deployed confirm-feed rate | Deployed mean Force if confirmed (N) | GBM confirm-feed rate | GBM mean Force (N) |")
add("|---|---|---|---|---|")
for spike in [300, 600, 1000, 1500, 1900]:
    D = idle_D.copy()
    ch = RNG.integers(0, 15, len(D))
    sign = RNG.choice([-1.0, 1.0], len(D))
    D[np.arange(len(D)), ch] += sign * spike
    ok = (np.abs(D) <= SANE_DELTA_MAX).all(axis=1)
    D = D[ok]
    fd, cd = predict_deployed(D)
    fg, cg = predict_gbm(D)
    feed_d = (cd == 1) & (fd > MIN_CONTACT_FORCE_N)
    feed_g = (cg == 1) & (fg > MIN_CONTACT_FORCE_N)
    add(f"| {spike} | {feed_d.mean():.4f} | "
        f"{fd[feed_d].mean() if feed_d.any() else 0:.2f} | "
        f"{feed_g.mean():.4f} | {fg[feed_g].mean() if feed_g.any() else 0:.2f} |")
add("")

# ---------------------------------------------------------- 2. scale sweep
add("## 2. Scale sweep on a real touch frame (OOD extrapolation)")
add("")
med_frame = touch.iloc[(touch.Force - touch.Force.median()).abs().argsort().iloc[0]]
base_D = med_frame[DELTA_COLS].values.astype(float).reshape(1, -1)
add(f"Base frame: true Force={med_frame.Force:.2f}N, Unit {med_frame.Unit}, "
    f"theta={med_frame.Theta}, phi={med_frame.Phi}.")
add("")
add("| Scale | Deployed Force (N) | GBM Force (N) | Deployed Contact | GBM Contact |")
add("|---|---|---|---|---|")
for s in [0.25, 0.5, 1, 2, 4, 8, 15, 30]:
    D = base_D * s
    if (np.abs(D) > SANE_DELTA_MAX).any():
        add(f"| x{s} | (blocked by glitch guard) | | | |")
        continue
    fd, cd = predict_deployed(D)
    fg, cg = predict_gbm(D)
    add(f"| x{s} | {fd[0]:.2f} | {fg[0]:.2f} | {cd[0]} | {cg[0]} |")
add("")

# ---------------------------------------------------------- 3. noise gain
add("## 3. Noise gain (sigma=2.5 raw units, 50 draws per frame)")
add("")
sub_idle = idle_D[:300]
touch_D = touch[DELTA_COLS].values
idx_t = RNG.choice(len(touch_D), 300, replace=False)
sub_touch = touch_D[idx_t]


def noise_gain(D, predict):
    stds = []
    for row in D[:100]:
        reps = row[None, :] + RNG.normal(0, 2.5, (50, 15))
        f, _ = predict(reps)
        stds.append(f.std())
    return float(np.mean(stds))


for label, D in [("idle", sub_idle), ("touch", sub_touch)]:
    gd = noise_gain(D, predict_deployed)
    gg = noise_gain(D, predict_gbm)
    add(f"- {label} frames: deployed Force std={gd:.4f}N per 2.5-unit sensor "
        f"noise; GBM std={gg:.4f}N")
add("")
add("This is the model's own noise amplification -- the floor that display "
    "smoothing has to fight. Lower = calmer displayed Force at rest and "
    "under steady press.")
add("")

# ---------------------------------------------------------- 4. exact zero
add("## 4. Exact-zero delta (perfect baseline)")
add("")
D0 = np.zeros((1, 15))
fd, cd = predict_deployed(D0)
fg, cg = predict_gbm(D0)
add(f"- Deployed: Force={fd[0]:+.4f}N, Contact={cd[0]}")
add(f"- GBM:      Force={fg[0]:+.4f}N, Contact={cg[0]}")
add("")

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))
print(f"Wrote {OUT_MD}")
