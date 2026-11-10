"""Train force_direction_model_v1_contact_forcefix_v2.pkl -- fixes the
scaler-dilution regression identified in force_direction_model_v1_contact_forcefix.pkl.

Root cause (confirmed via live A/B testing + offline analysis, see project
notes): train_model_forcefix.py fit ONE shared StandardScaler on Units 1+2+3
RAW pooled data. Unit 3's raw deltas/magnitudes have far larger spread than
Units 1/2 (e.g. S4_mag std: Unit1=49, Unit2=33, Unit3=12433 -- a ~250-370x
difference), and StandardScaler's fitted scale_ is the pooled standard
deviation, which a small but wide-spread minority group (Unit 3 is only
~8% of rows) can dominate. That inflated scale_ by up to 396x on some
channels relative to the original (Units-1+2-only) model, compressing
Units 1/2's real signal into a tiny sliver of the input range the network
had to also represent Unit 3's much wider range in -- diluting precision/
sensitivity specifically for the deployed hand (which behaves like Units 1/2
scale for ordinary front-of-fingertip presses), confirmed live: Finger 1 was
calm under the original model and noisy under forcefix, on identical presses.

Fix: PER-UNIT normalization before merging. Each unit gets its own
StandardScaler, fit on only that unit's own rows -- this removes each
unit's own raw-scale/offset independently, so no single source's raw
variance can dominate a shared fit. The single scaler actually deployed
(and used to train every head, so train/deploy always agree) is built by
taking the per-channel MEDIAN of the three per-unit scalers' mean_/scale_
-- an equal-weight AVERAGE was tried first and made the worst channel worse
(dS4Y's ratio to the original model's scale_ went 396x -> 475x), because a
couple of Unit 3's own channels (dS4Y, S4_mag) are themselves extreme
outliers even in isolation (dS4Y std: Unit1=9.37, Unit2=9.78, Unit3=12,531 --
over 1000x, almost certainly a hand-held-collection artifact on that one
channel, not real signal) -- averaging still let that one unit's outlier
channel dominate; the median simply discards whichever single unit is the
outlier on a given channel, while still giving Unit 3 fair (not
row-count-diluted) representation on every channel where it ISN'T one.

Force+Theta regressor: trained on all 3 units combined (matching
train_model_forcefix.py's own already-validated choice -- Force benefits
directly from Unit 3 coverage), through the new balanced scaler, with
sample weights balancing BOTH Contact class (idle vs touch, as before) AND
source unit (so Unit 3's much smaller row count doesn't get outvoted in the
loss the way it would with uniform weights) -- weighting fixes fairness in
the LOSS the same way the balanced scaler fixes fairness in the INPUT SCALE;
one alone doesn't address the other.

Phi classifier (jig-only) and Contact classifier: SAME training data source
and methodology as force_direction_model_v1_contact_forcefix.pkl (Units 1+2
only, not re-litigating the combined-vs-jigonly Phi decision already made
there) -- but necessarily refit against the NEW scaler (a classifier's
decision boundary is calibrated to whatever coordinate system its training
scaler produces; keeping the old classifier objects would silently feed
them differently-scaled numbers than they were trained on). This mirrors
the same "still needs a fresh fit when the shared scaler changes" pattern
already documented in train_model_forcefix.py's own module docstring.

Usage: uv run python model_v1/train_model_forcefix_v2.py
"""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import joblib

DELTA_COLS = [
    "dS1X", "dS1Y", "dS1Z",
    "dS2X", "dS2Y", "dS2Z",
    "dS3X", "dS3Y", "dS3Z",
    "dS4X", "dS4Y", "dS4Z",
    "dS5X", "dS5Y", "dS5Z",
]
MAGNITUDE_COLS = ["S1_mag", "S2_mag", "S3_mag", "S4_mag", "S5_mag"]
INPUT_COLS = DELTA_COLS + MAGNITUDE_COLS
REGRESSION_OUTPUT_COLS = ["Force", "Theta"]
PHI_COL = "Phi"
PHI_CLASSES_JIGONLY = [-90, 0, 90]
CONTACT_COL = "Contact"
CONTACT_CLASSES = [0, 1]
CONTACT_FORCE_THRESHOLD = 0.3
UNITS = [1, 2, 3]

TRAIN_CSV = "model_v1/train_data_backregion.csv"
FINAL_MODEL_PATH = "model_v1/active_model.pkl"

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
    verbose=True,
)


def balanced_weight(df, col=CONTACT_COL):
    n0 = (df[col] == 0).sum()
    n1 = (df[col] == 1).sum()
    weight_map = {0: len(df) / (2 * n0), 1: len(df) / (2 * n1)}
    return df[col].map(weight_map).values, weight_map


df = pd.read_csv(TRAIN_CSV)
print(f"Loaded {len(df)} rows from {TRAIN_CSV}")
df[CONTACT_COL] = (df["Force"] >= CONTACT_FORCE_THRESHOLD).astype(int)

unexpected_phi = sorted(set(df[PHI_COL].unique()) - {-90, -45, 0, 45, 90})
assert not unexpected_phi, f"Unexpected Phi values: {unexpected_phi}"

for u in UNITS:
    n = (df["Unit"] == u).sum()
    print(f"  Unit {u}: {n} rows ({n / len(df):.1%})")

# ============================================================================
# Step 1: per-unit scalers, fit independently on each unit's own rows only
# (with that unit's own Contact-balanced sample weights, same convention as
# every other scaler fit in this pipeline).
# ============================================================================
print("\n--- Fitting per-unit scalers ---")
per_unit_scalers = {}
for u in UNITS:
    df_u = df[df["Unit"] == u]
    w_u, wmap_u = balanced_weight(df_u)
    s_u = StandardScaler()
    s_u.fit(df_u[INPUT_COLS].values, sample_weight=w_u)
    per_unit_scalers[u] = s_u
    print(f"  Unit {u}: idle x{wmap_u[0]:.3f}, contact x{wmap_u[1]:.3f}  "
          f"mean_[:3]={s_u.mean_[:3].round(2).tolist()}  scale_[:3]={s_u.scale_[:3].round(2).tolist()}")

# ============================================================================
# Step 2: the ONE scaler that's actually deployed (used to transform every
# row for every head, train and inference alike). Per-channel MEDIAN (not
# mean) across the 3 per-unit scalers -- a plain equal-weight average was
# tried first and made the worst channel WORSE (dS4Y's ratio to the original
# model's scale_ went 396x -> 475x), because Unit 3 has a handful of
# channels (dS4Y, S4_mag) where ITS OWN per-unit std is a genuine, isolated
# outlier -- e.g. dS4Y std is 9.37/9.78 on Units 1/2 but 12,531 on Unit 3,
# over 1000x larger, almost certainly a hand-held-collection artifact on
# that one channel rather than real signal. Naive pooling had actually
# DAMPENED that outlier's effect on scale_ by weighting it down to Unit 3's
# ~7.7% row share; giving it a full, undamped 1/3 share made it worse, not
# better. The median of 3 values discards whichever single unit is the
# outlier (in either direction) for a given channel -- for dS4Y that means
# Unit 3 is simply excluded from that one channel's scale_, while it's still
# fairly represented on every channel where it ISN'T an outlier.
# ============================================================================
mean_stack = np.stack([per_unit_scalers[u].mean_ for u in UNITS])
scale_stack = np.stack([per_unit_scalers[u].scale_ for u in UNITS])
balanced_mean = np.median(mean_stack, axis=0)
balanced_scale = np.median(scale_stack, axis=0)

scaler = StandardScaler()
scaler.mean_ = balanced_mean
scaler.scale_ = balanced_scale
scaler.var_ = balanced_scale ** 2
scaler.n_features_in_ = len(INPUT_COLS)
scaler.n_samples_seen_ = len(df)
scaler.with_mean = True
scaler.with_std = True

print("\n--- Balanced deployed scaler (per-channel MEDIAN of the 3 per-unit scalers) ---")
print(f"  mean_[:3]={scaler.mean_[:3].round(2).tolist()}  scale_[:3]={scaler.scale_[:3].round(2).tolist()}")

X_all = df[INPUT_COLS].values
X_all_scaled = scaler.transform(X_all)

# ============================================================================
# Step 3: sample weights for the regressor -- Contact-class balance ONLY
# (same as v1/train_model_forcefix.py), i.e. natural row-count weighting
# across units. An earlier attempt ALSO added an explicit per-unit balance
# weight (equal loss-mass per unit, ~4.3x boost on Unit 3) on top of this
# median scaler and it made things worse, not better: this scaler already
# keeps jig data near its original scale, which means Unit 3's raw values
# -- still ~10-1000x larger in absolute terms -- land as large z-scores
# through it (the median fix repairs the deployed SCALE, it doesn't shrink
# Unit 3's inherently larger physical signal). Upweighting those already-
# large-magnitude rows 4.3x compounded rather than balanced that, dragging
# the fit away from jig accuracy. Leaving Unit 3 at its natural ~7.7% share
# of the loss lets jig data (the deployed hand's actual regime) dominate
# fitting as intended, while Unit 3 -- a non-trivial minority, not a
# vanishing one -- still contributes a learnable signal.
# ============================================================================
contact_w_all, contact_wmap_all = balanced_weight(df)
print(f"\nContact balance weights: idle x{contact_wmap_all[0]:.3f}, contact x{contact_wmap_all[1]:.3f}")

sample_weight_reg = contact_w_all

# ============================================================================
# Force + Theta regressor: COMBINED data (Units 1+2+3), through the balanced
# scaler, with the combined Contact x Unit sample weights.
# ============================================================================
print("\n--- Training Force/Theta regressor (combined Units 1+2+3, balanced scaler + weights) ---")
regressor = MLPRegressor(**MLP_KWARGS)
regressor.fit(X_all_scaled, df[REGRESSION_OUTPUT_COLS].values, sample_weight=sample_weight_reg)

pred_reg = regressor.predict(X_all_scaled)
for i, col in enumerate(REGRESSION_OUTPUT_COLS):
    mae = mean_absolute_error(df[REGRESSION_OUTPUT_COLS].values[:, i], pred_reg[:, i])
    r2 = r2_score(df[REGRESSION_OUTPUT_COLS].values[:, i], pred_reg[:, i])
    print(f"  Training-fit {col}: MAE={mae:.4f}  R2={r2:.4f}")

# ============================================================================
# Jig-only subset (Units 1+2), transformed with the SAME balanced scaler
# (reused, not refit) -- Phi and Contact stay jig-only, matching
# train_model_forcefix.py's already-decided methodology.
# ============================================================================
jig_mask = df["Unit"].isin([1, 2]).values
df_jig = df[jig_mask].reset_index(drop=True)
print(f"\nJig-only (Units 1+2) subset: {len(df_jig)} rows")
X_jig_scaled = scaler.transform(df_jig[INPUT_COLS].values)  # reuse balanced scaler, do not refit
sample_weight_jig, weight_map_jig = balanced_weight(df_jig)
print(f"Jig-only Contact balance weights: idle x{weight_map_jig[0]:.3f}, "
      f"contact x{weight_map_jig[1]:.3f}")

print("\n--- Training Phi classifier: jig-only Units 1+2 (unchanged methodology, new scaler) ---")
phi_classifier_jigonly = MLPClassifier(**MLP_KWARGS)
phi_classifier_jigonly.fit(X_jig_scaled, df_jig[PHI_COL].values, sample_weight=sample_weight_jig)
acc = accuracy_score(df_jig[PHI_COL].values, phi_classifier_jigonly.predict(X_jig_scaled))
print(f"  Training-fit Phi (jig-only): accuracy={acc:.4f}")

print("\n--- Training Contact classifier: jig-only Units 1+2 (unchanged methodology, new scaler) ---")
contact_classifier = MLPClassifier(**MLP_KWARGS)
contact_classifier.fit(X_jig_scaled, df_jig[CONTACT_COL].values, sample_weight=sample_weight_jig)
acc = accuracy_score(df_jig[CONTACT_COL].values, contact_classifier.predict(X_jig_scaled))
print(f"  Training-fit Contact (jig-only): accuracy={acc:.4f}")

joblib.dump(
    {
        "regressor": regressor,
        "classifier": phi_classifier_jigonly,
        "contact_classifier": contact_classifier,
        "scaler": scaler,
        "input_cols": INPUT_COLS,
        "regression_output_cols": REGRESSION_OUTPUT_COLS,
        "phi_col": PHI_COL,
        "phi_classes": PHI_CLASSES_JIGONLY,
        "phi_training_source": "jigonly_units_1_2",
        "contact_col": CONTACT_COL,
        "contact_classes": CONTACT_CLASSES,
        "contact_force_threshold": CONTACT_FORCE_THRESHOLD,
        "delta_cols": DELTA_COLS,
        "magnitude_cols": MAGNITUDE_COLS,
        "feature_kind": "delta_v1_contact_forcefix_v2",
        "scaler_source": "per_unit_balanced_average",
        "head_training_data": {
            "regressor_force_theta": "combined_units_1_2_3_balanced_scaler_and_weights",
            "classifier_phi": "jigonly_units_1_2",
            "contact_classifier": "jigonly_units_1_2",
        },
    },
    FINAL_MODEL_PATH,
)
print(f"\nSaved final model to {FINAL_MODEL_PATH}")
print(f"  Force+Theta regressor: combined Units 1+2+3, balanced per-unit scaler + Contact x Unit sample weights")
print(f"  Phi classifier:        jigonly_units_1_2")
print(f"  Contact classifier:    jigonly_units_1_2")
