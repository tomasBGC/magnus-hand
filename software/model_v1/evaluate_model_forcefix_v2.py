"""Evaluate force_direction_model_v1_contact_forcefix_v2.pkl (per-unit
balanced-scaler fix) against BOTH the original model
(force_direction_model_v1_contact.pkl) and forcefix v1
(force_direction_model_v1_contact_forcefix.pkl), on test_data_backregion.csv.

Three checks, matching what was asked:
  (a) Units 1+2 Force/Theta/Contact accuracy: v2 should return to ~original
      baseline (no scaler-related regression vs the original model).
  (b) Unit 3 (backregion) Force accuracy: v2 should stay close to forcefix
      v1's big improvement (~2.65N MAE), not regress back toward the
      original model's ~54.7N MAE.
  (c) Per-channel scaler_ ratio (v2 vs original): confirm the extreme
      distortion found in forcefix v1 (up to ~396x on one channel) is gone.

Usage: uv run python model_v1/evaluate_model_forcefix_v2.py
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, mean_absolute_error

TEST_CSV = "model_v1/test_data_backregion.csv"
OLD_MODEL_PATH = "model_v1/force_direction_model_v1_contact.pkl"
V1_MODEL_PATH = "model_v1/force_direction_model_v1_contact_forcefix.pkl"
V2_MODEL_PATH = "model_v1/active_model.pkl"

old = joblib.load(OLD_MODEL_PATH)
v1 = joblib.load(V1_MODEL_PATH)
v2 = joblib.load(V2_MODEL_PATH)

df = pd.read_csv(TEST_CSV)
print(f"Loaded {len(df)} rows from {TEST_CSV}")
df[v2["contact_col"]] = (df["Force"] >= v2["contact_force_threshold"]).astype(int)

unit3_mask = (df["Unit"] == 3).values
unit12_mask = (df["Unit"].isin([1, 2])).values

X_old_scaled = old["scaler"].transform(df[old["input_cols"]].values)
X_v1_scaled = v1["scaler"].transform(df[v1["input_cols"]].values)
X_v2_scaled = v2["scaler"].transform(df[v2["input_cols"]].values)

old_reg_pred = old["regressor"].predict(X_old_scaled)
v1_reg_pred = v1["regressor"].predict(X_v1_scaled)
v2_reg_pred = v2["regressor"].predict(X_v2_scaled)

old_contact_pred = old["contact_classifier"].predict(X_old_scaled)
v1_contact_pred = v1["contact_classifier"].predict(X_v1_scaled)
v2_contact_pred = v2["contact_classifier"].predict(X_v2_scaled)

y_reg = df[v2["regression_output_cols"]].values
y_contact = df[v2["contact_col"]].values


def mae(true, pred):
    return np.abs(true - pred).mean()


print("\n" + "=" * 78)
print("(a) UNITS 1+2 (jig) test rows -- Force/Theta/Contact: v2 vs original vs v1")
print("=" * 78)
n12 = unit12_mask.sum()
print(f"n={n12}")
force_mae_old12 = mae(y_reg[unit12_mask, 0], old_reg_pred[unit12_mask, 0])
force_mae_v112 = mae(y_reg[unit12_mask, 0], v1_reg_pred[unit12_mask, 0])
force_mae_v212 = mae(y_reg[unit12_mask, 0], v2_reg_pred[unit12_mask, 0])
print(f"  Force MAE:    original={force_mae_old12:.4f} N   v1(forcefix)={force_mae_v112:.4f} N   "
      f"v2(balanced)={force_mae_v212:.4f} N")

theta_mae_old12 = mae(y_reg[unit12_mask, 1], old_reg_pred[unit12_mask, 1])
theta_mae_v112 = mae(y_reg[unit12_mask, 1], v1_reg_pred[unit12_mask, 1])
theta_mae_v212 = mae(y_reg[unit12_mask, 1], v2_reg_pred[unit12_mask, 1])
print(f"  Theta MAE:    original={theta_mae_old12:.4f} deg  v1(forcefix)={theta_mae_v112:.4f} deg  "
      f"v2(balanced)={theta_mae_v212:.4f} deg")

contact_acc_old12 = accuracy_score(y_contact[unit12_mask], old_contact_pred[unit12_mask])
contact_acc_v112 = accuracy_score(y_contact[unit12_mask], v1_contact_pred[unit12_mask])
contact_acc_v212 = accuracy_score(y_contact[unit12_mask], v2_contact_pred[unit12_mask])
print(f"  Contact acc:  original={contact_acc_old12:.4f} ({contact_acc_old12:.1%})   "
      f"v1(forcefix)={contact_acc_v112:.4f} ({contact_acc_v112:.1%})   "
      f"v2(balanced)={contact_acc_v212:.4f} ({contact_acc_v212:.1%})")

# Also report bias in the light-touch band (0.3-1.0N), since that's what most
# directly governs perceived sensitivity/lag near the contact threshold.
for lo, hi in [(0.0, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 100.0)]:
    m = unit12_mask & (df["Force"].values >= lo) & (df["Force"].values < hi)
    if m.sum() == 0:
        continue
    bias_old = (old_reg_pred[m, 0] - y_reg[m, 0]).mean()
    bias_v1 = (v1_reg_pred[m, 0] - y_reg[m, 0]).mean()
    bias_v2 = (v2_reg_pred[m, 0] - y_reg[m, 0]).mean()
    print(f"    Force in [{lo:>4.1f},{hi:>5.1f})N n={m.sum():6d}  "
          f"bias: original={bias_old:+.3f}  v1={bias_v1:+.3f}  v2={bias_v2:+.3f}")

print("\n" + "=" * 78)
print("(b) UNIT 3 (backregion) test rows -- Force accuracy: v2 vs original vs v1")
print("=" * 78)
n3 = unit3_mask.sum()
print(f"n={n3}")
force_mae_old3 = mae(y_reg[unit3_mask, 0], old_reg_pred[unit3_mask, 0])
force_mae_v13 = mae(y_reg[unit3_mask, 0], v1_reg_pred[unit3_mask, 0])
force_mae_v23 = mae(y_reg[unit3_mask, 0], v2_reg_pred[unit3_mask, 0])
print(f"  Force MAE:    original={force_mae_old3:.4f} N   v1(forcefix)={force_mae_v13:.4f} N   "
      f"v2(balanced)={force_mae_v23:.4f} N")
theta_mae_old3 = mae(y_reg[unit3_mask, 1], old_reg_pred[unit3_mask, 1])
theta_mae_v13 = mae(y_reg[unit3_mask, 1], v1_reg_pred[unit3_mask, 1])
theta_mae_v23 = mae(y_reg[unit3_mask, 1], v2_reg_pred[unit3_mask, 1])
print(f"  Theta MAE:    original={theta_mae_old3:.4f} deg  v1(forcefix)={theta_mae_v13:.4f} deg  "
      f"v2(balanced)={theta_mae_v23:.4f} deg")

print("\n" + "=" * 78)
print("(c) Per-channel scaler_ ratio -- v2 vs original (confirm no more extreme outliers)")
print("=" * 78)
ratio_v1 = v1["scaler"].scale_ / old["scaler"].scale_
ratio_v2 = v2["scaler"].scale_ / old["scaler"].scale_
print(f"v1(forcefix)/original scale_ ratio:  min={ratio_v1.min():.2f}  median={np.median(ratio_v1):.2f}  max={ratio_v1.max():.2f}")
print(f"v2(balanced)/original scale_ ratio:  min={ratio_v2.min():.2f}  median={np.median(ratio_v2):.2f}  max={ratio_v2.max():.2f}")
print("\nPer-channel breakdown (v1 ratio -> v2 ratio):")
for col, r1, r2 in zip(v2["input_cols"], ratio_v1, ratio_v2):
    flag = "  <-- was the 396x/86x-style outlier" if r1 > 20 else ""
    print(f"  {col:10s}  v1={r1:7.2f}x   v2={r2:6.2f}x{flag}")

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"(a) Units 1+2 Force MAE   : original={force_mae_old12:.4f}  v1={force_mae_v112:.4f}  v2={force_mae_v212:.4f}  "
      f"(v2 vs original delta: {force_mae_v212 - force_mae_old12:+.4f} N)")
print(f"(a) Units 1+2 Contact acc : original={contact_acc_old12:.1%}  v1={contact_acc_v112:.1%}  v2={contact_acc_v212:.1%}  "
      f"(v2 vs original delta: {(contact_acc_v212 - contact_acc_old12) * 100:+.2f} pp)")
print(f"(b) Unit 3 Force MAE      : original={force_mae_old3:.4f}  v1={force_mae_v13:.4f}  v2={force_mae_v23:.4f}  "
      f"(v2 vs v1 delta: {force_mae_v23 - force_mae_v13:+.4f} N)")
print(f"(c) Max scale_ ratio      : v1={ratio_v1.max():.2f}x  v2={ratio_v2.max():.2f}x")
