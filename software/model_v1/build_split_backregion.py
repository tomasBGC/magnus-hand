"""Build train_data_backregion.csv / test_data_backregion.csv for
model_v1_contact_backregion from characterisation_delta_combined_backregion.csv,
holding out entire recordings (not individual rows) for the test set -- same
methodology as build_split.py, extended to cover the new Unit 3 backregion
recordings.

The original 10 held-out recordings (5 per unit, Unit 1 + Unit 2) are kept
EXACTLY as in build_split.py, so every row that was in the original model's
test set is still in this model's test set (the original model's reported
test performance remains reproducible/comparable from this split too).

Unit 3 (backregion, Theta=90 only, Phi in {-90,-45,0,45,90}) is a genuinely
small dataset: 5 recordings total, one per Phi value, nothing else. Phi=-45
and Phi=+45 do not appear ANYWHERE else in the combined dataset -- Unit 1
and Unit 2 only ever tested Phi in {-90,0,90}. That means a recording-level
holdout of the Phi=-45 or Phi=+45 recording would remove that class from
training entirely (the classifier would never learn it exists), which is
worse than not testing it. So the Unit 3 split is deliberately asymmetric:
  - TEST:  Phi=-90 and Phi=+90 recordings (theta=90) -- these Phi VALUES are
    already well represented elsewhere in train (Unit 1 + Unit 2, all
    thetas), so holding out the Unit-3-at-theta=90 INSTANCE of them is a
    fair generalization test, and it lets the old model (trained without
    any Unit 3 data at all, but which does know Phi classes -90/+90) be
    evaluated on the exact same rows for a like-for-like before/after
    comparison.
  - TRAIN: Phi=-45, Phi=0, Phi=+45 recordings (theta=90) -- includes both
    genuinely novel Phi classes, which must be in train for the retrained
    model to be able to predict them at all.

Consequence (documented, not hidden): Phi=-45 and Phi=+45 are NOT evaluated
in test_data_backregion.csv at all -- there is only one recording of each in
the entire dataset, and it has to go to train. Any claim about the model's
accuracy at those two specific angles is a training-set fit / interpolation
argument, not a held-out test result. See evaluate_model_contact_backregion.py.

Usage: uv run python model_v1/build_split_backregion.py
"""
import pandas as pd

SOURCE_CSV = "model_v1/characterisation_delta_combined_backregion.csv"
TRAIN_CSV = "model_v1/train_data_backregion.csv"
TEST_CSV = "model_v1/test_data_backregion.csv"

# Original held-out test recordings, unchanged from build_split.py.
ORIGINAL_TEST_RECORDINGS = {
    # Unit 1
    "eflesh_theta00_phi-90_0607_v1_Test15",   # theta=0,  phi=-90
    "eflesh_theta15_phi00_0607_v1_Test2",     # theta=15, phi=0
    "eflesh_theta30_phi90_0607_v1_Test10",    # theta=30, phi=90
    "eflesh_theta60_phi-90_0607_v1_Test19",   # theta=60, phi=-90
    "eflesh_theta75_phi00_0607_v1_Test6",     # theta=75, phi=0
    # Unit 2
    "eflesh_2_theta00_phi00_0807_v1_Test1",   # theta=0,  phi=0
    "eflesh_2_theta45_phi-90_0807_v1_Test18", # theta=45, phi=-90
    "eflesh_2_theta60_phi00_0807_v1_Test5",   # theta=60, phi=0
    "eflesh_2_theta90_phi00_0807_v1_Test7",   # theta=90, phi=0
    "eflesh_2_theta90_phi90_0807_v1_Test14",  # theta=90, phi=90
}

# New backregion (Unit 3) test recordings: the two Phi values (-90, +90)
# that are already covered elsewhere in train, so this is a fair recording
# -level holdout and a fair like-for-like comparison against the old model.
BACKREGION_TEST_RECORDINGS = {
    "eflesh_3_theta90_phi-90_1107_v1_Test3",  # theta=90, phi=-90
    "eflesh_3_theta90_phi90_1107_v1_Test2",   # theta=90, phi=90
}

# New backregion (Unit 3) train recordings: includes both novel Phi classes
# (-45, +45), which is why they cannot also be held out for test (see
# module docstring).
BACKREGION_TRAIN_RECORDINGS = {
    "eflesh_3_theta90_phi-45_1107_v1_Test4",  # theta=90, phi=-45 (novel class)
    "eflesh_3_theta90_phi00_1107_v1_Test1",   # theta=90, phi=0
    "eflesh_3_theta90_phi45_1107_v1_Test5",   # theta=90, phi=45  (novel class)
}

TEST_RECORDINGS = ORIGINAL_TEST_RECORDINGS | BACKREGION_TEST_RECORDINGS

df = pd.read_csv(SOURCE_CSV)
print(f"Loaded {len(df)} rows, {df['Recording Name'].nunique()} recordings from {SOURCE_CSV}")

all_recordings = set(df["Recording Name"].unique())
missing = TEST_RECORDINGS - all_recordings
assert not missing, f"Test recordings not found in source data: {missing}"
missing_train = BACKREGION_TRAIN_RECORDINGS - all_recordings
assert not missing_train, f"Backregion train recordings not found in source data: {missing_train}"

test_df = df[df["Recording Name"].isin(TEST_RECORDINGS)]
train_df = df[~df["Recording Name"].isin(TEST_RECORDINGS)]

train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

n_train_rec = train_df["Recording Name"].nunique()
n_test_rec = test_df["Recording Name"].nunique()
n_total_rec = n_train_rec + n_test_rec

print(f"\nTrain: {n_train_rec} recordings ({n_train_rec/n_total_rec:.1%}), {len(train_df)} rows -> {TRAIN_CSV}")
print(f"Test:  {n_test_rec} recordings ({n_test_rec/n_total_rec:.1%}), {len(test_df)} rows -> {TEST_CSV}")

print("\nHeld-out test recordings:")
test_summary = (
    test_df.groupby("Recording Name")
    .agg(Unit=("Unit", "first"), Theta=("Theta", "first"), Phi=("Phi", "first"), n=("Row Number", "count"))
    .sort_values(["Unit", "Theta", "Phi"])
)
print(test_summary.to_string())

print("\nBackregion (Unit 3) recordings, train vs test:")
u3 = df[df["Unit"] == 3]
u3_summary = (
    u3.groupby("Recording Name")
    .agg(Phi=("Phi", "first"), n=("Row Number", "count"))
    .sort_values("Phi")
)
u3_summary["split"] = [
    "TEST" if name in BACKREGION_TEST_RECORDINGS else "train"
    for name in u3_summary.index
]
print(u3_summary.to_string())

print("\nCoverage check:")
for name, sub in [("Train", train_df), ("Test", test_df)]:
    units = sorted(sub["Unit"].unique())
    thetas = sorted(sub["Theta"].unique())
    phis = sorted(sub["Phi"].unique())
    print(f"  {name}: Units={units}  Thetas={thetas}  Phis={phis}")

assert sorted(train_df["Unit"].unique()) == [1, 2, 3]
assert sorted(test_df["Unit"].unique()) == [1, 2, 3]
assert sorted(train_df["Theta"].unique()) == [0, 15, 30, 45, 60, 75, 90]
assert sorted(test_df["Theta"].unique()) == [0, 15, 30, 45, 60, 75, 90]
assert sorted(train_df["Phi"].unique()) == [-90, -45, 0, 45, 90]
test_phis = sorted(test_df["Phi"].unique())
assert test_phis == [-90, 0, 90], (
    f"Expected test Phi classes to be exactly [-90,0,90] (Phi=-45/+45 are "
    f"deliberately train-only, see module docstring), got {test_phis}"
)
print(
    "\nAll coverage checks passed: both original Units (1,2) plus the new "
    "Unit 3 present in both train and test, all 7 Thetas present in both, "
    "all 5 Phis present in train. Phi=-45/+45 are TRAIN-ONLY by necessity "
    "(only one Unit-3 recording exists for each) -- not evaluated in the "
    "held-out test set. See module docstring."
)
