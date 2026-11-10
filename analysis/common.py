"""Shared setup for the Force-head analysis scripts in analysis/.

Reads ONLY data/load_cell_dataset.csv
(plus the scaler and deployed contact classifier from models/deployed/active_model.pkl, which
is loaded read-only and never written).

Split: copied from software/model_v1/grasping_dashboard/retrain_contact_with_load_cell_data.py
(_split_load_cell_dataset): rep-grouped (source_file), stratified per finger,
20% of reps per finger held out, one shared np.random.default_rng(42) consumed
in groupby("finger") order -- so the holdout reps are identical to that script's.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # repository root; scripts can be run from anywhere
ACTIVE_MODEL_PATH = REPO / "models" / "deployed" / "active_model.pkl"
DATASET_CSV = REPO / "data" / "load_cell_dataset.csv"
# Optional: set EFLESH_OUT_DIR to write results/ and models/ somewhere else
# (e.g. a temporary folder) instead of overwriting the committed ones.
_OUT = os.environ.get("EFLESH_OUT_DIR")
RESULTS_DIR = Path(_OUT) / "results" if _OUT else HERE / "results"
MODELS_DIR = Path(_OUT) / "models" if _OUT else REPO / "models" / "grasp_trained_force"
CANDIDATE_PATH = MODELS_DIR / "force_head_load_cell_candidate.pkl"

SEED = 42
HOLDOUT_FRACTION = 0.2
FINGERS = ["Index", "Middle", "Ring", "Thumb"]
LEVELS = ["light", "medium", "firm"]
TARGET = "Force"  # the dataset's column holding the load-cell force (N)

DELTA_COLS = [f"d{c}" for c in [f"S{i}{axis}" for i in range(1, 6) for axis in "XYZ"]]
MAGNITUDE_COLS = [f"S{i}_mag" for i in range(1, 6)]
INPUT_COLS = DELTA_COLS + MAGNITUDE_COLS

# Deployed Force head: hidden (64, 32, 16), ReLU, Adam, lr 1e-3, alpha 1e-4,
# early stopping, n_iter_no_change 15, max_iter 500 (sklearn defaults otherwise,
# matching the deployed regressor's repr).
MLP_KWARGS = dict(
    hidden_layer_sizes=(64, 32, 16), activation="relu", solver="adam",
    alpha=1e-4, learning_rate_init=1e-3, max_iter=500, early_stopping=True,
    n_iter_no_change=15, random_state=SEED,
)


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_CSV)


def load_bundle() -> dict:
    return joblib.load(ACTIVE_MODEL_PATH)  # read-only


def standard_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    train_frames, holdout_frames = [], []
    for _, sub in df.groupby("finger"):
        reps = np.array(sub["source_file"].unique(), dtype=object)
        rng.shuffle(reps)
        n_holdout = max(1, round(len(reps) * HOLDOUT_FRACTION))
        holdout_reps = set(reps[:n_holdout])
        holdout_frames.append(sub[sub["source_file"].isin(holdout_reps)])
        train_frames.append(sub[~sub["source_file"].isin(holdout_reps)])
    return (pd.concat(train_frames, ignore_index=True),
            pd.concat(holdout_frames, ignore_index=True))


def loso_folds(df: pd.DataFrame):
    """Leave-one-scenario-out: yields (scenario, train_df, test_df)."""
    for scenario in sorted(df["scenario"].unique()):
        yield scenario, df[df["scenario"] != scenario], df[df["scenario"] == scenario]


def fit_force(train: pd.DataFrame, scaler) -> MLPRegressor:
    reg = MLPRegressor(**MLP_KWARGS)
    reg.fit(scaler.transform(train[INPUT_COLS].values), train[TARGET].values)
    return reg


def predict_force(reg, scaler, df: pd.DataFrame) -> np.ndarray:
    return reg.predict(scaler.transform(df[INPUT_COLS].values))


def level_boundaries(train: pd.DataFrame) -> tuple[float, float]:
    """light|medium and medium|firm cut points: midpoints of the TRAIN-set
    mean load-cell force of each level (touch rows only)."""
    m = train[train["level"].isin(LEVELS)].groupby("level")[TARGET].mean()
    return (m["light"] + m["medium"]) / 2, (m["medium"] + m["firm"]) / 2


def to_level(force: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return np.array(LEVELS, dtype=object)[np.digitize(force, bounds)]


def force_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    r = np.corrcoef(y, p)[0, 1] if len(y) > 1 and np.std(p) > 0 and np.std(y) > 0 else np.nan
    return dict(n=len(y), mae=float(np.mean(np.abs(y - p))),
                rmse=float(np.sqrt(np.mean((y - p) ** 2))), corr=float(r))


def standard_predictions():
    """Fit on the standard train split; return (train, holdout_with_pred, reg, scaler)."""
    df = load_dataset()
    scaler = load_bundle()["scaler"]
    train, hold = standard_split(df)
    reg = fit_force(train, scaler)
    hold = hold.copy()
    hold["pred"] = predict_force(reg, scaler, hold)
    return train, hold, reg, scaler


def loso_predictions() -> pd.DataFrame:
    """Out-of-scenario prediction for every row; adds 'pred', 'fold',
    'bound_lo', 'bound_hi' (boundaries from that fold's training data)."""
    df = load_dataset()
    scaler = load_bundle()["scaler"]
    out = []
    for scenario, tr, te in loso_folds(df):
        reg = fit_force(tr, scaler)
        te = te.copy()
        te["pred"] = predict_force(reg, scaler, te)
        te["train_mean"] = tr[TARGET].mean()
        te["bound_lo"], te["bound_hi"] = level_boundaries(tr)
        te["fold"] = scenario
        out.append(te)
    return pd.concat(out, ignore_index=True)


def save_csv(df: pd.DataFrame, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    df.to_csv(path, index=False)
    return path


def compare(label: str, value: float, earlier: float, tol: float, unit: str = "") -> None:
    diff = value - earlier
    flag = "  <-- FLAG" if abs(diff) > tol else ""
    print(f"  {label}: now {value:.3f}{unit} vs earlier {earlier:.3f}{unit} (diff {diff:+.3f}){flag}")
