"""Trains the Force-head candidate on the standard train split and saves it to
models/grasp_trained_force/. Nothing else is written; active_model.pkl is only read."""
import joblib

import common as c

df = c.load_dataset()
scaler = c.load_bundle()["scaler"]
train, hold = c.standard_split(df)
reg = c.fit_force(train, scaler)
c.MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(dict(force_regressor=reg, scaler=scaler, input_cols=c.INPUT_COLS,
                 target=c.TARGET, seed=c.SEED, hyperparameters=c.MLP_KWARGS,
                 n_train_rows=len(train), n_holdout_rows=len(hold),
                 holdout_reps=sorted(hold["source_file"].unique()),
                 note="Force-only pooled MLP trained on load_cell_dataset.csv train split; "
                      "scaler copied unchanged from active_model.pkl. Candidate, NOT deployed."),
            c.CANDIDATE_PATH)
mae = abs(hold[c.TARGET] - c.predict_force(reg, scaler, hold)).mean()
print(f"saved {c.CANDIDATE_PATH} (train={len(train)}, holdout={len(hold)}, "
      f"n_iter={reg.n_iter_}, holdout MAE={mae:.3f} N)")
