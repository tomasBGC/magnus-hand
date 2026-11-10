"""(f) Row-level predictions for the figures (intermediate, not a reported number).
Standard split: predicted with the SAVED candidate (models/grasp_trained_force/force_head_load_cell_candidate.pkl)
on its holdout reps. Leave-one-scenario-out: the 7 fold models are refit (seed 42), as in b_-e_.
Writes results/f_predictions.csv: split, fold, finger, level, true_force, pred_force."""
import joblib
import pandas as pd

import common as c

df = c.load_dataset()
cand = joblib.load(c.CANDIDATE_PATH)
hold = df[df["source_file"].isin(set(cand["holdout_reps"]))]
std = pd.DataFrame(dict(split="standard", fold="holdout", finger=hold.finger.values,
                        level=hold.level.values, true_force=hold[c.TARGET].values,
                        pred_force=cand["force_regressor"].predict(
                            cand["scaler"].transform(hold[cand["input_cols"]].values))))
loso = c.loso_predictions()
lo = pd.DataFrame(dict(split="loso", fold=loso.fold.values, finger=loso.finger.values,
                       level=loso.level.values, true_force=loso[c.TARGET].values,
                       pred_force=loso.pred.values))
out = pd.concat([std, lo], ignore_index=True)
print(f"standard n={len(std)}  loso n={len(lo)}  "
      f"standard MAE={abs(std.true_force - std.pred_force).mean():.3f} N")
print(f"saved {c.save_csv(out, 'f_predictions.csv')}")
