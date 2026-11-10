"""(c) Force-level 3-class (light/medium/firm) confusion matrices, standard and
LOSO. Predicted level = predicted force binned at midpoints of the TRAIN-set
class-mean load-cell force (per fold for LOSO). Touch rows only (control has no level)."""
import numpy as np
import pandas as pd

import common as c

train, hold, _, _ = c.standard_predictions()
b = c.level_boundaries(train)
hold["lvl_pred"] = c.to_level(hold.pred.values, b)
loso = c.loso_predictions()
loso["lvl_pred"] = [c.to_level(np.array([p]), (lo, hi))[0]
                    for p, lo, hi in zip(loso.pred, loso.bound_lo, loso.bound_hi)]
fold_bounds = loso.groupby("fold")[["bound_lo", "bound_hi"]].first()
bounds = {"standard": f"{b[0]:.4f}|{b[1]:.4f}",
          "loso": "per-fold: " + "; ".join(f"{k}={r.bound_lo:.3f}|{r.bound_hi:.3f}"
                                           for k, r in fold_bounds.iterrows())}
rows, acc = [], {}
for split, d in [("standard", hold), ("loso", loso)]:
    d = d[d.level != "control"]
    for f in ["ALL"] + c.FINGERS:
        g = d if f == "ALL" else d[d.finger == f]
        cm = pd.crosstab(g.level, g.lvl_pred).reindex(index=c.LEVELS, columns=c.LEVELS, fill_value=0)
        for t in c.LEVELS:
            rows.append(dict(split=split, finger=f, true_level=t,
                             **{f"pred_{p}": int(cm.loc[t, p]) for p in c.LEVELS}))
        acc[(split, f)] = float(np.trace(cm.values) / cm.values.sum())
out = pd.DataFrame(rows)
out["boundaries"] = out.split.map(bounds)
out["accuracy_all_levels"] = [acc[(s, f)] for s, f in zip(out.split, out.finger)]
print(out.drop(columns="boundaries").to_string(index=False))
print(f"saved {c.save_csv(out, 'c_level_confusion.csv')}")
print("Check vs earlier (pooled accuracy):")
c.compare("standard", acc[("standard", "ALL")] * 100, 76, 2, " %")
c.compare("LOSO", acc[("loso", "ALL")] * 100, 46, 2, " %")
