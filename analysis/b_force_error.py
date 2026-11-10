"""(b) Force MAE / RMSE / correlation per finger: standard split and
leave-one-scenario-out (7 folds), with the constant-mean baseline (predict the
training-set mean force). Subsets: all rows, and touch-only (control excluded)."""
import pandas as pd

import common as c

train, hold, _, _ = c.standard_predictions()
hold["train_mean"] = train[c.TARGET].mean()
loso = c.loso_predictions()


def block(d, split, key):
    rows = []
    for subset, s in [("all_rows", d), ("touch_only", d[d.level != "control"])]:
        groups = [(f, s[s.finger == f]) for f in c.FINGERS] + [("ALL", s)]
        for f, g in groups:
            m = c.force_metrics(g[c.TARGET], g.pred)
            bm = c.force_metrics(g[c.TARGET], g.train_mean)
            rows.append(dict(split=split, fold=key, subset=subset, finger=f, n=m["n"],
                             mae=m["mae"], rmse=m["rmse"], corr=m["corr"],
                             baseline_mae=bm["mae"], baseline_rmse=bm["rmse"]))
    return rows


rows = block(hold, "standard", "pooled") + block(loso, "loso", "pooled_all_folds")
for sc, g in loso.groupby("fold"):
    rows += [r for r in block(g, "loso", sc) if r["finger"] == "ALL"]
out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out[out.fold.isin(["pooled", "pooled_all_folds"])].round(3).to_string(index=False))
print(f"saved {c.save_csv(out, 'b_force_error.csv')}")
print("Check vs earlier (pooled, all rows):")


def sel(sp, k):
    return out[(out.split == sp) & (out.fold == k) & (out.subset == "all_rows")
               & (out.finger == "ALL")].iloc[0]


c.compare("MAE standard", sel("standard", "pooled").mae, 0.210, 0.03, " N")
c.compare("MAE LOSO", sel("loso", "pooled_all_folds").mae, 0.657, 0.03, " N")
