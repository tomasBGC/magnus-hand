"""Two-level (light vs firm) force analysis. Additive: writes ONLY to results_two_level/.

Variant T : label = load-cell force < THR ("light") / >= THR ("firm"); prediction
            classified with the same THR. Touch rows only (control excluded).
            Run at THR = 1.0 (main), 0.8, 1.2 (sensitivity).
Variant A : drop medium-cued grasps; label = original cue (light/firm); predicted class
            = predicted force vs the midpoint of the TRAIN-set light and firm mean
            load-cell force (per fold for LOSO).
Baselines : majority class (on the evaluated rows) and constant-mean predictor
            (train-set mean force over all training rows, as in b_force_error.py,
            classified with the same rule).
"""
import numpy as np
import pandas as pd

import common as c

OUT = c.HERE / "results_two_level"
OUT.mkdir(exist_ok=True)
THRS = [1.0, 0.8, 1.2]

train, hold, _, _ = c.standard_predictions()
hold["train_mean"] = train[c.TARGET].mean()
hold["train_mean_touch"] = train[train.level != "control"][c.TARGET].mean()
hold["fold"] = "holdout"
hold["bound_ab"] = (train[train.level == "light"][c.TARGET].mean()
                    + train[train.level == "firm"][c.TARGET].mean()) / 2
loso = c.loso_predictions()
# per-fold quantities
tm_touch, bab = {}, {}
for sc, tr, _ in c.loso_folds(c.load_dataset()):
    tm_touch[sc] = tr[tr.level != "control"][c.TARGET].mean()
    bab[sc] = (tr[tr.level == "light"][c.TARGET].mean() + tr[tr.level == "firm"][c.TARGET].mean()) / 2
loso["train_mean_touch"] = loso.fold.map(tm_touch)
loso["bound_ab"] = loso.fold.map(bab)
SPLITS = {"standard": hold, "loso": loso}
for k, d in SPLITS.items():
    d["split"] = k

# row-level predictions file (used by the figures)
pred_cols = ["split", "fold", "source_file", "finger", "level", c.TARGET, "pred", "train_mean", "bound_ab"]
pd.concat([hold[pred_cols], loso[pred_cols]]).rename(columns={c.TARGET: "true_force"}).to_csv(
    OUT / "predictions_rows.csv", index=False)


def metrics(y, p, base_pred):
    """y,p,base_pred are boolean 'firm' arrays. Returns dict of accuracy etc."""
    y, p, b = map(np.asarray, (y, p, base_pred))
    tp, tn = (p & y).sum(), (~p & ~y).sum()
    fp, fn = (p & ~y).sum(), (~p & y).sum()
    rec = [tn / (tn + fp) if tn + fp else np.nan, tp / (tp + fn) if tp + fn else np.nan]
    return dict(n=len(y), n_light=int((~y).sum()), n_firm=int(y.sum()),
                acc=(tp + tn) / len(y), balanced_acc=np.nanmean(rec),
                recall_light=rec[0], recall_firm=rec[1],
                majority_acc=max(y.mean(), 1 - y.mean()),
                const_mean_acc=float((b == y).mean()),
                tn_light_light=int(tn), fp_light_as_firm=int(fp),
                fn_firm_as_light=int(fn), tp_firm_firm=int(tp))


summ, perf = [], []
for thr in THRS:
    for sp, d in SPLITS.items():
        t = d[d.level != "control"]
        y = (t[c.TARGET] >= thr).values
        p = (t.pred >= thr).values
        b = (t.train_mean >= thr).values
        b2 = (t.train_mean_touch >= thr).values
        m = metrics(y, p, b)
        m["const_mean_acc_touch_mean"] = float((b2 == y).mean())
        summ.append(dict(variant="T", threshold=thr, split=sp, finger="ALL", **m))
        for f in c.FINGERS:
            g = t[t.finger == f]
            yf, pf = (g[c.TARGET] >= thr).values, (g.pred >= thr).values
            summ.append(dict(variant="T", threshold=thr, split=sp, finger=f,
                             **metrics(yf, pf, (g.train_mean >= thr).values)))
        if thr == 1.0:  # per-finger light/firm row counts, full touch dataset and per split
            for f in c.FINGERS:
                g = t[t.finger == f]
                perf.append(dict(split=sp, finger=f, n_light=int((g[c.TARGET] < thr).sum()),
                                 n_firm=int((g[c.TARGET] >= thr).sum())))
full = c.load_dataset()
ft = full[full.level != "control"]
for f in c.FINGERS:
    g = ft[ft.finger == f]
    perf.append(dict(split="full_dataset", finger=f, n_light=int((g[c.TARGET] < 1).sum()),
                     n_firm=int((g[c.TARGET] >= 1).sum())))

# Variant A
a_info = []
for sp, d in SPLITS.items():
    t = d[d.level.isin(["light", "firm"])]
    y = (t.level == "firm").values
    p = (t.pred >= t.bound_ab).values
    b = (t.train_mean >= t.bound_ab).values
    m = metrics(y, p, b)
    summ.append(dict(variant="A", threshold=np.nan, split=sp, finger="ALL", **m))
    for f in c.FINGERS:
        g = t[t.finger == f]
        if len(g):
            summ.append(dict(variant="A", threshold=np.nan, split=sp, finger=f,
                             **metrics((g.level == "firm").values, (g.pred >= g.bound_ab).values,
                                       (g.train_mean >= g.bound_ab).values)))
    a_info.append(dict(split=sp, n_grasps=t.source_file.nunique(), n_rows=len(t),
                       n_light_rows=int((~y).sum()), n_firm_rows=int(y.sum()),
                       boundary=(f"{t.bound_ab.iloc[0]:.4f}" if sp == "standard"
                                 else "per-fold " + ", ".join(f"{k}={v:.3f}" for k, v in bab.items()))))
a_info.append(dict(split="full_dataset", n_grasps=ft[ft.level != "medium"].source_file.nunique(),
                   n_rows=int((ft.level != "medium").sum()),
                   n_light_rows=int((ft.level == "light").sum()), n_firm_rows=int((ft.level == "firm").sum()),
                   boundary=""))
S = pd.DataFrame(summ)
S.to_csv(OUT / "summary.csv", index=False)
pd.DataFrame(perf).to_csv(OUT / "per_finger_label_counts_T1.0.csv", index=False)
pd.DataFrame(a_info).to_csv(OUT / "variantA_sizes.csv", index=False)

# cued-light grasps at/above 1.0 N, by finger. A "grasp" = one rep (source_file).
gl = ft[ft.level == "light"].groupby(["finger", "source_file"])[c.TARGET].agg(["mean", "min", "max", "size"])
gl["rows_ge1"] = ft[ft.level == "light"].assign(ge=lambda x: x[c.TARGET] >= 1).groupby(
    ["finger", "source_file"]).ge.sum()
rows = []
for f in c.FINGERS:
    g = gl.loc[f]
    lr = ft[(ft.level == "light") & (ft.finger == f)]
    rows.append(dict(finger=f, n_cued_light_grasps=len(g),
                     grasps_mean_ge_1N=int((g["mean"] >= 1).sum()),
                     grasps_any_row_ge_1N=int((g["max"] >= 1).sum()),
                     grasps_all_rows_ge_1N=int((g["min"] >= 1).sum()),
                     n_cued_light_rows=len(lr), rows_ge_1N=int((lr[c.TARGET] >= 1).sum()),
                     frac_rows_ge_1N=float((lr[c.TARGET] >= 1).mean()),
                     mean_force_N=float(lr[c.TARGET].mean())))
tot = pd.DataFrame(rows)
tot.loc[len(tot)] = dict(finger="ALL", **{k: tot[k].sum() for k in tot.columns[1:] if k not in
                                           ("frac_rows_ge_1N", "mean_force_N")},
                         frac_rows_ge_1N=tot.rows_ge_1N.sum() / tot.n_cued_light_rows.sum(),
                         mean_force_N=float(ft[ft.level == "light"][c.TARGET].mean()))
tot.to_csv(OUT / "cued_light_ge_1N.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 40)
show = ["variant", "threshold", "split", "finger", "n", "n_light", "n_firm", "acc", "balanced_acc",
        "majority_acc", "const_mean_acc", "tn_light_light", "fp_light_as_firm", "fn_firm_as_light", "tp_firm_firm"]
print(S[S.finger == "ALL"][show + ["const_mean_acc_touch_mean"]].round(4).to_string(index=False))
print(S[(S.finger != "ALL") & (S.threshold.fillna(1.0) == 1.0)][show].round(3).to_string(index=False))
print(pd.DataFrame(perf).to_string(index=False))
print(pd.DataFrame(a_info).to_string(index=False))
print(tot.round(3).to_string(index=False))
