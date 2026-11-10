"""(d) No-touch (control) force bias per finger: predicted force on control rows
(true force ~0), standard holdout and LOSO. bias = mean(pred - true)."""
import pandas as pd

import common as c

_, hold, _, _ = c.standard_predictions()
loso = c.loso_predictions()
rows = []
for split, d in [("standard", hold), ("loso", loso)]:
    d = d[d.level == "control"]
    for f, g in [(f, d[d.finger == f]) for f in c.FINGERS] + [("ALL", d)]:
        e = g.pred - g[c.TARGET]
        rows.append(dict(split=split, finger=f, n=len(g), mean_true=g[c.TARGET].mean(),
                         mean_pred=g.pred.mean(), bias=e.mean(), sd_pred=g.pred.std(),
                         mae=e.abs().mean(), frac_pred_over_0p3=(g.pred >= 0.3).mean()))
out = pd.DataFrame(rows)
print(out.round(4).to_string(index=False))
print(f"saved {c.save_csv(out, 'd_control_bias.csv')}")
