"""(e) Per-finger, per-level mean and SD of the true load-cell force (full
dataset) and of the model's predicted force (standard holdout and LOSO)."""
import pandas as pd

import common as c

df = c.load_dataset()
_, hold, _, _ = c.standard_predictions()
loso = c.loso_predictions()
rows = []
for src, d, col in [("true_force_full_dataset", df, c.TARGET),
                    ("pred_standard_holdout", hold, "pred"), ("pred_loso", loso, "pred")]:
    for f in c.FINGERS:
        for lv in ["control"] + c.LEVELS:
            g = d[(d.finger == f) & (d.level == lv)][col]
            rows.append(dict(source=src, finger=f, level=lv, n=len(g), mean=g.mean(), sd=g.std()))
out = pd.DataFrame(rows)
print(out.round(3).to_string(index=False))
print(f"saved {c.save_csv(out, 'e_level_stats.csv')}")
