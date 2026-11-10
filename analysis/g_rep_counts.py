"""(g) Number of reps (distinct source_file) per finger and cued grip level in
load_cell_dataset.csv, whole dataset. Feeds the Fig 3 legend; no model involved."""
import pandas as pd

import common as c

df = c.load_dataset()
out = (df.groupby(["finger", "level"]).source_file.nunique().rename("reps").reset_index())
out["finger"] = pd.Categorical(out.finger, c.FINGERS, ordered=True)
out = out.sort_values(["finger", "level"]).reset_index(drop=True)
out["finger"] = out.finger.astype(str)
print(out.pivot(index="finger", columns="level", values="reps").loc[c.FINGERS].to_string())
print(f"saved {c.save_csv(out, 'g_rep_counts.csv')}")
