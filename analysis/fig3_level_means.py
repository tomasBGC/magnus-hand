"""Figure 3: per-finger load-cell force at each cued grip level, mean +/- 1 SD (full dataset).
Reads results/e_level_stats.csv and results/g_rep_counts.csv only."""
import numpy as np
import pandas as pd

import fig_style as s

d = pd.read_csv(s.RESULTS / "e_level_stats.csv")
d = d[(d.source == "true_force_full_dataset") & d.level.isin(s.LEVELS)]
reps = pd.read_csv(s.RESULTS / "g_rep_counts.csv")


def rep_label(f):
    r = reps[(reps.finger == f) & reps.level.isin(s.LEVELS)].reps
    return f"{r.min()}" if r.min() == r.max() else f"{r.min()}-{r.max()}"


fig, ax = s.plt.subplots(figsize=s.ONE_PANEL)
offsets = np.linspace(-0.27, 0.27, len(s.FINGERS))
for off, f in zip(offsets, s.FINGERS):
    g = d[d.finger == f].set_index("level").loc[s.LEVELS]
    mk, face, edge = s.MARKER_STYLE[f]
    ax.errorbar(np.arange(3) + off, g["mean"], yerr=g["sd"], fmt=mk, mfc=face if face != "none" else "white",
                mec=edge, color=edge, ms=4, mew=0.8, capsize=2, elinewidth=0.8, lw=0,
                label=f"{f} ({rep_label(f)} reps per level)")
ax.set_xticks(range(3), s.LEVELS)
ax.set_xlabel("Cued grip level")
ax.set_ylabel("Load-cell force (N)")
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(0, 3.2)
ax.legend(loc="upper left", frameon=False, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25)
fig.tight_layout()
print("saved", s.save(fig, "fig_level_means.pdf"))
