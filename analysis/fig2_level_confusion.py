"""Figure 2: row-normalised 3-class force-level confusion matrices (pooled over fingers),
standard split (a) and leave-one-scenario-out (b), with raw counts and overall accuracy.
Reads results/c_level_confusion.csv only."""
import pandas as pd

import fig_style as s

d = pd.read_csv(s.RESULTS / "c_level_confusion.csv")
pred_cols = [f"pred_{lv}" for lv in s.LEVELS]
fig, axes = s.plt.subplots(1, 2, figsize=s.TWO_PANEL, layout="constrained")
for ax, split, tag in zip(axes, ["standard", "loso"], ["(a)", "(b)"]):
    g = d[(d.split == split) & (d.finger == "ALL")].set_index("true_level").loc[s.LEVELS]
    counts = g[pred_cols].to_numpy()
    frac = counts / counts.sum(axis=1, keepdims=True)
    im = ax.imshow(frac, cmap="Greys", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{frac[i, j] * 100:.0f}%\n(n={counts[i, j]})", ha="center", va="center",
                    color="white" if frac[i, j] > 0.5 else "black")
    ax.set_xticks(range(3), s.LEVELS)
    ax.set_yticks(range(3), s.LEVELS)
    ax.set_xlabel("Predicted grip level")
    ax.set_ylabel("Cued grip level")
    ax.text(-0.02, 1.02, tag, transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")
    acc = g["accuracy_all_levels"].iloc[0]
    ax.text(0.5, -0.30, f"Overall accuracy {acc * 100:.1f}%", transform=ax.transAxes,
            ha="center", va="top")
    for sp in ax.spines.values():
        sp.set_visible(False)
fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02, label="Fraction of true level")
print("saved", s.save(fig, "fig_level_confusion.pdf"))
