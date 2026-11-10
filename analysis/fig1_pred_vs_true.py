"""Figure 1: predicted vs true force, standard split (a) and leave-one-scenario-out (b).
Reads results/f_predictions.csv (points) and results/b_force_error.csv (MAE, r) only."""
import pandas as pd

import fig_style as s

d = pd.read_csv(s.RESULTS / "f_predictions.csv")
b = pd.read_csv(s.RESULTS / "b_force_error.csv")


def pooled(split, fold):
    r = b[(b.split == split) & (b.fold == fold) & (b.subset == "all_rows") & (b.finger == "ALL")]
    assert len(r) == 1
    return r.iloc[0]


panels = [("standard", "pooled", "(a)"), ("loso", "pooled_all_folds", "(b)")]
lim = (-0.5, 3.5)
fig, axes = s.plt.subplots(1, 2, figsize=s.TWO_PANEL, sharex=True, sharey=True)
for ax, (split, fold, tag) in zip(axes, panels):
    g = d[d.split == split]
    for f in s.FINGERS:
        h = g[g.finger == f]
        mk, face, edge = s.MARKER_STYLE[f]
        ax.scatter(h.true_force, h.pred_force, s=6, marker=mk, facecolors=face, edgecolors=edge,
                   linewidths=0.4, alpha=0.5, label=f)
    ax.plot(lim, lim, color="k", lw=0.8, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Load-cell force (N)")
    ax.text(0.03, 0.97, tag, transform=ax.transAxes, va="top", fontweight="bold")
    m = pooled(split, fold)
    ax.text(0.17, 0.97, f"MAE {m.mae:.2f} N, r = {m['corr']:.2f}", transform=ax.transAxes,
            ha="left", va="top")
    out = ((g.pred_force < lim[0]) | (g.pred_force > lim[1])).sum()
    print(f"panel {tag}: {out} of {len(g)} predictions fall outside the y-axis range")
axes[0].set_ylabel("Predicted force (N)")
leg = axes[0].legend(loc="lower right", frameon=False, markerscale=1.6,
                     handletextpad=0.2, labelspacing=0.25)
for h in leg.legend_handles:
    h.set_alpha(1)
fig.tight_layout()
print("saved", s.save(fig, "fig_force_pred_vs_true.pdf"))
