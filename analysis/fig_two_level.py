"""Variant-T (1.0 N threshold) versions of the three paper figures, in Okabe-Ito colour.
Reads results_two_level/ only; writes figures_two_level/ (PDF + preview/PNG). Layout follows fig1-3."""
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import fig_style as s

THR = 1.0
R = s.HERE / "results_two_level"
s.FIGURES = s.HERE / "figures_two_level"
s.PREVIEWS = s.FIGURES / "preview"
# Okabe-Ito
BLUE, ORANGE, GREEN, PINK, VERM = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"
COL = dict(Index=BLUE, Middle=ORANGE, Ring=GREEN, Thumb=PINK)
MK = {f: s.MARKER_STYLE[f][0] for f in s.FINGERS}  # same marker shapes -> still separable in greyscale
OPEN = {"Ring"}  # open marker, as in the greyscale originals

p = pd.read_csv(R / "predictions_rows.csv")
p = p[p.level != "control"].copy()
p["y"] = p.true_force >= THR
p["yhat"] = p.pred >= THR
S = pd.read_csv(R / "summary.csv")
S = S[(S.variant == "T") & (S.threshold == THR) & (S.finger == "ALL")].set_index("split")

# ---- Figure 1: predicted vs true force ----
lim = (-0.5, 3.5)
fig, axes = s.plt.subplots(1, 2, figsize=s.TWO_PANEL, sharex=True, sharey=True)
for ax, split, tag in zip(axes, ["standard", "loso"], ["(a)", "(b)"]):
    g = p[p.split == split]
    for f in s.FINGERS:
        h = g[g.finger == f]
        face = "none" if f in OPEN else COL[f]
        ax.scatter(h.true_force, h.pred, s=6, marker=MK[f], facecolors=face, edgecolors=COL[f],
                   linewidths=0.4, alpha=0.5, label=f)
    ax.plot(lim, lim, color="k", lw=0.8, ls="--")
    ax.axvline(THR, color=VERM, lw=0.9, ls=":")
    ax.axhline(THR, color=VERM, lw=0.9, ls=":")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Load-cell force (N)")
    ax.text(0.03, 0.97, tag, transform=ax.transAxes, va="top", fontweight="bold")
    r = np.corrcoef(g.true_force, g.pred)[0, 1]
    mae = np.abs(g.true_force - g.pred).mean()
    ax.text(0.17, 0.97, f"MAE {mae:.2f} N, r = {r:.2f}", transform=ax.transAxes, ha="left", va="top")
    print(tag, "outside y range:", int(((g.pred < lim[0]) | (g.pred > lim[1])).sum()), "of", len(g))
axes[0].set_ylabel("Predicted force (N)")
leg = axes[0].legend(loc="lower right", frameon=False, markerscale=1.6, handletextpad=0.2, labelspacing=0.25)
for h in leg.legend_handles:
    h.set_alpha(1)
fig.tight_layout()
print("saved", s.save(fig, "fig_force_pred_vs_true.pdf"))

# ---- Figure 2: 2x2 confusion ----
cmap = LinearSegmentedColormap.from_list("oi_blue", ["#FFFFFF", BLUE])
LV = ["light", "firm"]
fig, axes = s.plt.subplots(1, 2, figsize=s.TWO_PANEL, layout="constrained")
for ax, split, tag in zip(axes, ["standard", "loso"], ["(a)", "(b)"]):
    g = p[p.split == split]
    counts = pd.crosstab(g.y, g.yhat).reindex(index=[False, True], columns=[False, True], fill_value=0).to_numpy()
    frac = counts / counts.sum(axis=1, keepdims=True)
    im = ax.imshow(frac, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{frac[i, j] * 100:.0f}%\n(n={counts[i, j]})", ha="center", va="center",
                    color="white" if frac[i, j] > 0.5 else "black")
    ax.set_xticks(range(2), LV); ax.set_yticks(range(2), LV)
    ax.set_xlabel("Predicted grip level (1.0 N)")
    ax.set_ylabel("Measured grip level (1.0 N)")
    ax.text(-0.02, 1.02, tag, transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")
    ax.text(0.5, -0.30, f"Accuracy {S.loc[split, 'acc'] * 100:.1f}%, balanced {S.loc[split, 'balanced_acc'] * 100:.1f}%,\n"
            f"majority-class {S.loc[split, 'majority_acc'] * 100:.1f}%", transform=ax.transAxes, ha="center", va="top")
    for sp in ax.spines.values():
        sp.set_visible(False)
fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02, label="Fraction of true level")
print("saved", s.save(fig, "fig_level_confusion.pdf"))

# ---- Figure 3: per-finger mean +/- SD load-cell force, light vs firm (1.0 N label) ----
full = pd.read_csv(s.HERE.parent / "data" / "load_cell_dataset.csv")
full = full[full.level != "control"].copy()
full["lab"] = np.where(full.Force >= THR, "firm", "light")
fig, ax = s.plt.subplots(figsize=s.ONE_PANEL)
offs = np.linspace(-0.27, 0.27, 4)
for off, f in zip(offs, s.FINGERS):
    g = full[full.finger == f].groupby("lab").Force.agg(["mean", "std", "size"])
    xs, ms, sds = [], [], []
    for k, lv in enumerate(LV):
        if lv in g.index:
            xs.append(k + off); ms.append(g.loc[lv, "mean"]); sds.append(g.loc[lv, "std"])
    n = ", ".join(f"{int(g.loc[lv, 'size'])}" if lv in g.index else "0" for lv in LV)
    ax.errorbar(xs, ms, yerr=sds, fmt=MK[f], mfc="white" if f in OPEN else COL[f], mec=COL[f], color=COL[f],
                ms=4, mew=0.8, capsize=2, elinewidth=0.8, lw=0, label=f"{f} (n = {n})")
ax.axhline(THR, color=VERM, lw=0.9, ls=":")
ax.set_xticks(range(2), LV)
ax.set_xlabel("Measured grip level (1.0 N)")
ax.set_ylabel("Load-cell force (N)")
ax.set_xlim(-0.5, 1.5); ax.set_ylim(0, 3.2)
ax.legend(loc="upper left", frameon=False, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25)
fig.tight_layout()
print("saved", s.save(fig, "fig_level_means.pdf"))
print(full.groupby(["finger", "lab"]).Force.agg(["mean", "std", "size"]).round(3))
