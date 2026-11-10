"""Shared figure style. Figure scripts read only results/*.csv (no dataset, no model)."""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
# Optional: set EFLESH_OUT_DIR to read results/ from, and write figures/ to,
# a different folder (e.g. a temporary one).
_OUT = os.environ.get("EFLESH_OUT_DIR")
RESULTS = Path(_OUT) / "results" if _OUT else HERE / "results"
FIGURES = Path(_OUT) / "figures" if _OUT else HERE / "figures"
PREVIEWS = FIGURES / "preview"
FINGERS = ["Index", "Middle", "Ring", "Thumb"]
LEVELS = ["light", "medium", "firm"]
# Distinguishable by marker shape alone in greyscale print.
# (marker, face colour, edge colour); "none" face = open marker.
MARKER_STYLE = dict(
    Index=("o", "black", "black"),
    Middle=("s", "0.55", "0.55"),
    Ring=("^", "none", "0.1"),
    Thumb=("D", "0.3", "0.3"),
)
TWO_PANEL = (7.0, 2.6)
ONE_PANEL = (3.4, 2.4)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "font.family": "sans-serif",
    "pdf.fonttype": 42,  # TrueType; IEEE rejects Type 3
    "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name: str) -> Path:
    """Vector PDF to figures/, 200 dpi PNG preview to figures/preview/."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, format="pdf", metadata={"Title": None, "Creator": None, "Producer": None})
    png = PREVIEWS / (Path(name).stem + ".png")
    fig.savefig(png, format="png", dpi=200)
    plt.close(fig)
    print("saved preview", png)
    return path
