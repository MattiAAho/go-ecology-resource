"""
plot_monopoly_heatmap.py — Monopoly rate heatmap: k × grid_size, ER and SBM.

Layout: 2 rows (ER top, SBM bottom) × 3 columns
  col 0: Synchronous scheduler
  col 1: Gillespie scheduler
  col 2: Sync − Gillespie difference (scheduler effect magnitude)

Monopoly defined as ci_2p(r) < 0.1  (stones-normalised coexistence index).
Colour scale: green (0%) → red (100%) for main panels;
              white (0) → orange/red for the difference panel.

Usage:
    python plot_monopoly_heatmap.py
    python plot_monopoly_heatmap.py results/two_p_all_v1.pkl

Output:
    figures/fig_monopoly_heatmap.pdf / .png
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FixedLocator

from analysis_helpers import ci_2p

# ── load ──────────────────────────────────────────────────────────────────────

PKL_PATH    = sys.argv[1] if len(sys.argv) > 1 else "results/two_p_all_v1.pkl"
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

print(f"Loading {PKL_PATH} …")
with open(PKL_PATH, "rb") as fh:
    raw: list[dict] = pickle.load(fh)
print(f"  {len(raw)} total records")

valid = [
    r for r in raw
    if not r.get("connectivity_failed")
    and r.get("black_occupancy") is not None
]
print(f"  {len(valid)} valid records")

# ── constants ─────────────────────────────────────────────────────────────────

K_VALUES    = [8, 10, 12, 14, 19]
GRID_SIZES  = [9, 15, 19, 25]          # → n = 81, 225, 361, 625
N_VALUES    = [gs**2 for gs in GRID_SIZES]
GRAPH_TYPES = ["er", "sbm", "dc_sbm"]
GRAPH_LABEL = {"er": "ER (random)", "sbm": "SBM (modular)", "dc_sbm": "DC-SBM (modular)"}
SCHEDULERS  = ["sync", "gillespie_occ"]
SCHED_LABEL = {"sync": "Synchronous", "gillespie_occ": "Gillespie-like"}

# colourmap: green → yellow → red  (low monopoly = good for coexistence)
CMAP_MONO = mcolors.LinearSegmentedColormap.from_list(
    "mono", ["#1a9641", "#ffffbf", "#d73027"], N=256
)

MONO_THRESH = 0.1   # ci_2p < this → monopoly


def monopoly_rate(runs: list[dict]) -> float | None:
    if not runs:
        return None
    return float(np.mean([ci_2p(r) < MONO_THRESH for r in runs]))


def build_matrix(graph_type: str, scheduler: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (rate_matrix, count_matrix) shaped (n_grid_sizes, n_k_values).
    Rows = grid sizes ascending, cols = k ascending.
    """
    rates  = np.full((len(GRID_SIZES), len(K_VALUES)), np.nan)
    counts = np.zeros_like(rates, dtype=int)
    for gi, gs in enumerate(GRID_SIZES):
        for ki, k in enumerate(K_VALUES):
            subset = [
                r for r in valid
                if r["_graph_type"] == graph_type
                and r["_scheduler"]  == scheduler
                and r["_grid_size"]  == gs
                and r["_target_k"]   == k
            ]
            if subset:
                rates[gi, ki]  = monopoly_rate(subset)
                counts[gi, ki] = len(subset)
    return rates, counts


# ── figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(
    3, 2,
    figsize=(10, 11),
    constrained_layout=True,
)

norm_mono = mcolors.Normalize(vmin=0.0, vmax=1.0)

last_im = None
for row_i, gtype in enumerate(GRAPH_TYPES):
    for col_i, scheduler in enumerate(SCHEDULERS):
        ax = axes[row_i, col_i]
        mat, _ = build_matrix(gtype, scheduler)

        last_im = ax.imshow(
            mat, aspect="auto", origin="lower",
            cmap=CMAP_MONO, norm=norm_mono,
            extent=[-0.5, len(K_VALUES) - 0.5,
                    -0.5, len(GRID_SIZES) - 0.5],
        )

        # annotate cells
        for gi in range(len(GRID_SIZES)):
            for ki in range(len(K_VALUES)):
                v = mat[gi, ki]
                if np.isnan(v):
                    ax.text(ki, gi, "—", ha="center", va="center",
                            fontsize=9, color="0.5")
                    continue
                fc = "white" if (v < 0.25 or v > 0.75) else "black"
                ax.text(ki, gi, f"{v*100:.0f}%", ha="center", va="center",
                        fontsize=9, color=fc, fontweight="bold")

        ax.set_xticks(range(len(K_VALUES)))
        ax.set_xticklabels(K_VALUES, fontsize=9)
        ax.set_yticks(range(len(GRID_SIZES)))
        ax.set_yticklabels([f"n={gs**2}" for gs in GRID_SIZES], fontsize=9)

        if row_i == len(GRAPH_TYPES) - 1:
            ax.set_xlabel("Mean degree k", fontsize=10)
        if col_i == 0:
            ax.set_ylabel(f"{GRAPH_LABEL[gtype]}\n\nBoard size", fontsize=10)
        if row_i == 0:
            ax.set_title(SCHED_LABEL[scheduler], fontsize=11, fontweight="bold")

# single shared colorbar below all panels
cb = fig.colorbar(last_im, ax=axes.ravel().tolist(),
                  fraction=0.02, pad=0.03, orientation="horizontal")
cb.set_label("Dominance rate  (fraction of runs with CI < 0.10)", fontsize=10)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

out_pdf = FIGURES_DIR / "fig_monopoly_heatmap.pdf"
out_png = FIGURES_DIR / "fig_monopoly_heatmap.png"
fig.savefig(out_pdf, dpi=200)
fig.savefig(out_png, dpi=150)
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")
print("Done.")
