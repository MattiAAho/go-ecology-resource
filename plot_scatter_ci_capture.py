"""
plot_scatter_ci_capture.py — ci_2p vs capture rate, coloured by interblock_mix.
SBM only, Gillespie-like scheduler.

Outputs:
  fig_scatter_ci_capture_5k.pdf/png   — five panels, one per k value
  fig_scatter_ci_capture_pooled.pdf/png — single panel, all k pooled

fill  = interblock_mix  (viridis: yellow=high mixing, purple=low)
size  = grid size       (four sizes for n=81, 225, 361, 625)
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from analysis_helpers import ci_2p

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

print("Loading …")
with open("results/two_p_all_v1.pkl", "rb") as fh:
    raw = pickle.load(fh)

valid = [r for r in raw
         if not r.get("connectivity_failed")
         and r["_scheduler"] == "gillespie_occ"
         and r["_graph_type"] == "sbm"
         and r.get("black_occupancy") is not None
         and r.get("interblock_mix") is not None
         and r.get("capture_rate_per_move") is not None]

for r in valid:
    r["_ci2p"] = ci_2p(r)

print(f"  {len(valid)} valid SBM Gillespie-like runs")

# ── constants ─────────────────────────────────────────────────────────────────
K_VALUES = [8, 10, 12, 14, 19]
N_NODES  = [81, 225, 361, 625]
GS_LABEL = {81: "9×9", 225: "15×15", 361: "19×19", 625: "25×25"}
SIZE_MAP = {81: 18, 225: 40, 361: 70, 625: 110}

CMAP   = plt.cm.viridis
IB_MAX = 0.23
norm   = mcolors.Normalize(vmin=0.0, vmax=IB_MAX)
rng    = np.random.default_rng(42)

SUPTITLE_BASE = (
    "Two-player SBM — coexistence index vs capture rate\n"
    "Gillespie-like scheduler, all policies and grid sizes pooled  ·  "
    "colour = inter-block mixing  ·  size = board size"
)

size_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor="0.4", markeredgecolor="0.3",
           markersize=np.sqrt(SIZE_MAP[n]), label=GS_LABEL[n])
    for n in N_NODES
]

def draw_scatter(ax, subset, title=None, ylabel=True, xlabel=True):
    if not subset:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        return

    cap = np.array([r["capture_rate_per_move"] for r in subset])
    ci  = np.array([r["_ci2p"] for r in subset])
    ib  = np.array([r["interblock_mix"] for r in subset])
    sz  = np.array([SIZE_MAP[r["_n_nodes"]] for r in subset])

    jx = rng.normal(0, 0.003, len(cap))
    jy = rng.normal(0, 0.008, len(ci))

    ax.scatter(cap + jx, ci + jy, c=ib, s=sz,
               cmap=CMAP, norm=norm,
               alpha=0.55, linewidths=0.3, edgecolors="0.3")

    ax.set_xlim(0.60, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.axhline(0.1, color="0.5", lw=0.6, ls="--", alpha=0.5)
    ax.axhline(0.9, color="0.5", lw=0.6, ls="--", alpha=0.5)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    if xlabel:
        ax.set_xlabel("Capture rate per move", fontsize=9)
    if ylabel:
        ax.set_ylabel("ci_2p (stones-norm.)", fontsize=9.5)

    n_dom = np.sum(ci < 0.10)
    n_seg = np.sum(ci > 0.90)
    ax.text(0.02, 0.97, f"dom {n_dom}", transform=ax.transAxes,
            fontsize=7, va="top", color="0.3")
    ax.text(0.02, 0.03, f"seg {n_seg}", transform=ax.transAxes,
            fontsize=7, va="bottom", color="0.3")

def add_colorbar(fig, axes_list):
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes_list, fraction=0.012, pad=0.01, aspect=35)
    cb.set_label("Inter-block mixing  (0 = confined, yellow = cross-block)", fontsize=9)
    cb.set_ticks([0, 0.05, 0.10, 0.15, 0.20, IB_MAX])
    cb.set_ticklabels(["0", "0.05", "0.10", "0.15", "0.20", f"≥{IB_MAX}"])

# ── Figure 1: five panels by k ────────────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 5, figsize=(17, 4.2),
                            sharex=True, sharey=True, constrained_layout=True)
fig1.suptitle(SUPTITLE_BASE, fontsize=11, fontweight="bold")

for col_i, k in enumerate(K_VALUES):
    subset = [r for r in valid if r["_target_k"] == k]
    draw_scatter(axes1[col_i], subset,
                 title=f"k = {k}",
                 ylabel=(col_i == 0),
                 xlabel=True)

add_colorbar(fig1, axes1.tolist())
axes1[2].legend(handles=size_handles, title="Board size", title_fontsize=8,
                fontsize=8, loc="lower center",
                bbox_to_anchor=(0.5, -0.32), ncol=4, framealpha=0.85)

out = FIGURES_DIR / "fig_scatter_ci_capture_5k.pdf"
fig1.savefig(out, dpi=200)
fig1.savefig(out.with_suffix(".png"), dpi=150)
print(f"Saved {out}")

# ── Figure 2: single panel, all k pooled ─────────────────────────────────────
fig2, ax2 = plt.subplots(1, 1, figsize=(5.5, 4.8), constrained_layout=True)
fig2.suptitle(
    "Two-player SBM — coexistence index vs capture rate\n"
    "Gillespie-like, all k and grid sizes pooled",
    fontsize=11, fontweight="bold",
)

# colour k values differently via marker instead of overloading ib colour
# — keep ib as colour, add k as annotation in title; all k pooled means
#   the two-cloud structure should still be visible across the full dataset
draw_scatter(ax2, valid, ylabel=True, xlabel=True)
add_colorbar(fig2, [ax2])
ax2.legend(handles=size_handles, title="Board size", title_fontsize=8,
           fontsize=8, loc="upper left", framealpha=0.85)

out2 = FIGURES_DIR / "fig_scatter_ci_capture_pooled.pdf"
fig2.savefig(out2, dpi=200)
fig2.savefig(out2.with_suffix(".png"), dpi=150)
print(f"Saved {out2}")
print("Done.")
