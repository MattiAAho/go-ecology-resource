"""
plot_bimodality_by_k.py — Bimodality by density (target_k), sync + Gillespie.

Two figures:
  figures/fig_bimodality_by_k_5x2.pdf   — 5 rows (k=8,10,12,14,19) × 2 cols
                                           (sync / Gillespie), primary figure
  figures/fig_bimodality_by_k_2x3.pdf   — 2×3 compact grid (k=8,10,12 top;
                                           k=14,19,pooled bottom), sync only

SBM (blue) + DC-SBM (orange) overlaid in each panel.
DC-SBM has no k=8 data; that panel shows SBM only.
Metric: stones-normalised CI via ci_2p(r).

Usage:
    python plot_bimodality_by_k.py
    python plot_bimodality_by_k.py results/two_p_all_v1.pkl
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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
    and r.get("_graph_type") in ("sbm", "dc_sbm")
    and r.get("black_occupancy") is not None
]
print(f"  {len(valid)} valid SBM/DC-SBM records")

# ── constants ─────────────────────────────────────────────────────────────────

K_VALUES    = [8, 10, 12, 14, 19]
SCHEDULERS  = [
    ("sync",          "Synchronous (round-robin)"),
    ("gillespie_occ", "Gillespie-like (energy-weighted async)"),
]
GRAPH_STYLE = {
    "sbm":    dict(color="#1f77b4", label="SBM",    alpha=0.55, zorder=3),
    "dc_sbm": dict(color="#ff7f0e", label="DC-SBM", alpha=0.55, zorder=2),
}
CI_GRID  = np.linspace(0.0, 1.0, 500)
BW       = 0.06


def kde(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    return gaussian_kde(v, bw_method=BW)(CI_GRID) if len(v) >= 2 else np.zeros_like(CI_GRID)


def fill_panel(ax, runs_by_gtype: dict[str, list], k: int, sched_key: str):
    """Draw overlaid SBM + DC-SBM densities on ax; return (mono%, coex%) combined."""
    any_data = False
    n_labels = []   # collect per-gtype n for panel annotation

    for gtype in ("sbm", "dc_sbm"):
        st    = GRAPH_STYLE[gtype]
        runs  = runs_by_gtype.get(gtype, [])
        if not runs:
            # Always include a line even when no data (e.g. DC-SBM at k=8)
            n_labels.append(f"{st['label']}: no k={k} data")
            continue
        ci_vals = [ci_2p(r) for r in runs]
        density = kde(ci_vals)
        n       = sum(np.isfinite(v) for v in ci_vals)
        n_labels.append(f"{st['label']} n={n:,}")

        ax.fill_between(CI_GRID, density, color=st["color"],
                        alpha=st["alpha"], zorder=st["zorder"],
                        label=st["label"])           # no n in legend
        ax.plot(CI_GRID, density, color=st["color"], lw=1.3,
                zorder=st["zorder"] + 1)
        any_data = True


    if not any_data:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        return None, None

    # Per-panel n annotation — upper centre of each panel
    ax.text(0.62, 0.86, "\n".join(n_labels),
            transform=ax.transAxes, fontsize=6.5, ha="left", va="top",
            color="0.4", family="monospace")

    # Reference lines
    ax.axvline(0.1, color="0.55", lw=0.75, ls="--", alpha=0.65)
    ax.axvline(0.9, color="0.55", lw=0.75, ls="--", alpha=0.65)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    ax.grid(axis="x", alpha=0.1, lw=0.5)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=8.5)

    # Pole annotations (based on combined SBM+DC-SBM)
    all_ci = np.array([ci_2p(r) for gtype_r in runs_by_gtype.values()
                       for r in gtype_r])
    all_ci = all_ci[np.isfinite(all_ci)]
    mono = np.mean(all_ci < 0.1)
    coex = np.mean(all_ci > 0.9)
    ymax = ax.get_ylim()[1]

    return mono, coex


# ── Figure 1: 5 rows × 2 cols (k × scheduler) ────────────────────────────────

fig1, axes1 = plt.subplots(
    len(K_VALUES), 2,
    figsize=(11, 3.2 * len(K_VALUES)),
    sharey=False, sharex=True,
    constrained_layout=True,
)

for col_i, (sched_key, sched_label) in enumerate(SCHEDULERS):
    axes1[0, col_i].set_title(sched_label, fontsize=11, fontweight="bold")

    for row_i, k in enumerate(K_VALUES):
        ax = axes1[row_i, col_i]

        runs_by_gtype = {
            gtype: [r for r in valid
                    if r["_scheduler"] == sched_key
                    and r["_graph_type"] == gtype
                    and r["_target_k"] == k]
            for gtype in ("sbm", "dc_sbm")
        }

        mono, coex = fill_panel(ax, runs_by_gtype, k, sched_key)

        # k label on left side of left column only
        if col_i == 0:
            ax.set_ylabel(f"k = {k}\nDensity (KDE)", fontsize=9.5)
        else:
            ax.set_ylabel("Density (KDE)", fontsize=8.5)

        if row_i == len(K_VALUES) - 1:
            ax.set_xlabel("Coexistence index  (0 = monopoly → 1 = equal)", fontsize=9)

        # Legend: k=10 left panel (first row where both SBM and DC-SBM exist)
        if row_i == 1 and col_i == 0:
            ax.legend(fontsize=8.5, loc="upper center", framealpha=0.85)

        # Stash stats for post-pass annotation
        ax._pole_stats = (mono, coex)

# Annotate pole fractions after all ylims are set
for row_i in range(len(K_VALUES)):
    for col_i in range(2):
        ax = axes1[row_i, col_i]
        if ax._pole_stats[0] is None:
            continue
        mono, coex = ax._pole_stats
        ylim = ax.get_ylim()[1]
        ax.text(0.03, ylim * 0.96, f"{mono:.0%}\nmonopoly",
                fontsize=7.5, ha="left", va="top", color="#1a1a1a",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75))
        ax.text(0.97, ylim * 0.96, f"{coex:.0%}\ncoexist.",
                fontsize=7.5, ha="right", va="top", color="#1a1a1a",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75))

out1_pdf = FIGURES_DIR / "fig_bimodality_by_k_5x2.pdf"
out1_png = FIGURES_DIR / "fig_bimodality_by_k_5x2.png"
fig1.savefig(out1_pdf, dpi=200)
fig1.savefig(out1_png, dpi=150)
print(f"Saved {out1_pdf}")
print(f"Saved {out1_png}")


# ── Figure 2: 2×3 compact grid, sync only ─────────────────────────────────────
# Panels: k=8,10,12 (top row)  k=14,19,pooled (bottom row)

POOLED_K = [8, 10, 12, 14, 19]  # all k for "pooled" panel

fig2, axes2 = plt.subplots(
    2, 3,
    figsize=(13, 5.8),
    sharey=False, sharex=True,
    constrained_layout=True,
)

PANEL_SPECS = [
    (8,  "k = 8",      False),
    (10, "k = 10",     False),
    (12, "k = 12",     False),
    (14, "k = 14",     False),
    (19, "k = 19",     False),
    (None, "All k pooled", True),
]

for idx, (k_val, panel_title, is_pooled) in enumerate(PANEL_SPECS):
    row, col = divmod(idx, 3)
    ax = axes2[row, col]

    sched_key = "sync"
    if is_pooled:
        runs_by_gtype = {
            gtype: [r for r in valid
                    if r["_scheduler"] == sched_key
                    and r["_graph_type"] == gtype]
            for gtype in ("sbm", "dc_sbm")
        }
    else:
        runs_by_gtype = {
            gtype: [r for r in valid
                    if r["_scheduler"] == sched_key
                    and r["_graph_type"] == gtype
                    and r["_target_k"] == k_val]
            for gtype in ("sbm", "dc_sbm")
        }

    mono, coex = fill_panel(ax, runs_by_gtype, k_val, sched_key)

    ax.set_title(panel_title, fontsize=11, fontweight="bold")

    if col == 0:
        ax.set_ylabel("Density (KDE)", fontsize=9)
    if row == 1:
        ax.set_xlabel("Coexistence index", fontsize=9)

    if idx == 0:
        ax.legend(fontsize=8, loc="upper center", framealpha=0.85)
    if k_val == 8:
        ax.text(0.98, 0.97, "DC-SBM: no k=8",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=7, color="0.5", style="italic")

    ax._pole_stats = (mono, coex)

for idx in range(len(PANEL_SPECS)):
    row, col = divmod(idx, 3)
    ax = axes2[row, col]
    if ax._pole_stats[0] is None:
        continue
    mono, coex = ax._pole_stats
    ylim = ax.get_ylim()[1]
    ax.text(0.03, ylim * 0.96, f"{mono:.0%}\nmonopoly",
            fontsize=7.5, ha="left", va="top", color="#1a1a1a",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75))
    ax.text(0.97, ylim * 0.96, f"{coex:.0%}\ncoexist.",
            fontsize=7.5, ha="right", va="top", color="#1a1a1a",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75))

out2_pdf = FIGURES_DIR / "fig_bimodality_by_k_2x3_sync.pdf"
out2_png = FIGURES_DIR / "fig_bimodality_by_k_2x3_sync.png"
fig2.savefig(out2_pdf, dpi=200)
fig2.savefig(out2_png, dpi=150)
print(f"Saved {out2_pdf}")
print(f"Saved {out2_png}")

print("Done.")
