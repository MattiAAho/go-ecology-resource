"""
plot_bimodality.py — Two-player SBM/DC-SBM bimodality figure.

Produces a 2×2 panel: rows = scheduler (sync, Gillespie), cols = dummy
(both graph types overlaid in each panel).

Layout:
  Top row    — Sync scheduler
  Bottom row — Gillespie scheduler
  Left col   — full CI distribution (density)
  Right col  — (unused — we do 1×2 instead; see LAYOUT below)

Actual layout chosen: 1 row × 2 cols
  Left  panel: Sync        (SBM blue + DC-SBM orange, semi-transparent)
  Right panel: Gillespie   (same colour scheme)

Both panels share the x-axis (CI 0–1) and y-axis scale.

The dramatic emptiness between ci≈0.1 and ci≈0.85 is the visual anchor
for Claim 2.  Overlaying the two graph types in each panel directly
demonstrates that DC-SBM and SBM produce nearly identical distributions
(Claim 3).

Usage:
    python plot_bimodality.py                        # loads results/two_p_all_v1.pkl
    python plot_bimodality.py results/my.pkl

Output:
    figures/fig_bimodality_2x2.pdf
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

# ── paths ─────────────────────────────────────────────────────────────────────

PKL_PATH    = sys.argv[1] if len(sys.argv) > 1 else "results/two_p_all_v1.pkl"
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

print(f"Loading {PKL_PATH} …")
with open(PKL_PATH, "rb") as fh:
    raw: list[dict] = pickle.load(fh)
print(f"  {len(raw)} total records loaded")

# ── filter ────────────────────────────────────────────────────────────────────

valid = [
    r for r in raw
    if not r.get("connectivity_failed")
    and r.get("_graph_type") in ("sbm", "dc_sbm")
    and r.get("black_occupancy") is not None  # needed for ci_2p
]
print(f"  {len(valid)} valid SBM/DC-SBM records")

# ── helpers ───────────────────────────────────────────────────────────────────

SCHED_MAP = {
    "sync":          "Synchronous (round-robin)",
    "gillespie_occ": "Gillespie-like (energy-weighted async)",
}
SCHED_ORDER = ["sync", "gillespie_occ"]

GRAPH_STYLE = {
    "sbm":    dict(color="#1f77b4", label="SBM",    alpha=0.55, zorder=3),
    "dc_sbm": dict(color="#ff7f0e", label="DC-SBM", alpha=0.55, zorder=2),
}

CI_GRID = np.linspace(0.0, 1.0, 500)
BW_METHOD = 0.06   # narrow bandwidth to keep the two poles visible


def kde_density(ci_vals: np.ndarray) -> np.ndarray:
    """Return KDE density evaluated on CI_GRID; zeros if < 2 points."""
    clean = ci_vals[np.isfinite(ci_vals)]
    if len(clean) < 2:
        return np.zeros_like(CI_GRID)
    kde = gaussian_kde(clean, bw_method=BW_METHOD)
    return kde(CI_GRID)


def get_ci(runs: list[dict]) -> np.ndarray:
    return np.array([ci_2p(r) for r in runs], dtype=float)


# ── figure: 1×2 panels (one per scheduler), graph types overlaid ─────────────

fig, axes = plt.subplots(
    1, 2,
    figsize=(11, 4.4),
    sharey=True,
    constrained_layout=True,
)

for ax, sched in zip(axes, SCHED_ORDER):
    sched_runs = [r for r in valid if r["_scheduler"] == sched]
    n_total = len(sched_runs)

    # Regime shading — both panels
    ax.axvspan(0.0, 0.1, alpha=0.10, color="#d73027", zorder=0, lw=0)
    ax.axvspan(0.85, 1.0, alpha=0.10, color="#1a9850", zorder=0, lw=0)   # segregation: CI > 0.85 (m1, MA 2026-10-31)

    for gtype in ("sbm", "dc_sbm"):
        style = GRAPH_STYLE[gtype]
        subset = [r for r in sched_runs if r["_graph_type"] == gtype]
        ci = get_ci(subset)
        if ci.size == 0:
            continue

        density = kde_density(ci)

        ax.fill_between(
            CI_GRID, density,
            color=style["color"],
            alpha=style["alpha"],
            zorder=style["zorder"],
            label=f"{style['label']}  ({len(ci):,} runs)",
        )
        ax.plot(
            CI_GRID, density,
            color=style["color"],
            linewidth=1.4,
            zorder=style["zorder"] + 1,
        )

    # Reference lines at regime boundaries
    ax.axvline(0.1, color="0.5", lw=0.8, ls="--", alpha=0.7)
    ax.axvline(0.85, color="0.5", lw=0.8, ls="--", alpha=0.7)

    # Regime labels — sync panel only, placed above KDE curves
    # x=0.13 / x=0.87 keeps text clear of the y-axis and right edge
    if sched == "sync":
        ax.text(0.13, 0.97, "dominance\nregime",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, fontstyle="italic", color="#a01010", zorder=6)
        ax.text(0.87, 0.97, "segregation\nregime",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, fontstyle="italic", color="#1a6e35", zorder=6)

    ax.set_title(SCHED_MAP[sched], fontsize=11, pad=6)
    ax.set_xlabel("Coexistence index  (0 = one species only → 1 = equal coexistence)", fontsize=9.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)
    ax.legend(fontsize=9, loc="upper center", framealpha=0.8)
    ax.tick_params(labelsize=9)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

axes[0].set_ylabel("Density (KDE)", fontsize=10)

out = FIGURES_DIR / "fig_bimodality_2x2.pdf"
fig.savefig(out, dpi=200)
print(f"Saved {out}")

# ── also save a PNG for quick preview ─────────────────────────────────────────
out_png = FIGURES_DIR / "fig_bimodality_2x2.png"
fig.savefig(out_png, dpi=150)
print(f"Saved {out_png}")

print("Done.")
