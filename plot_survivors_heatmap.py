"""
plot_survivors_heatmap.py — Three-panel survivors and weakest-occ heatmaps.

Panels: ER | SBM | DC-SBM
Rows (y): n_nodes (81, 225, 361, 625) — largest at top
Cols (x): target_k (8, 10, 12, 14, 19)
DC-SBM k=8 cells shown as "—".
All policies pooled, Gillespie scheduler only.
Data: results/full_all_v2.pkl (RPS, three-player)

Outputs:
    figures/fig1_survivors_heatmap.pdf / .png
    figures/fig1b_weakest_occ_heatmap.pdf / .png
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

PKL_PATH    = sys.argv[1] if len(sys.argv) > 1 else "results/full_all_v2.pkl"
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

print(f"Loading {PKL_PATH} …")
with open(PKL_PATH, "rb") as fh:
    raw = pickle.load(fh)

valid = [
    r for r in raw
    if not r.get("connectivity_failed")
    and r["_scheduler"] == "gillespie_occ"
    and r.get("survivors_count") is not None
]
print(f"  {len(valid)} valid Gillespie runs (all policies pooled)")

# ── layout constants ───────────────────────────────────────────────────────────

GRAPH_TYPES  = ["er", "sbm", "dc_sbm"]
GRAPH_LABELS = {"er": "ER (random)", "sbm": "SBM (modular)", "dc_sbm": "DC-SBM (modular)"}
K_VALUES     = [8, 10, 12, 14, 19]
GRID_SIZES   = [9, 15, 19, 25]          # n = 81, 225, 361, 625  (imshow: row 0 = gs=9)
N_LABELS     = [f"n={gs**2}" for gs in GRID_SIZES]

# ── helper: build (n_grid_sizes × n_k_values) matrices ────────────────────────

def build_survivors_matrices(gtype: str):
    """Return mean survivors matrix."""
    mat = np.full((len(GRID_SIZES), len(K_VALUES)), np.nan)
    for gi, gs in enumerate(GRID_SIZES):
        for ki, k in enumerate(K_VALUES):
            subset = [r for r in valid
                      if r["_graph_type"] == gtype
                      and r["_grid_size"]  == gs
                      and r["_target_k"]   == k]
            if subset:
                mat[gi, ki] = np.mean([r["survivors_count"] for r in subset])
    return mat


def build_weakest_matrices(gtype: str):
    """Return (mean weakest occ matrix, extinction rate matrix)."""
    mat_occ = np.full((len(GRID_SIZES), len(K_VALUES)), np.nan)
    mat_ext = np.full((len(GRID_SIZES), len(K_VALUES)), np.nan)
    for gi, gs in enumerate(GRID_SIZES):
        for ki, k in enumerate(K_VALUES):
            subset = [r for r in valid
                      if r["_graph_type"] == gtype
                      and r["_grid_size"]  == gs
                      and r["_target_k"]   == k
                      and r.get("occ_avg_last30p")]
            if subset:
                weak = [min(r["occ_avg_last30p"].values()) for r in subset]
                mat_occ[gi, ki] = np.mean(weak)
                mat_ext[gi, ki] = np.mean([v < 0.005 for v in weak])
    return mat_occ, mat_ext


def is_missing(gtype: str, k: int) -> bool:
    """True for DC-SBM k=8 (no data by design)."""
    return gtype == "dc_sbm" and k == 8


def annotate_cell(ax, ki, gi, text, bg_dark: bool, fontsize: int = 8):
    fc = "white" if bg_dark else "black"
    ax.text(ki, gi, text, ha="center", va="center", fontsize=fontsize,
            color=fc)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Survivors heatmap
# ══════════════════════════════════════════════════════════════════════════════

cmap_surv = mcolors.LinearSegmentedColormap.from_list(
    "surv3", ["#d73027", "#fee090", "#1a9641"], N=256
)
norm_surv = mcolors.Normalize(vmin=1, vmax=3)

fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
fig1.suptitle(
    "Mean survivors count — three-player RPS, Gillespie-like, all policies pooled",
    fontsize=11, fontweight="bold",
)

last_im1 = None
for col, gtype in enumerate(GRAPH_TYPES):
    ax  = axes1[col]
    mat = build_survivors_matrices(gtype)

    last_im1 = ax.imshow(
        mat, aspect="auto", origin="lower",
        cmap=cmap_surv, norm=norm_surv,
        extent=[-0.5, len(K_VALUES) - 0.5, -0.5, len(GRID_SIZES) - 0.5],
    )

    for gi in range(len(GRID_SIZES)):
        for ki, k in enumerate(K_VALUES):
            if is_missing(gtype, k):
                ax.text(ki, gi, "—", ha="center", va="center",
                        fontsize=9, color="0.5")
                continue
            v = mat[gi, ki]
            if np.isnan(v):
                continue
            dark = v < 1.8 or v > 2.7
            annotate_cell(ax, ki, gi, f"{v:.2f}", dark)

    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels(K_VALUES, fontsize=9)
    ax.set_yticks(range(len(GRID_SIZES)))
    ax.set_yticklabels(N_LABELS, fontsize=9)
    ax.set_xlabel("Mean degree k", fontsize=10)
    ax.set_title(GRAPH_LABELS[gtype], fontsize=10, fontweight="bold")
    if col == 0:
        ax.set_ylabel("Grid size", fontsize=10)

cb1 = fig1.colorbar(last_im1, ax=axes1.tolist(), fraction=0.02, pad=0.02)
cb1.set_label("Mean survivors (1 = monopoly → 3 = full coexistence)", fontsize=9)
cb1.set_ticks([1, 1.5, 2, 2.5, 3])

out1_pdf = FIGURES_DIR / "fig1_survivors_heatmap.pdf"
out1_png = FIGURES_DIR / "fig1_survivors_heatmap.png"
fig1.savefig(out1_pdf, dpi=200)
fig1.savefig(out1_png, dpi=150)
print(f"Saved {out1_pdf}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1b — Weakest-species extinction rate heatmap
# ══════════════════════════════════════════════════════════════════════════════

cmap_ext = mcolors.LinearSegmentedColormap.from_list(
    "ext", ["#1a9641", "#fee090", "#d73027"], N=256
)
norm_ext = mcolors.Normalize(vmin=0.0, vmax=1.0)

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
fig2.suptitle(
    "Weakest-species extinction rate — three-player RPS, Gillespie-like, all policies pooled",
    fontsize=11, fontweight="bold",
)

last_im2 = None
for col, gtype in enumerate(GRAPH_TYPES):
    ax = axes2[col]
    mat_occ, mat_ext = build_weakest_matrices(gtype)

    last_im2 = ax.imshow(
        mat_ext, aspect="auto", origin="lower",
        cmap=cmap_ext, norm=norm_ext,
        extent=[-0.5, len(K_VALUES) - 0.5, -0.5, len(GRID_SIZES) - 0.5],
    )

    for gi in range(len(GRID_SIZES)):
        for ki, k in enumerate(K_VALUES):
            if is_missing(gtype, k):
                ax.text(ki, gi, "—", ha="center", va="center",
                        fontsize=9, color="0.5")
                continue
            ext = mat_ext[gi, ki]
            occ = mat_occ[gi, ki]
            if np.isnan(ext):
                continue
            dark = ext < 0.15 or ext > 0.75
            fc   = "white" if dark else "black"
            ax.text(ki, gi, f"{ext:.0%}",
                    ha="center", va="center", fontsize=9, color=fc,
                    fontweight="bold")

    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels(K_VALUES, fontsize=9)
    ax.set_yticks(range(len(GRID_SIZES)))
    ax.set_yticklabels(N_LABELS, fontsize=9)
    ax.set_xlabel("Mean degree k", fontsize=10)
    ax.set_title(GRAPH_LABELS[gtype], fontsize=10, fontweight="bold")
    if col == 0:
        ax.set_ylabel("Grid size", fontsize=10)

cb2 = fig2.colorbar(last_im2, ax=axes2.tolist(), fraction=0.02, pad=0.02)
cb2.set_label("Extinction rate (fraction of runs, weakest species)", fontsize=9)
cb2.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb2.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

out2_pdf = FIGURES_DIR / "fig1b_weakest_occ_heatmap.pdf"
out2_png = FIGURES_DIR / "fig1b_weakest_occ_heatmap.png"
fig2.savefig(out2_pdf, dpi=200)
fig2.savefig(out2_png, dpi=150)
print(f"Saved {out2_pdf}")
print("Done.")
