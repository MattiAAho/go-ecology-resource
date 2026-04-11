"""
generate_final_figures.py — Regenerate all six main figures without suptitles.

Figures saved to figures/final/ as fig1.pdf through fig6.pdf.

Figure mapping (updated 2026-04-08):
  fig1 — Monopoly rate heatmap (plot_monopoly_heatmap.py)
  fig2 — Two-player CI distribution, SBM/DC-SBM, pooled (plot_bimodality.py)
            — includes regime shading and dominance/coexistence labels
  fig3 — CI distribution by k, 5×2 (plot_bimodality_by_k.py)
  fig4 — Weakest-species extinction rate, RPS, ER/SBM/DC-SBM
          (redrawn from full_all_v2.pkl — plot_rps_sweep.py uses wrong data file)
  fig5 — Coexistence vs k, three-panel (plot_coexistence_vs_k_combined.py)
            — includes "coexistence collapse" annotation on panel (a) ER line
  fig6 — Gate diagnostic (plot_gate_diagnostic.py)

Usage:
    python generate_final_figures.py
"""

from __future__ import annotations

import pickle
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.figure

# ── monkeypatch: silence all suptitle calls in every script run below ─────────
matplotlib.figure.Figure.suptitle = lambda self, *a, **kw: self

# Pre-cache analysis_helpers so runpy sub-scripts don't re-import go_engine
from analysis_helpers import ci_2p   # noqa: E402

# ── output directory ──────────────────────────────────────────────────────────
FINAL_DIR = Path("figures/final")
FINAL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path("figures")

print("=" * 60)
print("Generating final figures (no suptitles)")
print(f"Output: {FINAL_DIR}/")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run an existing plot script via runpy, copy named output to final/
# ─────────────────────────────────────────────────────────────────────────────

def run_and_copy(script: str, src_filename: str, dest_filename: str) -> None:
    print(f"\n── {dest_filename} from {script} ──")
    runpy.run_path(script, run_name="__main__")
    src = FIGURES_DIR / src_filename
    dst = FINAL_DIR / dest_filename
    shutil.copy(src, dst)
    print(f"  Copied → {dst}")


# ─────────────────────────────────────────────────────────────────────────────
# fig1 — Monopoly rate heatmap
# ─────────────────────────────────────────────────────────────────────────────
run_and_copy("plot_monopoly_heatmap.py", "fig_monopoly_heatmap.pdf", "fig1.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# fig2 — Two-player CI distribution, SBM/DC-SBM (pooled, with annotations)
# ─────────────────────────────────────────────────────────────────────────────
run_and_copy("plot_bimodality.py", "fig_bimodality_2x2.pdf", "fig2.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# fig3 — CI distribution by k, 5×2 layout
# ─────────────────────────────────────────────────────────────────────────────
run_and_copy("plot_bimodality_by_k.py", "fig_bimodality_by_k_5x2.pdf", "fig3.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# fig4 — Weakest-species extinction rate (RPS, Gillespie, ER/SBM/DC-SBM)
#         Redrawn from full_all_v2.pkl — plot_rps_sweep.py uses full_final.pkl
#         which lacks DC-SBM.
# ─────────────────────────────────────────────────────────────────────────────
print("\n── fig4.pdf (bespoke, full_all_v2.pkl) ──")

with open("results/full_all_v2.pkl", "rb") as fh:
    rps_raw = pickle.load(fh)

rps = [
    r for r in rps_raw
    if r.get("_contest") == "rps"
    and r.get("_scheduler") == "gillespie_occ"
    and not r.get("connectivity_failed")
    and r.get("occ_avg_last30p")
]
print(f"  {len(rps)} valid RPS Gillespie-like runs")

GRAPH_TYPES_4  = ["er", "sbm", "dc_sbm"]
GRAPH_LABEL_4  = {"er": "ER", "sbm": "SBM", "dc_sbm": "DC-SBM"}
K_VALUES_4     = sorted({r["_target_k"] for r in rps
                         if r["_graph_type"] in GRAPH_TYPES_4})
GRID_SIZES_4   = sorted({r["_grid_size"] for r in rps})   # grid_size, not _n_nodes
N_LABELS_4     = {gs: f"n={gs**2}" for gs in GRID_SIZES_4}

cmap_ext = mcolors.LinearSegmentedColormap.from_list(
    "ext_rate", ["#1a9641", "#fee090", "#d73027"], N=256
)
norm_ext = mcolors.Normalize(vmin=0.0, vmax=1.0)

fig4, axes4 = plt.subplots(
    1, len(GRAPH_TYPES_4),
    figsize=(4.8 * len(GRAPH_TYPES_4), 4.4),
    constrained_layout=True,
)

last_im4 = None
for ax, gtype in zip(axes4, GRAPH_TYPES_4):
    mat_ext = np.full((len(GRID_SIZES_4), len(K_VALUES_4)), np.nan)
    mat_occ = np.full_like(mat_ext, np.nan)

    for gi, gs in enumerate(GRID_SIZES_4):
        for ki, k in enumerate(K_VALUES_4):
            subset = [
                r for r in rps
                if r["_graph_type"] == gtype
                and r["_grid_size"]  == gs
                and r["_target_k"]   == k
            ]
            if subset:
                weak = [min(r["occ_avg_last30p"].values()) for r in subset]
                mat_occ[gi, ki] = float(np.mean(weak))
                mat_ext[gi, ki] = sum(1 for v in weak if v < 0.005) / len(weak)

    last_im4 = ax.imshow(
        mat_ext, aspect="auto", origin="lower",
        cmap=cmap_ext, norm=norm_ext,
        extent=[-0.5, len(K_VALUES_4) - 0.5, -0.5, len(GRID_SIZES_4) - 0.5],
    )

    for gi in range(len(GRID_SIZES_4)):
        for ki in range(len(K_VALUES_4)):
            ext = mat_ext[gi, ki]
            occ = mat_occ[gi, ki]
            if np.isnan(ext):
                ax.text(ki, gi, "—", ha="center", va="center",
                        fontsize=9, color="0.5")
                continue
            fc = "white" if ext < 0.15 or ext > 0.75 else "black"
            ax.text(ki, gi + 0.18, f"{ext:.0%}",
                    ha="center", va="center", fontsize=8, color=fc)
            ax.text(ki, gi - 0.22, f"occ={occ:.3f}",
                    ha="center", va="center", fontsize=6, color=fc, alpha=0.85)

    ax.set_xticks(range(len(K_VALUES_4)))
    ax.set_xticklabels(K_VALUES_4, fontsize=9)
    ax.set_yticks(range(len(GRID_SIZES_4)))
    ax.set_yticklabels([N_LABELS_4[gs] for gs in GRID_SIZES_4], fontsize=9)
    ax.set_xlabel("Mean degree k", fontsize=10)
    ax.set_title(GRAPH_LABEL_4[gtype], fontsize=11, fontweight="bold")

axes4[0].set_ylabel("Board size", fontsize=10)

cb4 = fig4.colorbar(last_im4, ax=axes4.tolist(),
                    fraction=0.02, pad=0.03, orientation="horizontal")
cb4.set_label(
    "Extinction rate  (fraction of runs with weakest-species occupancy < 0.5%)",
    fontsize=10,
)
cb4.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb4.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

out4 = FINAL_DIR / "fig4.pdf"
fig4.savefig(out4, dpi=200, bbox_inches="tight")
fig4.savefig(out4.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"  Saved {out4}")
plt.close(fig4)


# ─────────────────────────────────────────────────────────────────────────────
# fig5 — Coexistence vs k, three-panel
# ─────────────────────────────────────────────────────────────────────────────
run_and_copy("plot_coexistence_vs_k_combined.py",
             "fig_coexistence_vs_k_combined.pdf", "fig5.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# fig6 — Gate diagnostic
# ─────────────────────────────────────────────────────────────────────────────
run_and_copy("plot_gate_diagnostic.py", "fig_gate_diagnostic.pdf", "fig6.pdf")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Done. Final figures in figures/final/:")
for p in sorted(FINAL_DIR.glob("fig*.pdf")):
    print(f"  {p.name}")
print("=" * 60)
