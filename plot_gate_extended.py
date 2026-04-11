"""
plot_gate_extended.py — Gate on/off diagnostic, both board sizes (n=225 and n=625).

Extends plot_gate_diagnostic.py (Figure 6) to include n=225 alongside n=625,
showing that the gate effect on segregation rate is strongest at large board
sizes and weakens at n=225.

Layout: 2 rows × 3 cols
  Rows: Gillespie-like (top) | Synchronous (bottom)
  Cols: ER | SBM | DC-SBM

Each panel: 4 stacked bars in two pairs, separated by a gap:
  Left pair  — n=225:  gate=ON  |  gate=OFF
  Right pair — n=625:  gate=ON  |  gate=OFF

Stacks: segregation (ci>0.85, green) / dominance (ci<0.10, red) / intermediate (grey).
Seg% annotated in green where ≥ 3%.

Output: figures/fig_gate_extended.pdf / .png
        figures/final/S3.pdf
"""

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── load ──────────────────────────────────────────────────────────────────────
results_path = Path("results/gate_diag_v1.pkl")
with open(results_path, "rb") as f:
    results = pickle.load(f)

valid = [r for r in results if not r.get("connectivity_failed")]

CI_SEG = 0.85
CI_DOM = 0.10
K_VALUES   = [10, 14]        # pool both k values (as in Figure 6)
GRID_MAP   = {15: "n=225", 25: "n=625"}   # grid_size → label
GRID_ORDER = [15, 25]

GRAPH_ORDER  = ["er", "sbm", "dc_sbm"]
GRAPH_LABELS = {"er": "ER", "sbm": "SBM", "dc_sbm": "DC-SBM"}
SCHED_ORDER  = ["gillespie_occ", "sync"]
SCHED_LABELS = {
    "gillespie_occ": "Gillespie-like (energy-weighted async)",
    "sync":          "Synchronous",
}

C_DOM = "#d62728"
C_SEG = "#2ca02c"
C_MID = "#bdbdbd"

# ── aggregate proportions ─────────────────────────────────────────────────────
# key: (graph_type, scheduler, grid_size, use_gate)
ci_store: dict[tuple, list[float]] = defaultdict(list)

for r in valid:
    if r.get("_target_k") not in K_VALUES:
        continue
    key = (r["_graph_type"], r["_scheduler"],
           r["_grid_size"], r["_use_gate"])
    ci_store[key].append(r["coexistence_index"])


def proportions(ci_list):
    ci = np.array(ci_list)
    n  = len(ci)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    dom = np.sum(ci < CI_DOM) / n
    seg = np.sum(ci > CI_SEG) / n
    mid = 1.0 - dom - seg
    return dom, seg, mid, n


# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 3,
    figsize=(11, 6),
    sharey=True,
    gridspec_kw={"hspace": 0.48, "wspace": 0.10},
)

bar_w   = 0.28          # bar width
pair_gap = 0.20         # extra gap between the two board-size pairs

# x positions: ON and OFF within each board-size pair, with gap between pairs
#   n=225 gate=ON  → 0.0
#   n=225 gate=OFF → bar_w + 0.06
#   [gap]
#   n=625 gate=ON  → 2*(bar_w + 0.06) + pair_gap
#   n=625 gate=OFF → 3*(bar_w + 0.06) + pair_gap
inner   = bar_w + 0.06
x_on_s  = 0.0
x_off_s = inner
x_on_l  = 2 * inner + pair_gap
x_off_l = 3 * inner + pair_gap
x_panel_width = x_off_l + bar_w

for row, sched in enumerate(SCHED_ORDER):
    for col, graph in enumerate(GRAPH_ORDER):
        ax = axes[row, col]

        for gs, x_on, x_off in [(15, x_on_s, x_off_s),
                                  (25, x_on_l, x_off_l)]:
            for x, use_gate, gate_label in [(x_on, True, "ON"), (x_off, False, "OFF")]:
                key = (graph, sched, gs, use_gate)
                dom, seg, mid, n = proportions(ci_store.get(key, []))

                ax.bar(x, mid, bar_w, bottom=0,         color=C_MID, zorder=2)
                ax.bar(x, dom, bar_w, bottom=mid,       color=C_DOM, zorder=2)
                ax.bar(x, seg, bar_w, bottom=mid + dom, color=C_SEG, zorder=2)

                seg_pct = seg * 100
                if seg_pct >= 3:
                    ax.text(x, mid + dom + seg + 0.02,
                            f"{seg_pct:.0f}%",
                            ha="center", va="bottom", fontsize=7,
                            color=C_SEG, fontweight="bold")

            # Board-size bracket label between the two pairs
            x_centre = (x_on + x_off + bar_w) / 2
            ax.text(x_centre, -0.09, GRID_MAP[gs],
                    ha="center", va="top", fontsize=7.5,
                    transform=ax.get_xaxis_transform())

        # Dividing line between the two board-size pairs
        x_mid = (x_off_s + bar_w + x_on_l) / 2
        ax.axvline(x_mid, color="0.75", lw=0.8, ls=":", zorder=1)

        # Gate ON / OFF sub-labels
        for x, lbl in [(x_on_s, "ON"), (x_off_s, "OFF"),
                        (x_on_l, "ON"), (x_off_l, "OFF")]:
            ax.text(x + bar_w / 2, -0.175, lbl,
                    ha="center", va="top", fontsize=6.5, color="0.4",
                    transform=ax.get_xaxis_transform())

        ax.set_xticks([])
        ax.set_xlim(-0.25, x_panel_width + 0.15)
        ax.set_ylim(0, 1.22)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.0%}")
        )
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="black", linewidth=0.6)

        # Column header on top row
        if row == 0:
            ax.set_title(GRAPH_LABELS[graph], fontsize=10,
                         fontweight="bold", pad=6)

        # Row label
        if col == 0:
            ax.set_ylabel("Proportion of runs", fontsize=8.5)
            ax.text(-0.52, 0.50, SCHED_LABELS[sched],
                    transform=ax.transAxes, fontsize=8.5,
                    ha="right", va="center", rotation=90, style="italic")

# ── legend ────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=C_SEG, label="Segregation  (CI > 0.85)"),
    mpatches.Patch(color=C_DOM, label="Dominance  (CI < 0.10)"),
    mpatches.Patch(color=C_MID, label="Intermediate"),
]
fig.legend(
    handles=legend_patches,
    loc="lower center",
    ncol=3,
    fontsize=8.5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.04),
)

# ── save ──────────────────────────────────────────────────────────────────────
out_dir = Path("figures")
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "fig_gate_extended.pdf", bbox_inches="tight")
fig.savefig(out_dir / "fig_gate_extended.png", bbox_inches="tight", dpi=150)
print("Saved figures/fig_gate_extended.pdf and .png")

final_dir = Path("figures/final")
final_dir.mkdir(exist_ok=True)
fig.savefig(final_dir / "S3.pdf", bbox_inches="tight")
print("Saved figures/final/S3.pdf")

plt.close(fig)
