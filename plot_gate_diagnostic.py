"""
plot_gate_diagnostic.py — Gate on/off regime proportions figure.

Two-panel layout:
  Row 1: Gillespie-like (energy-weighted async), n=625
  Row 2: Sync, n=625

  Cols: ER | SBM | DC-SBM

Each panel: stacked bars gate=ON (left) vs gate=OFF (right),
            k=10 and k=14 pooled.
Stacks: dom (ci<0.10) | seg (ci>0.85) | mid (rest).

Output: figures/fig_gate_diagnostic.pdf/.png
"""

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── load data ──────────────────────────────────────────────────────────────
results_path = Path("results/gate_diag_v1.pkl")
with open(results_path, "rb") as f:
    results = pickle.load(f)

valid = [r for r in results if not r.get("connectivity_failed")]

CI_SEG = 0.85
CI_DOM = 0.10
TARGET_N = 25          # grid_size (n=625)
K_VALUES  = [10, 14]   # pool both

GRAPH_ORDER  = ["er", "sbm", "dc_sbm"]
GRAPH_LABELS = {"er": "ER", "sbm": "SBM", "dc_sbm": "DC-SBM"}
SCHED_ORDER  = ["gillespie_occ", "sync"]
SCHED_LABELS = {"gillespie_occ": "Gillespie-like (energy-weighted async)",
                "sync":          "Synchronous"}

# colours: dom=warm red, seg=teal, mid=light grey
C_DOM = "#d62728"
C_SEG = "#2ca02c"
C_MID = "#bdbdbd"

# ── aggregate proportions ───────────────────────────────────────────────────
# key: (graph_type, scheduler, use_gate)  value: list of ci values
ci_store: dict[tuple, list[float]] = defaultdict(list)

for r in valid:
    if r["_grid_size"] != TARGET_N:
        continue
    if r.get("_target_k") not in K_VALUES:
        continue
    key = (r["_graph_type"], r["_scheduler"], r["_use_gate"])
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


# ── figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 3,
    figsize=(9, 5.5),
    sharey=True,
    gridspec_kw={"hspace": 0.45, "wspace": 0.12},
)

bar_w  = 0.32
x_on   = 0.0
x_off  = 0.55
xticks = [x_on, x_off]

for row, sched in enumerate(SCHED_ORDER):
    for col, graph in enumerate(GRAPH_ORDER):
        ax = axes[row, col]

        on_key  = (graph, sched, True)
        off_key = (graph, sched, False)

        dom_on,  seg_on,  mid_on,  n_on  = proportions(ci_store.get(on_key,  []))
        dom_off, seg_off, mid_off, n_off = proportions(ci_store.get(off_key, []))

        for x, dom, seg, mid, n, gate_label in [
            (x_on,  dom_on,  seg_on,  mid_on,  n_on,  "ON"),
            (x_off, dom_off, seg_off, mid_off, n_off, "OFF"),
        ]:
            # stacked bar: bottom=0 → mid → dom → seg (top)
            ax.bar(x, mid, bar_w, bottom=0,           color=C_MID, zorder=2)
            ax.bar(x, dom, bar_w, bottom=mid,         color=C_DOM, zorder=2)
            ax.bar(x, seg, bar_w, bottom=mid + dom,   color=C_SEG, zorder=2)

            # annotate seg % if non-trivial
            seg_pct = seg * 100
            if seg_pct >= 3:
                ax.text(x, mid + dom + seg + 0.02, f"{seg_pct:.0f}%",
                        ha="center", va="bottom", fontsize=7.5,
                        color=C_SEG, fontweight="bold")

        ax.set_xticks(xticks)
        ax.set_xticklabels(["gate\nON", "gate\nOFF"], fontsize=8)
        ax.set_xlim(-0.35, 0.90)
        ax.set_ylim(0, 1.18)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="black", linewidth=0.6)

        # column header (graph type) on top row only
        if row == 0:
            ax.set_title(GRAPH_LABELS[graph], fontsize=10, fontweight="bold", pad=6)

        # row label
        if col == 0:
            ax.set_ylabel("Proportion of runs", fontsize=8.5)
            ax.text(-0.55, 0.50, SCHED_LABELS[sched],
                    transform=ax.transAxes, fontsize=8.5,
                    ha="right", va="center", rotation=90, style="italic")

# ── legend ──────────────────────────────────────────────────────────────────
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
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle(
    "Gate on/off: regime proportions  (n = 625, k = 10 & 14 pooled)",
    fontsize=11, fontweight="bold", y=1.01,
)

# ── save ────────────────────────────────────────────────────────────────────
out = Path("figures")
out.mkdir(exist_ok=True)
fig.savefig(out / "fig_gate_diagnostic.pdf", bbox_inches="tight")
fig.savefig(out / "fig_gate_diagnostic.png", bbox_inches="tight", dpi=150)
print("Saved figures/fig_gate_diagnostic.pdf and .png")
plt.close(fig)
