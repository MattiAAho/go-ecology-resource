"""
plot_gate_3p_diagnostic.py — Gate on/off regime proportions, three-player games.

Mirrors plot_gate_diagnostic.py (two-player) as closely as possible.

Layout: 4 rows × 3 cols
  Rows: Gillespie-like RPS | Gillespie-like AVA | Sync RPS | Sync AVA
  Cols: ER | SBM | DC-SBM

Each panel: stacked bars gate=ON (left) vs gate=OFF (right), k=10 & 14 pooled, n=625.
Stacks:
  coexist (Jain ≥ 0.85 AND survivors=3)  — green
  extinct (survivors < 3)                 — red
  mid     (survivors=3 but Jain < 0.85)   — grey

Output: figures/fig_gate_3p_diagnostic.pdf/.png
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
results_path = Path("results/gate_3p_diag_v1.pkl")
with open(results_path, "rb") as f:
    results = pickle.load(f)

valid = [r for r in results if not r.get("connectivity_failed")]

JAIN_COEXIST = 0.85
TARGET_N     = 25          # grid_size → n=625
K_VALUES     = [10, 14]    # pool both

GRAPH_ORDER  = ["er", "sbm", "dc_sbm"]
GRAPH_LABELS = {"er": "ER", "sbm": "SBM", "dc_sbm": "DC-SBM"}

# rows: (scheduler, contest)
ROW_ORDER = [
    ("gillespie_occ", "rps"),
    ("gillespie_occ", "all_vs_all"),
    ("sync",          "rps"),
    ("sync",          "all_vs_all"),
]
ROW_LABELS = {
    ("gillespie_occ", "rps"):       "Gillespie-like\nRPS",
    ("gillespie_occ", "all_vs_all"): "Gillespie-like\nAVA",
    ("sync",          "rps"):       "Sync\nRPS",
    ("sync",          "all_vs_all"): "Sync\nAVA",
}

# colours: coexist=teal, extinct=red, mid=light grey  (mirrors seg/dom/mid in 2P figure)
C_COEXIST = "#2ca02c"
C_EXTINCT = "#d62728"
C_MID     = "#bdbdbd"

# ── aggregate ───────────────────────────────────────────────────────────────
# key: (contest, graph_type, scheduler, use_gate)
store: dict[tuple, list[dict]] = defaultdict(list)

for r in valid:
    if r["_grid_size"] != TARGET_N:
        continue
    if r.get("_target_k") not in K_VALUES:
        continue
    key = (r["_contest"], r["_graph_type"], r["_scheduler"], r["_use_gate"])
    store[key].append(r)


def proportions(rs: list[dict]):
    n = len(rs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    surv = np.array([r["survivors_count"]     for r in rs])
    jain = np.array([r["jain_fairness_final"] for r in rs])
    coexist = np.sum((surv == 3) & (jain >= JAIN_COEXIST)) / n
    extinct = np.sum(surv < 3) / n
    mid     = 1.0 - coexist - extinct
    return coexist, extinct, mid, n


def mean_bite(rs: list[dict]) -> float:
    vals = [r.get("gate_bite_rate", float("nan")) for r in rs]
    return float(np.nanmean(vals)) if vals else float("nan")


# ── figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    4, 3,
    figsize=(9, 10),
    sharey=True,
    gridspec_kw={"hspace": 0.55, "wspace": 0.12},
)

bar_w  = 0.32
x_on   = 0.0
x_off  = 0.55
xticks = [x_on, x_off]

for row, (sched, contest) in enumerate(ROW_ORDER):
    for col, graph in enumerate(GRAPH_ORDER):
        ax = axes[row, col]

        on_key  = (contest, graph, sched, True)
        off_key = (contest, graph, sched, False)

        coex_on,  ext_on,  mid_on,  n_on  = proportions(store.get(on_key,  []))
        coex_off, ext_off, mid_off, n_off = proportions(store.get(off_key, []))
        bite = mean_bite(store.get(on_key, []))

        for x, coex, ext, mid, n, gate_label in [
            (x_on,  coex_on,  ext_on,  mid_on,  n_on,  "ON"),
            (x_off, coex_off, ext_off, mid_off, n_off, "OFF"),
        ]:
            # stacked: bottom=0 → mid → extinct → coexist (top)
            ax.bar(x, mid,  bar_w, bottom=0,            color=C_MID,     zorder=2)
            ax.bar(x, ext,  bar_w, bottom=mid,          color=C_EXTINCT,  zorder=2)
            ax.bar(x, coex, bar_w, bottom=mid + ext,    color=C_COEXIST,  zorder=2)

            # annotate coexist % if non-trivial
            if coex * 100 >= 3:
                ax.text(x, mid + ext + coex + 0.02, f"{coex*100:.0f}%",
                        ha="center", va="bottom", fontsize=7.5,
                        color=C_COEXIST, fontweight="bold")

        # annotate bite rate inside gate=ON bar (bottom area)
        if not np.isnan(bite):
            ax.text(x_on, 0.03, f"bite\n{bite:.0%}",
                    ha="center", va="bottom", fontsize=6.5, color="#555",
                    style="italic")

        ax.set_xticks(xticks)
        ax.set_xticklabels(["gate\nON", "gate\nOFF"], fontsize=8)
        ax.set_xlim(-0.35, 0.90)
        ax.set_ylim(0, 1.22)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="black", linewidth=0.6)

        # column header on top row only
        if row == 0:
            ax.set_title(GRAPH_LABELS[graph], fontsize=10, fontweight="bold", pad=6)

        # row label on leftmost column
        if col == 0:
            ax.set_ylabel("Proportion of runs", fontsize=8.5)
            ax.text(-0.55, 0.50, ROW_LABELS[(sched, contest)],
                    transform=ax.transAxes, fontsize=8.5,
                    ha="right", va="center", rotation=90, style="italic")

# ── legend ──────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=C_COEXIST, label=f"Genuine coexistence  (survivors = 3, Jain ≥ {JAIN_COEXIST})"),
    mpatches.Patch(color=C_EXTINCT, label="Extinction  (survivors < 3)"),
    mpatches.Patch(color=C_MID,     label="Intermediate  (survivors = 3, Jain < 0.85)"),
]
fig.legend(
    handles=legend_patches,
    loc="lower center",
    ncol=1,
    fontsize=8.5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.04),
)

# ── save ────────────────────────────────────────────────────────────────────
out = Path("figures")
out.mkdir(exist_ok=True)
fig.savefig(out / "fig_gate_3p_diagnostic.pdf", bbox_inches="tight")
fig.savefig(out / "fig_gate_3p_diagnostic.png", bbox_inches="tight", dpi=150)
print("Saved figures/fig_gate_3p_diagnostic.pdf and .png")
plt.close(fig)
