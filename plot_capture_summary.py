"""
plot_capture_summary.py — Mean capture size and capture event count by graph type.

Two rows × two columns:
  Row 1: mean capture size per run  (captures_total / captures_n)
  Row 2: number of capture events per run  (captures_n)
  Col 1: Synchronous (round-robin)
  Col 2: Gillespie-like (energy-weighted async)

Each panel: one violin per graph type (lattice, torus, ER, SBM, DC-SBM).
Horizontal reference line = overall mean across graph types for that panel.

Labelled clearly as "mean capture size per run" — not a distribution of
individual events.

Output: figures/fig_capture_summary.pdf / .png
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── load ──────────────────────────────────────────────────────────────────────
PKL = Path("results/two_p_all_v1.pkl")
print(f"Loading {PKL} …")
with open(PKL, "rb") as fh:
    raw: list[dict] = pickle.load(fh)

# Drop records missing capture fields or flagged as failed
valid = [
    r for r in raw
    if not r.get("connectivity_failed")
    and "captures_n" in r
    and r.get("captures_n", 0) > 0          # exclude zero-capture runs
]
print(f"  {len(raw)} total → {len(valid)} valid (captures_n > 0)")

# ── constants ─────────────────────────────────────────────────────────────────

GRAPH_ORDER  = ["lattice", "torus", "er", "sbm", "dc_sbm"]
GRAPH_LABELS = {
    "lattice": "Lattice",
    "torus":   "Torus",
    "er":      "ER",
    "sbm":     "SBM",
    "dc_sbm":  "DC-SBM",
}

SCHED_ORDER  = ["sync", "gillespie_occ"]
SCHED_LABELS = {
    "sync":          "Synchronous (round-robin)",
    "gillespie_occ": "Gillespie-like (energy-weighted async)",
}

# Colour per graph type — perceptually distinct, consistent with project palette
# lattice/torus: warm neutrals; random graphs: blues/oranges/green
GRAPH_COLORS = {
    "lattice": "#8c564b",   # brown
    "torus":   "#9467bd",   # purple
    "er":      "#1f77b4",   # blue
    "sbm":     "#2ca02c",   # green
    "dc_sbm":  "#ff7f0e",   # orange
}

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ── build data arrays ─────────────────────────────────────────────────────────

def get_vals(graph_type: str, scheduler: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_cap_sizes, cap_counts) arrays for this slice."""
    runs = [
        r for r in valid
        if r["_graph_type"] == graph_type and r["_scheduler"] == scheduler
    ]
    mean_sizes = np.array([r["captures_total"] / r["captures_n"] for r in runs])
    cap_counts  = np.array([r["captures_n"] for r in runs], dtype=float)
    return mean_sizes, cap_counts


# ── figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(
    2, 2,
    figsize=(10, 7),
    gridspec_kw={"hspace": 0.42, "wspace": 0.28},
)

x_positions = np.arange(len(GRAPH_ORDER))
VIOLIN_WIDTH = 0.6

ROW_METRICS = [
    ("mean_size",  "Mean capture size per run\n(stones captured / capture events)"),
    ("cap_count",  "Capture events per run"),
]

for col, sched in enumerate(SCHED_ORDER):
    for row, (metric, ylabel) in enumerate(ROW_METRICS):
        ax = axes[row, col]

        # Collect per-graph data
        all_vals_for_mean: list[float] = []
        violin_data: list[np.ndarray] = []

        for gt in GRAPH_ORDER:
            mean_sizes, cap_counts = get_vals(gt, sched)
            vals = mean_sizes if metric == "mean_size" else cap_counts
            violin_data.append(vals)
            all_vals_for_mean.extend(vals.tolist())

        # Reference line — overall mean across all graph types
        overall_mean = np.mean(all_vals_for_mean)

        # Draw violins
        parts = ax.violinplot(
            violin_data,
            positions=x_positions,
            widths=VIOLIN_WIDTH,
            showmedians=True,
            showextrema=True,
        )

        # Recolour each violin body
        for i, (body, gt) in enumerate(zip(parts["bodies"], GRAPH_ORDER)):
            c = GRAPH_COLORS[gt]
            body.set_facecolor(c)
            body.set_edgecolor(c)
            body.set_alpha(0.55)

        # Recolour medians / whiskers
        for partname in ("cmedians", "cmins", "cmaxes", "cbars"):
            if partname in parts:
                parts[partname].set_color("black")
                parts[partname].set_linewidth(0.8)

        # Overlay individual data points (jittered) for small-n groups
        for i, (vals, gt) in enumerate(zip(violin_data, GRAPH_ORDER)):
            jitter = np.random.default_rng(42 + i).uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(
                x_positions[i] + jitter, vals,
                s=2, alpha=0.25, color=GRAPH_COLORS[gt], zorder=3, linewidths=0,
            )

        # Reference line
        ax.axhline(overall_mean, color="#555555", linewidth=1.0,
                   linestyle="--", zorder=1,
                   label=f"Overall mean = {overall_mean:.1f}")
        ax.text(
            len(GRAPH_ORDER) - 0.5, overall_mean,
            f" mean={overall_mean:.1f}",
            va="bottom", ha="right", fontsize=7.5, color="#555555",
        )

        # Axes formatting
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [GRAPH_LABELS[gt] for gt in GRAPH_ORDER],
            fontsize=9,
        )
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

        # y-axis floor at 0
        ax.set_ylim(bottom=0)

        # Column header on top row
        if row == 0:
            ax.set_title(SCHED_LABELS[sched], fontsize=10, fontweight="bold", pad=8)


# ── legend — graph type colours ───────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=GRAPH_COLORS[gt], alpha=0.7, label=GRAPH_LABELS[gt])
    for gt in GRAPH_ORDER
]
fig.legend(
    handles=legend_patches,
    loc="lower center",
    ncol=5,
    fontsize=8.5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.03),
)

fig.suptitle(
    "Capture dynamics by graph topology — mean capture size per run\n"
    "(captures_total / captures_n; individual capture-size distribution not recorded)",
    fontsize=10, fontweight="bold", y=1.01,
)

# ── save ──────────────────────────────────────────────────────────────────────
out_pdf = FIGURES_DIR / "fig_capture_summary.pdf"
out_png = FIGURES_DIR / "fig_capture_summary.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"Saved {out_pdf} and {out_png}")
plt.close(fig)
