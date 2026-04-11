"""
plot_coexistence_vs_k_combined.py — Three-panel coexistence vs k figure.

Panel 1: Two-player Gillespie  (ci_2p, stones-normalised)
Panel 2: Three-player RPS      (Jain fairness index)
Panel 3: Three-player AVA      (Jain fairness index)

y-axis: panels share 0-1 scale but carry distinct labels reflecting the
different metrics.  Caption notes this explicitly.

Usage:  python plot_coexistence_vs_k_combined.py
Output: figures/fig_coexistence_vs_k_combined.pdf / .png
"""
from __future__ import annotations
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_helpers import ci_2p

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────
print("Loading data …")
with open("results/full_all_v2.pkl", "rb") as fh:
    rps_raw = pickle.load(fh)
with open("results/ava_all_v1.pkl", "rb") as fh:
    ava_raw = pickle.load(fh)
with open("results/two_p_all_v1.pkl", "rb") as fh:
    twop_raw = pickle.load(fh)

rps  = [r for r in rps_raw  if not r.get("connectivity_failed")
        and r.get("jain_fairness_final") is not None]
ava  = [r for r in ava_raw  if not r.get("connectivity_failed")
        and r.get("jain_fairness_final") is not None]
twop = [r for r in twop_raw if not r.get("connectivity_failed")
        and r.get("black_occupancy") is not None
        and r["_scheduler"] == "gillespie_occ"]
print(f"  2P Gillespie: {len(twop)}, RPS: {len(rps)}, AVA: {len(ava)}")

# ── compute group means ───────────────────────────────────────────────────────
def group_means(data, metric_fn):
    groups = defaultdict(list)
    for r in data:
        v = metric_fn(r)
        if v is not None and np.isfinite(v):
            groups[(r["_graph_type"], r["_target_k"])].append(v)
    return {k: (np.mean(v), 1.96 * np.std(v) / np.sqrt(len(v))) for k, v in groups.items()}

means_2p  = group_means(twop, ci_2p)
means_rps = group_means(rps,  lambda r: r["jain_fairness_final"])
means_ava = group_means(ava,  lambda r: r["jain_fairness_final"])

# ── style ─────────────────────────────────────────────────────────────────────
GRAPH_STYLE = {
    "er":      dict(color="#e41a1c", marker="o", ls="-",  label="ER"),
    "sbm":     dict(color="#377eb8", marker="s", ls="-",  label="SBM"),
    "dc_sbm":  dict(color="#ff7f00", marker="^", ls="-",  label="DC-SBM"),
    "lattice": dict(color="#4daf4a", marker="D", ls="--", label="Lattice"),
    "torus":   dict(color="#984ea3", marker="v", ls="--", label="Torus"),
}
GRAPH_ORDER = ["lattice", "torus", "er", "sbm", "dc_sbm"]

# ── draw ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8),
                         sharey=False, constrained_layout=True)


PANEL_SPECS = [
    (axes[0], means_2p,  "Two-player (Gillespie-like)",
     "Mean coexistence index\n(stones-normalised, 0 = monopoly)", None),
    (axes[1], means_rps, "Three-player RPS",
     "Mean Jain fairness index\n(J = 1/3 monopoly floor, J = 1 equal)", 1/3),
    (axes[2], means_ava, "Three-player AVA",
     "Mean Jain fairness index\n(J = 1/3 monopoly floor, J = 1 equal)", 1/3),
]

for panel_i, (ax, means, title, ylabel, jain_floor) in enumerate(PANEL_SPECS):
    for gtype in GRAPH_ORDER:
        st  = GRAPH_STYLE[gtype]
        pts = sorted((k, m, e) for (gt, k), (m, e) in means.items() if gt == gtype)
        if not pts:
            continue
        xs, ys, es = zip(*pts)
        ax.errorbar(xs, ys, yerr=es,
                    color=st["color"], marker=st["marker"], ls=st["ls"],
                    markersize=6, linewidth=1.8, capsize=4,
                    label=st["label"])

    if jain_floor is not None:
        ax.axhline(jain_floor, color="0.55", lw=0.9, ls=":", alpha=0.7,
                   label=f"monopoly floor (J = 1/3)")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean degree k", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([4, 8, 10, 12, 14, 19])
    ax.grid(alpha=0.25)

    if panel_i == 0:
        ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
        # Collapse annotation: arrow along the ER line from k=10→k=14
        ax.annotate(
            "coexistence\ncollapse",
            xy=(14, 0.18),          # arrowhead — near actual ER value at k=14
            xytext=(12, 0.09),      # text centre at x=12, y≈0.10
            fontsize=8.5,
            color="#e41a1c",        # ER colour
            ha="center",
            fontstyle="italic",
            arrowprops=dict(
                arrowstyle="->",
                color="#e41a1c",
                lw=1.5,
                shrinkA=4,
                shrinkB=4,
            ),
        )

    # panel label (a), (b), (c)
    ax.text(-0.08, 1.02, f"({'abc'[panel_i]})",
            transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")

out_pdf = FIGURES_DIR / "fig_coexistence_vs_k_combined.pdf"
out_png = FIGURES_DIR / "fig_coexistence_vs_k_combined.png"
fig.savefig(out_pdf, dpi=200)
fig.savefig(out_png, dpi=150)
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")
