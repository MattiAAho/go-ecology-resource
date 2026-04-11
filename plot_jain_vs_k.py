"""
plot_jain_vs_k.py — Mean Jain fairness index vs target_k, three-player contests.

Figure 1 (fig_jain_vs_k_2panel.pdf):
  2 panels side by side — RPS (left), AVA (right)
  One line per graph type, all grid sizes and policies pooled.
  y-axis = Jain fairness index (J=1 perfect coexistence, J=1/3 monopoly for 3 species)

Figure 2 (fig_jain_vs_k_3panel.pdf):
  Same but with a third panel showing two-player Gillespie ci_2p for direct comparison.

Usage:  python plot_jain_vs_k.py
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
print("Loading three-player data …")
with open("results/full_all_v2.pkl", "rb") as fh:
    rps_raw = pickle.load(fh)
with open("results/ava_all_v1.pkl", "rb") as fh:
    ava_raw = pickle.load(fh)
print("Loading two-player data …")
with open("results/two_p_all_v1.pkl", "rb") as fh:
    twop_raw = pickle.load(fh)

def valid_3p(data):
    return [r for r in data
            if not r.get("connectivity_failed")
            and r.get("jain_fairness_final") is not None]

rps = valid_3p(rps_raw)
ava = valid_3p(ava_raw)
twop = [r for r in twop_raw
        if not r.get("connectivity_failed")
        and r.get("black_occupancy") is not None
        and r["_scheduler"] == "gillespie_occ"]

print(f"  RPS: {len(rps)}, AVA: {len(ava)}, 2P Gillespie: {len(twop)}")

# ── style ─────────────────────────────────────────────────────────────────────
GRAPH_STYLE = {
    "er":      dict(color="#e41a1c", marker="o", ls="-",  label="ER"),
    "sbm":     dict(color="#377eb8", marker="s", ls="-",  label="SBM"),
    "dc_sbm":  dict(color="#ff7f00", marker="^", ls="-",  label="DC-SBM"),
    "lattice": dict(color="#4daf4a", marker="D", ls="--", label="Lattice"),
    "torus":   dict(color="#984ea3", marker="v", ls="--", label="Torus"),
}
GRAPH_ORDER = ["lattice", "torus", "er", "sbm", "dc_sbm"]

def compute_means(data, metric_fn):
    """Return {(graph_type, k): (mean, 95ci_half, n)}."""
    groups = defaultdict(list)
    for r in data:
        v = metric_fn(r)
        if v is not None and np.isfinite(v):
            groups[(r["_graph_type"], r["_target_k"])].append(v)
    out = {}
    for key, vals in groups.items():
        a = np.array(vals)
        out[key] = (a.mean(), 1.96 * a.std() / np.sqrt(len(a)), len(a))
    return out

means_rps  = compute_means(rps,  lambda r: r["jain_fairness_final"])
means_ava  = compute_means(ava,  lambda r: r["jain_fairness_final"])
means_twop = compute_means(twop, ci_2p)

def draw_panel(ax, means, graph_types, title, ylabel=None,
               jain_floor=None, legend=False):
    for gtype in graph_types:
        st = GRAPH_STYLE[gtype]
        pts = sorted((k, v) for (gt, k), v in means.items() if gt == gtype)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1][0] for p in pts]
        es = [p[1][1] for p in pts]
        ax.errorbar(xs, ys, yerr=es,
                    color=st["color"], marker=st["marker"], ls=st["ls"],
                    markersize=6, linewidth=1.8, capsize=4,
                    label=st["label"])

    if jain_floor is not None:
        ax.axhline(jain_floor, color="0.55", lw=0.8, ls=":", alpha=0.7,
                   label=f"monopoly floor (1/{round(1/jain_floor)})")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean degree k", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_xticks([4, 8, 10, 12, 14, 19])
    ax.grid(alpha=0.25)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if legend:
        ax.legend(fontsize=9, loc="upper right", framealpha=0.85)

# ── Figure 1: 2-panel RPS / AVA ───────────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True,
                            constrained_layout=True)
fig1.suptitle(
    "Mean Jain fairness index vs graph density — three-player contest\n"
    "Gillespie-like scheduler, all grid sizes and policies pooled",
    fontsize=12, fontweight="bold",
)

draw_panel(axes1[0], means_rps, GRAPH_ORDER,
           title="RPS (rock-paper-scissors)",
           ylabel="Mean Jain fairness index  (J=1/3 monopoly → J=1 equal)",
           jain_floor=1/3, legend=True)
draw_panel(axes1[1], means_ava, GRAPH_ORDER,
           title="AVA (all-vs-all)",
           jain_floor=1/3)

out1 = FIGURES_DIR / "fig_jain_vs_k_2panel.pdf"
fig1.savefig(out1, dpi=200)
fig1.savefig(out1.with_suffix(".png"), dpi=150)
print(f"Saved {out1}")

# ── Figure 2: 3-panel RPS / AVA / 2P-Gillespie ───────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True,
                            constrained_layout=True)
fig2.suptitle(
    "Coexistence vs graph density across competition modes\n"
    "Gillespie-like scheduler, all grid sizes and policies pooled",
    fontsize=12, fontweight="bold",
)

draw_panel(axes2[0], means_rps, GRAPH_ORDER,
           title="3-player RPS",
           ylabel="Mean coexistence index  (0 = monopoly, 1 = equal)",
           jain_floor=1/3, legend=True)
draw_panel(axes2[1], means_ava, GRAPH_ORDER,
           title="3-player AVA",
           jain_floor=1/3)
draw_panel(axes2[2], means_twop, GRAPH_ORDER,
           title="2-player (Gillespie)",
           jain_floor=None)   # no natural floor for 2-player

# add ci=0 reference line on 2P panel
axes2[2].axhline(0.0, color="0.55", lw=0.8, ls=":", alpha=0.5)

out2 = FIGURES_DIR / "fig_jain_vs_k_3panel.pdf"
fig2.savefig(out2, dpi=200)
fig2.savefig(out2.with_suffix(".png"), dpi=150)
print(f"Saved {out2}")
print("Done.")
