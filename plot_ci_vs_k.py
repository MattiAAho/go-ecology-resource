"""
plot_ci_vs_k.py — Mean ci_2p vs target_k, one line per graph type, two scheduler panels.

Usage:  python plot_ci_vs_k.py
Output: figures/fig_ci_vs_k.pdf / .png
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis_helpers import ci_2p

PKL_PATH    = sys.argv[1] if len(sys.argv) > 1 else "results/two_p_all_v1.pkl"
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

with open(PKL_PATH, "rb") as fh:
    raw = pickle.load(fh)

valid = [r for r in raw
         if not r.get("connectivity_failed")
         and r.get("black_occupancy") is not None]

# ── style ─────────────────────────────────────────────────────────────────────
GRAPH_STYLE = {
    "er":      dict(color="#e41a1c", marker="o", ls="-",  label="ER"),
    "sbm":     dict(color="#377eb8", marker="s", ls="-",  label="SBM"),
    "dc_sbm":  dict(color="#ff7f00", marker="^", ls="-",  label="DC-SBM"),
    "lattice": dict(color="#4daf4a", marker="D", ls="--", label="Lattice"),
    "torus":   dict(color="#984ea3", marker="v", ls="--", label="Torus"),
}
GRAPH_ORDER = ["lattice", "torus", "er", "sbm", "dc_sbm"]
SCHEDULERS  = [("sync", "Synchronous"), ("gillespie_occ", "Gillespie-like (energy-weighted async)")]

# ── compute means ─────────────────────────────────────────────────────────────
from collections import defaultdict
groups: dict[tuple, list[float]] = defaultdict(list)
for r in valid:
    groups[(r["_graph_type"], r["_target_k"], r["_scheduler"])].append(ci_2p(r))

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True,
                         constrained_layout=True)
fig.suptitle("Mean coexistence index vs graph density — two-player contest, all grid sizes pooled",
             fontsize=12, fontweight="bold")

for ax, (sched_key, sched_label) in zip(axes, SCHEDULERS):
    for gtype in GRAPH_ORDER:
        st = GRAPH_STYLE[gtype]
        k_vals = sorted({r["_target_k"] for r in valid
                         if r["_graph_type"] == gtype})
        xs, ys, es = [], [], []
        for k in k_vals:
            vals = groups[(gtype, k, sched_key)]
            if len(vals) < 2:
                continue
            a = np.array(vals)
            xs.append(k)
            ys.append(a.mean())
            es.append(1.96 * a.std() / np.sqrt(len(a)))   # 95% CI

        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=es,
                    color=st["color"], marker=st["marker"], ls=st["ls"],
                    markersize=6, linewidth=1.8, capsize=4,
                    label=st["label"])

    ax.set_title(sched_label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean degree k", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_xticks([4, 8, 10, 12, 14, 19])
    ax.grid(alpha=0.25)
    ax.axhline(0.5, color="0.6", lw=0.7, ls=":", alpha=0.7)

axes[0].set_ylabel("Mean coexistence index (stones-normalised)", fontsize=10)
axes[0].legend(fontsize=9, loc="upper right", framealpha=0.85)

out_pdf = FIGURES_DIR / "fig_ci_vs_k.pdf"
out_png = FIGURES_DIR / "fig_ci_vs_k.png"
fig.savefig(out_pdf, dpi=200)
fig.savefig(out_png, dpi=150)
print(f"Saved {out_pdf}\nSaved {out_png}")
