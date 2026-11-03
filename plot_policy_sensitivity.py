"""
plot_policy_sensitivity.py — Diagnostic policy sensitivity check. Rough only.

2×2 figure:
  Row 0: two-player   | sync (left)  | Gillespie-like (right)
  Row 1: three-player | RPS (left)   | AVA (right)

x-axis: policy label; one line per graph type; error bars = 89% HDI of each mean (Bayesian bootstrap, seeded).
2026-11-03 (MA): error bars were ±1 std (the spread of runs); now the 89% HDI of the mean, as elsewhere in m1.
All k and grid sizes pooled.
"""
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az
from analysis_helpers import ci_2p

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────
with open("results/two_p_all_v1.pkl",  "rb") as fh: twop = pickle.load(fh)
with open("results/full_all_v2.pkl",   "rb") as fh: rps  = pickle.load(fh)
with open("results/ava_all_v1.pkl",    "rb") as fh: ava  = pickle.load(fh)

twop = [r for r in twop if not r.get("connectivity_failed") and r.get("black_occupancy") is not None]
# 2026-11-01 (DEFECT-NOTE: D8, dev repo OPEN_DEFECTS.md): the filter was r.get("black_occupancy"), which dropped every game black
# ended with no stones (1,998 of 13,303) and inflated the two-player means; repaired with MA's OK.
rps  = [r for r in rps  if not r.get("connectivity_failed") and r.get("jain_fairness_final")]
ava  = [r for r in ava  if not r.get("connectivity_failed") and r.get("jain_fairness_final")]

# ── style ─────────────────────────────────────────────────────────────────────
COLORS = {"er":"#e41a1c","sbm":"#377eb8","dc_sbm":"#ff7f00",
          "lattice":"#4daf4a","torus":"#984ea3"}
LABELS = {"er":"ER","sbm":"SBM","dc_sbm":"DC-SBM","lattice":"lattice","torus":"torus"}
GTYPES_2P = ["er","sbm","dc_sbm","lattice","torus"]
GTYPES_3P = ["er","sbm","dc_sbm","lattice","torus"]

POLICIES_2P  = ["AA","AB","BA","BB"]
POLICIES_3P  = ["AAA","BAA","BAB","BBB"]

def means_by_policy(data, policies, metric_fn, gtype):
    out = {}
    for p in policies:
        vals = [metric_fn(r) for r in data
                if r["_graph_type"] == gtype and r["_policy"] == p]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        out[p] = (np.mean(vals), mean_hdi(np.array(vals, float)), len(vals)) if vals else (np.nan, (np.nan, np.nan), 0)
    return out


RNG = np.random.default_rng(20261103)


def mean_hdi(vals, draws=4000, prob=0.89):
    """89% HDI of the mean: Bayesian bootstrap (Dirichlet(1, ..., 1) weights over the runs)."""
    m = RNG.dirichlet(np.ones(len(vals)), size=draws) @ vals
    lo, hi = az.hdi(m, hdi_prob=prob)
    return float(lo), float(hi)

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

PANEL_SPECS = [
    # (ax, data, policies, metric_fn, gtypes, title)
    (axes[0,0], [r for r in twop if r["_scheduler"]=="sync"],
     POLICIES_2P, ci_2p, GTYPES_2P, "Two-player — Synchronous"),
    (axes[0,1], [r for r in twop if r["_scheduler"]=="gillespie_occ"],
     POLICIES_2P, ci_2p, GTYPES_2P, "Two-player — Gillespie-like"),
    (axes[1,0], rps, POLICIES_3P, lambda r: r["jain_fairness_final"],
     GTYPES_3P, "Three-player RPS — Gillespie-like"),
    (axes[1,1], ava, POLICIES_3P, lambda r: r["jain_fairness_final"],
     GTYPES_3P, "Three-player AVA — Gillespie-like"),
]

xs_2p = np.arange(len(POLICIES_2P))
xs_3p = np.arange(len(POLICIES_3P))

for ax, data, policies, metric_fn, gtypes, title in PANEL_SPECS:
    xs = xs_2p if len(policies) == 4 and policies[0] == "AA" else xs_3p
    offsets = np.linspace(-0.18, 0.18, len(gtypes))

    for i, gtype in enumerate(gtypes):
        stats = means_by_policy(data, policies, metric_fn, gtype)
        ys = [stats[p][0] for p in policies]
        es = np.array([[ys[j] - stats[p][1][0] for j, p in enumerate(policies)],
                       [stats[p][1][1] - ys[j] for j, p in enumerate(policies)]])
        ns = [stats[p][2] for p in policies]
        xpos = xs + offsets[i]
        ax.errorbar(xpos, ys, yerr=es,
                    fmt="o-", color=COLORS[gtype], label=LABELS[gtype],
                    markersize=5, linewidth=1.2, capsize=3, alpha=0.85)

    ax.set_xticks(xs)
    ax.set_xticklabels(policies, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Mean coexistence index" if "Two" in title else "Mean Jain index")
    ax.grid(axis="y", alpha=0.3)
    if ax is axes[0,0]:
        ax.legend(fontsize=8, loc="upper left", framealpha=0.8)

out = FIGURES_DIR / "fig_policy_sensitivity.png"
fig.savefig(out, dpi=150)
fig.savefig(out.with_suffix(".pdf"))   # generate_supplementary_figures.py copies the .pdf to figures/final/S1.pdf
print(f"Saved {out} and {out.with_suffix('.pdf')}")
