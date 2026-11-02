"""
plot_eps_dcsbm_ava_coexistence.py — AVA coexistence rate by ε for DC-SBM vs ER/SBM.

Metric: three-way coexistence rate (survivors == 3), pooled over k=10 & 14 and
        n=225 & 625.  Error bars: 89% HDI (Jeffreys beta prior, analytically
        computed).

Note on Jain≥0.85 threshold: on irregular graphs (ER/SBM/DC-SBM), nearly all
survivors=3 runs have Jain ~0.67–0.71 (one species dominant, others surviving).
Only 0.5–7.5% pass Jain≥0.85, so that threshold would suppress the ε signal.
survivors=3 is the appropriate coexistence metric here.

Layout: 1 row × 3 panels  (DC-SBM | ER | SBM)
  - Each panel: 4 lines, one per policy, x = ε value
  - 89% HDI error bars (asymmetric)
  - Reference line at pooled mean across all ε and policies

Output: figures/fig_eps_dcsbm_ava_coexistence.pdf
"""

import pickle
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────

with open("results/eps_sweep_ava_v1.pkl", "rb") as fh:
    raw = pickle.load(fh)

POLICIES = ["BBB", "BAB", "BAA", "AAA"]
EPS      = [0.02, 0.04, 0.07, 0.15]
PANELS   = [
    ("dc_sbm", "DC-SBM  (ε effect present)"),
    ("er",     "ER  (no ε effect)"),
    ("sbm",    "SBM  (no ε effect)"),
]

HDI_LO, HDI_HI = 5.5, 94.5   # 89% HDI

POLICY_COLORS = {
    "BBB": "#d7191c",   # red    — all invasive
    "BAB": "#f07c28",   # orange — 2 invasive
    "BAA": "#4daf4a",   # green  — 1 invasive
    "AAA": "#377eb8",   # blue   — all coexist
}
POLICY_MARKERS = {
    "BBB": "o",
    "BAB": "s",
    "BAA": "^",
    "AAA": "D",
}

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def jeffreys_hdi(k, n):
    """89% HDI for proportion k/n with Jeffreys Beta(0.5, 0.5) prior."""
    lo = float(stats.beta.ppf(HDI_LO / 100, k + 0.5, n - k + 0.5))
    hi = float(stats.beta.ppf(HDI_HI / 100, k + 0.5, n - k + 0.5))
    return lo, hi


def coex_stats(gt, pol, eps):
    sub = [r for r in raw if r["_graph_type"] == gt
           and r["_policy"] == pol and r["_eps"] == eps]
    if not sub:
        return np.nan, np.nan, np.nan, 0
    k = sum(1 for r in sub if r.get("survivors_count", 0) == 3)
    n = len(sub)
    p = k / n
    lo, hi = jeffreys_hdi(k, n)
    return p, lo, hi, n


def pooled_mean(gt):
    sub = [r for r in raw if r["_graph_type"] == gt]
    k = sum(1 for r in sub if r.get("survivors_count", 0) == 3)
    return k / len(sub) if sub else np.nan


# ─────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                         sharey=True, constrained_layout=True)
fig.suptitle(
    "AVA all-three-species presence by ε — DC-SBM vs ER vs SBM\n"
    "Gillespie-like scheduler  ·  k=10 & 14 pooled  ·  n=225 & 625 pooled  ·  "
    "error bars = 89% HDI (Jeffreys prior)",
    fontsize=10,
)

x = np.array(EPS)

for ax, (gt, title) in zip(axes, PANELS):
    ref = pooled_mean(gt)
    ax.axhline(ref, color="0.6", lw=0.9, ls="--", zorder=1)
    ax.annotate(f"overall mean {ref:.3f}", xy=(0.62, ref), xycoords=("axes fraction", "data"),
                ha="center", va="center", fontsize=7.5, color="0.35", zorder=2.5,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))   # every panel (E8, 2026-11-01)

    for pol in POLICIES:
        ps, los, his, ns = [], [], [], []
        for e in EPS:
            p, lo, hi, n = coex_stats(gt, pol, e)
            ps.append(p); los.append(lo); his.append(hi); ns.append(n)

        ps  = np.array(ps)
        lo_err = ps - np.array(los)
        hi_err = np.array(his) - ps

        ax.errorbar(
            x, ps,
            yerr=[lo_err, hi_err],
            fmt=POLICY_MARKERS[pol] + "-",
            color=POLICY_COLORS[pol],
            label=pol,
            markersize=6,
            linewidth=1.6,
            capsize=3.5,
            capthick=1.1,
            alpha=0.90,
            zorder=3,
        )

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("ε (stochastic exploration rate)", fontsize=9)
    ax.set_xticks(EPS)
    ax.set_xticklabels([str(e) for e in EPS], fontsize=9)
    ax.set_xlim(0.005, 0.175)
    ax.set_ylim(0.50, 1.02)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    # Annotate n per cell (one annotation per panel, bottom right)
    # Use the BBB/ε=0.02 cell as representative n
    _, _, _, n_rep = coex_stats(gt, "BBB", 0.02)
    ax.annotate(f"n≈{n_rep}/cell", xy=(0.97, 0.03),
                xycoords="axes fraction", ha="right", va="bottom",
                fontsize=7.5, color="0.4")

axes[0].set_ylabel("All-three-species presence  (survivors = 3)", fontsize=9)

# Legend on the DC-SBM panel (first)
handles, lbls = axes[0].get_legend_handles_labels()
axes[0].legend(handles, lbls, fontsize=8.5, loc="lower right",
               framealpha=0.88, title="Policy", title_fontsize=8)

out = FIGURES_DIR / "fig_eps_dcsbm_ava_coexistence.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"Saved {out}")

# ─────────────────────────────────────────────
# Console table
# ─────────────────────────────────────────────

for gt, title in PANELS:
    print(f"\n── {title} ──")
    print(f"  {'Policy':<6}  " + "  ".join(f"ε={e:4.2f}" for e in EPS))
    print("  " + "-" * 52)
    for pol in POLICIES:
        row = f"  {pol:<6}  "
        for e in EPS:
            p, lo, hi, n = coex_stats(gt, pol, e)
            row += f"{p:.3f}  " if np.isfinite(p) else "  —    "
        print(row)
    print(f"  Reference mean: {pooled_mean(gt):.3f}")
