"""
plot_eps_policy_sensitivity.py — Policy sensitivity analysis for the epsilon sweep.

Two figures:

fig_eps_policy_sensitivity.pdf
  4 panels (one per ε), each showing mean CI by policy for all graph types.
  Pools k=10 and k=14, n=225 and n=625.  Error bars ±1 SD across replicates.
  Answers: does policy effect appear at lower ε, and is it topology-specific?

fig_eps_sbm_ci_distribution.pdf
  4 panels (one per ε), SBM n=625, k=10 and k=14 pooled.
  KDE of CI distribution, one curve per policy.
  Answers: is the wide variance at SBM n=625 policy-driven or replicate noise?

Key question: if KDE curves overlap → insensitivity holds.
             If curves separate → policy effect is real at that ε.
"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, t as t_dist

from analysis_helpers import ci_2p

# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────

with open("results/eps_sweep_2p_v1.pkl", "rb") as fh:
    raw = pickle.load(fh)

valid = [r for r in raw if not r.get("connectivity_failed")]
print(f"Loaded {len(raw)} records, {len(valid)} valid")

# ─────────────────────────────────────────────
# Style constants
# ─────────────────────────────────────────────

GTYPE_COLORS = {
    "er":      "#e41a1c",
    "sbm":     "#377eb8",
    "dc_sbm":  "#ff7f00",
    "lattice": "#4daf4a",
    "torus":   "#984ea3",
}
GTYPE_LABELS = {
    "er": "ER", "sbm": "SBM", "dc_sbm": "DC-SBM",
    "lattice": "Lattice", "torus": "Torus",
}

POLICY_COLORS = {
    "AA": "#1b9e77",
    "AB": "#d95f02",
    "BA": "#7570b3",
    "BB": "#e7298a",
}

POLICIES  = ["AA", "AB", "BA", "BB"]
GTYPES    = ["er", "sbm", "dc_sbm", "lattice", "torus"]

# ε=0.04 and 0.07 are irregular-graph only; all four values present for ER/SBM/DC-SBM
EPS_IRREGULAR = [0.02, 0.04, 0.07, 0.15]
EPS_ALL       = [0.02, 0.04, 0.07, 0.15]   # used as panel order

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_ci(r):
    return ci_2p(r)


def means_by_policy(records, policies, gtype):
    """Return {policy: (mean, lo_err, hi_err, n)} pooled over all k and grid sizes.
    Error bars are 89% HDI on the mean (t-interval: posterior on mean under
    normal model with unknown variance, equivalent to frequentist 89% CI)."""
    out = {}
    for p in policies:
        vals = [get_ci(r) for r in records
                if r["_graph_type"] == gtype and r["_policy"] == p]
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) >= 2:
            mean = float(np.mean(vals))
            se   = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            lo, hi = t_dist.interval(0.89, df=len(vals) - 1, loc=mean, scale=se)
            out[p] = (mean, max(0.0, mean - lo), max(0.0, hi - mean), len(vals))
        elif len(vals) == 1:
            out[p] = (float(vals[0]), 0.0, 0.0, 1)
        else:
            out[p] = (np.nan, 0.0, 0.0, 0)
    return out


# ─────────────────────────────────────────────
# Figure 1 — Policy sensitivity, faceted by ε
# ─────────────────────────────────────────────

fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
fig1.suptitle(
    "Two-player policy sensitivity by ε  "
    "(Gillespie-like, k=10 & 14 pooled, n=225 & 625 pooled)\n"
    "y-axis: mean CI (stones-normalised)  ·  error bars = 89% HDI on the mean",
    fontsize=10,
)

xs = np.arange(len(POLICIES))
offsets = np.linspace(-0.20, 0.20, len(GTYPES))

for panel_idx, eps_val in enumerate(EPS_ALL):
    ax = axes1.flat[panel_idx]
    subset = [r for r in valid if r.get("_eps") == eps_val]

    for gi, gtype in enumerate(GTYPES):
        stats = means_by_policy(subset, POLICIES, gtype)
        ys      = [stats[p][0] for p in POLICIES]
        lo_errs = [stats[p][1] for p in POLICIES]
        hi_errs = [stats[p][2] for p in POLICIES]
        ns      = [stats[p][3] for p in POLICIES]

        # Skip graph types with no data for this ε (lattice/torus at ε=0.04, 0.07)
        if all(np.isnan(y) for y in ys):
            continue

        mask    = [np.isfinite(y) for y in ys]
        xpos    = xs[mask] + offsets[gi]
        yv      = np.array(ys)[mask]
        yerr    = np.array([[lo_errs[i] for i, m in enumerate(mask) if m],
                            [hi_errs[i] for i, m in enumerate(mask) if m]])
        ax.errorbar(
            xpos, yv, yerr=yerr,
            fmt="o-", color=GTYPE_COLORS[gtype], label=GTYPE_LABELS[gtype],
            markersize=5, linewidth=1.3, capsize=3, alpha=0.85,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(POLICIES, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"ε = {eps_val}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean CI (stones-normalised)", fontsize=9)
    ax.axhline(0.5, color="0.7", lw=0.8, ls="--")
    ax.grid(axis="y", alpha=0.25)

    # Note which graph types are missing at this ε
    if eps_val in (0.04, 0.07):
        ax.annotate(
            "Lattice & torus: ε ∈ {0.02, 0.15} only",
            xy=(0.01, 0.02), xycoords="axes fraction",
            fontsize=7, color="0.5", va="bottom",
        )

    if panel_idx == 0:
        ax.legend(fontsize=8, loc="upper left", framealpha=0.85)

out1 = FIGURES_DIR / "fig_eps_policy_sensitivity.pdf"
fig1.savefig(out1, bbox_inches="tight")
fig1.savefig(out1.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"Saved {out1}")


# ─────────────────────────────────────────────
# Figure 2 — SBM n=625 CI distributions by policy and ε
# ─────────────────────────────────────────────

# Filter: SBM, grid_size=25 (n=625), pool k=10 and k=14
sbm625 = [
    r for r in valid
    if r["_graph_type"] == "sbm" and r["_grid_size"] == 25
]
print(f"\nSBM n=625 records: {len(sbm625)}")

fig2, axes2 = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
fig2.suptitle(
    "CI distributions — SBM n=625  (k=10 & 14 pooled)  "
    "Gillespie-like two-player",
    fontsize=11,
)

ci_grid = np.linspace(0, 1, 300)

for panel_idx, eps_val in enumerate(EPS_ALL):
    ax = axes2[panel_idx]
    subset = [r for r in sbm625 if r.get("_eps") == eps_val]

    if not subset:
        ax.set_title(f"ε = {eps_val}\n(no data)", fontsize=10)
        continue

    any_plotted = False
    for policy in POLICIES:
        vals = [get_ci(r) for r in subset if r["_policy"] == policy]
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) < 3:
            continue
        kde = gaussian_kde(vals, bw_method=0.15)
        ax.plot(ci_grid, kde(ci_grid),
                color=POLICY_COLORS[policy], label=policy,
                linewidth=1.8, alpha=0.85)
        any_plotted = True

    # Annotate policy spread: max mean - min mean across policies
    policy_means = []
    for policy in POLICIES:
        vals = [get_ci(r) for r in subset if r["_policy"] == policy]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            policy_means.append(np.mean(vals))
    if len(policy_means) >= 2:
        spread = max(policy_means) - min(policy_means)
        ax.annotate(
            f"policy spread: {spread:.3f}",
            xy=(0.03, 0.97), xycoords="axes fraction",
            fontsize=8, va="top",
            color="0.3",
        )

    ax.set_xlim(0, 1)
    ax.set_xlabel("CI (stones-normalised)", fontsize=9)
    ax.set_ylabel("Density" if panel_idx == 0 else "", fontsize=9)
    ax.set_title(f"ε = {eps_val}  (n={len(subset)} runs)", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.2)

    if panel_idx == 0:
        ax.legend(title="Policy", fontsize=8, framealpha=0.85)

out2 = FIGURES_DIR / "fig_eps_sbm_ci_distribution.pdf"
fig2.savefig(out2, bbox_inches="tight")
fig2.savefig(out2.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"Saved {out2}")


# ─────────────────────────────────────────────
# Console summary: policy spread per ε × graph type
# ─────────────────────────────────────────────

print("\n── Policy spread (max mean CI − min mean CI) ──")
print(f"{'eps':>6}  {'graph':<8}  {'spread':>8}  {'min_policy':<10}  {'max_policy':<10}  {'n/cell':>7}")
print("-" * 58)

for eps_val in EPS_ALL:
    subset_eps = [r for r in valid if r.get("_eps") == eps_val]
    for gtype in GTYPES:
        subset_g = [r for r in subset_eps if r["_graph_type"] == gtype]
        if not subset_g:
            continue
        policy_means = {}
        for p in POLICIES:
            vals = [get_ci(r) for r in subset_g if r["_policy"] == p]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                policy_means[p] = (np.mean(vals), len(vals))
        if len(policy_means) < 2:
            continue
        means_only = {p: v[0] for p, v in policy_means.items()}
        spread     = max(means_only.values()) - min(means_only.values())
        min_p      = min(means_only, key=means_only.get)
        max_p      = max(means_only, key=means_only.get)
        n_per_cell = list(policy_means.values())[0][1]
        print(f"{eps_val:>6.2f}  {gtype:<8}  {spread:>8.4f}  {min_p:<10}  {max_p:<10}  {n_per_cell:>7}")
    print()
