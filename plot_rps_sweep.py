"""
plot_rps_sweep.py  —  Three figures for the RPS density sweep.

Usage:
    python plot_rps_sweep.py                   # loads results/full_final.pkl
    python plot_rps_sweep.py results/my.pkl    # custom file

Outputs (saved to figures/):
    fig1_survivors_heatmap.pdf
    fig2_jain_vs_k.pdf
    fig3_interblock_mixing.pdf
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

# ── load ──────────────────────────────────────────────────────────────────────

PKL_PATH = sys.argv[1] if len(sys.argv) > 1 else "results/full_final.pkl"
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

print(f"Loading {PKL_PATH} …")
with open(PKL_PATH, "rb") as f:
    raw = pickle.load(f)
print(f"  {len(raw)} runs loaded")

# ── helpers ───────────────────────────────────────────────────────────────────

COLOR_LABEL = {1: "black", -1: "white", 2: "red"}
SPECIES_COLOR = {1: "#2166ac", -1: "#d6604d", 2: "#1a9641"}  # blue, red-orange, green
POLICY_ORDER  = ["BBB", "BAB", "BAA", "AAA"]
POLICY_STYLE  = {
    "BBB": dict(color="#d73027", ls="-",  marker="o", label="BBB (all invasive)"),
    "BAB": dict(color="#fc8d59", ls="--", marker="s", label="BAB"),
    "BAA": dict(color="#4575b4", ls="-.", marker="^", label="BAA"),
    "AAA": dict(color="#313695", ls=":",  marker="D", label="AAA (all coexist)"),
}
GRAPH_LABELS = {"er": "ER (random)", "sbm": "SBM (modular)"}

def group_mean_std(runs, key):
    vals = [r[key] for r in runs if r.get(key) is not None]
    if not vals:
        return float("nan"), float("nan")
    a = np.array(vals, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a))

def groupby(runs, keys):
    g = defaultdict(list)
    for r in runs:
        k = tuple(r.get(kk) for kk in keys)
        g[k].append(r)
    return g


# ── subset: RPS only ──────────────────────────────────────────────────────────
runs = [r for r in raw if r["_contest"] == "rps"]

K_VALUES   = sorted({r["_target_k"]  for r in runs})
GRID_SIZES = sorted({r["_n_nodes"]   for r in runs})
POLICIES   = [p for p in POLICY_ORDER if p in {r["_policy"] for r in runs}]
GRAPH_TYPES = ["er", "sbm"]

print(f"  k_values={K_VALUES}  grid_sizes={GRID_SIZES}  policies={POLICIES}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Survivors heatmap: k × grid_size, one panel per graph type
#            colour = mean survivors_count (averaged over policies and reps)
# ─────────────────────────────────────────────────────────────────────────────

fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
fig1.suptitle("Mean survivors count (RPS, Gillespie, all policies pooled)", fontsize=11)

# 3-level discrete colormap: 1 species (red) → 2 (yellow) → 3 (green)
cmap_surv = mcolors.LinearSegmentedColormap.from_list(
    "surv3", ["#d73027", "#fee090", "#1a9641"], N=256
)
norm_surv = mcolors.Normalize(vmin=1, vmax=3)

for ax, gtype in zip(axes1, GRAPH_TYPES):
    mat = np.full((len(GRID_SIZES), len(K_VALUES)), np.nan)
    for gi, n in enumerate(GRID_SIZES):
        for ki, k in enumerate(K_VALUES):
            subset = [r for r in runs
                      if r["_graph_type"] == gtype
                      and r["_n_nodes"] == n
                      and r["_target_k"] == k]
            if subset:
                mat[gi, ki] = np.mean([r["survivors_count"] for r in subset])

    im = ax.imshow(mat, aspect="auto", origin="lower",
                   cmap=cmap_surv, norm=norm_surv,
                   extent=[-0.5, len(K_VALUES)-0.5, -0.5, len(GRID_SIZES)-0.5])

    # annotate cells
    for gi in range(len(GRID_SIZES)):
        for ki in range(len(K_VALUES)):
            v = mat[gi, ki]
            if not np.isnan(v):
                ax.text(ki, gi, f"{v:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if v < 1.8 or v > 2.7 else "black")

    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels(K_VALUES)
    ax.set_yticks(range(len(GRID_SIZES)))
    ax.set_yticklabels(GRID_SIZES)
    ax.set_xlabel("Mean degree k", fontsize=10)
    ax.set_ylabel("Board size (nodes)", fontsize=10)
    ax.set_title(GRAPH_LABELS[gtype], fontsize=10, fontweight="bold")

cb = fig1.colorbar(im, ax=axes1, fraction=0.03, pad=0.02)
cb.set_label("Mean survivors (1–3)", fontsize=9)
cb.set_ticks([1, 1.5, 2, 2.5, 3])

out1 = FIGURES_DIR / "fig1_survivors_heatmap.pdf"
fig1.savefig(out1, dpi=150)
print(f"Saved {out1}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1b — Weakest-species last-30% occupancy heatmap
#             Same axes as Fig1; colour = mean min(occ_avg_last30p) across species
#             Zero = true extinction in the time-averaged sense; ~0.04 = robust cycling
# ─────────────────────────────────────────────────────────────────────────────

fig1b, axes1b = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
fig1b.suptitle(
    "Weakest-species extinction rate (fraction of runs with <0.5% occupancy, last 30%)\n"
    "Cell text = mean weakest-species occupancy",
    fontsize=11,
)

# Green = low extinction (good for coexistence), Red = high extinction (bad)
cmap_occ = mcolors.LinearSegmentedColormap.from_list(
    "ext_rate", ["#1a9641", "#fee090", "#d73027"], N=256
)
norm_occ = mcolors.Normalize(vmin=0.0, vmax=1.0)

for ax, gtype in zip(axes1b, GRAPH_TYPES):
    mat_occ = np.full((len(GRID_SIZES), len(K_VALUES)), np.nan)   # mean weakest occupancy (for text)
    mat_ext = np.full((len(GRID_SIZES), len(K_VALUES)), np.nan)   # extinction rate (for colour)

    for gi, n in enumerate(GRID_SIZES):
        for ki, k in enumerate(K_VALUES):
            subset = [r for r in runs
                      if r["_graph_type"] == gtype
                      and r["_n_nodes"] == n
                      and r["_target_k"] == k
                      and r.get("occ_avg_last30p")]
            if subset:
                weak = [min(r["occ_avg_last30p"].values()) for r in subset]
                mat_occ[gi, ki] = float(np.mean(weak))
                mat_ext[gi, ki] = sum(1 for v in weak if v < 0.005) / len(weak)

    im1b = ax.imshow(mat_ext, aspect="auto", origin="lower",
                     cmap=cmap_occ, norm=norm_occ,
                     extent=[-0.5, len(K_VALUES)-0.5, -0.5, len(GRID_SIZES)-0.5])

    # annotate: extinction rate % (top) and mean occupancy (bottom)
    for gi in range(len(GRID_SIZES)):
        for ki in range(len(K_VALUES)):
            ext = mat_ext[gi, ki]
            occ = mat_occ[gi, ki]
            if not np.isnan(ext):
                # white text on extreme colours (very low or very high extinction)
                fc = "white" if ext < 0.15 or ext > 0.75 else "black"
                ax.text(ki, gi + 0.18, f"{ext:.0%}", ha="center", va="center",
                        fontsize=7, color=fc)
                ax.text(ki, gi - 0.22, f"occ={occ:.3f}", ha="center", va="center",
                        fontsize=6, color=fc, alpha=0.85)

    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels(K_VALUES)
    ax.set_yticks(range(len(GRID_SIZES)))
    ax.set_yticklabels(GRID_SIZES)
    ax.set_xlabel("Mean degree k", fontsize=10)
    ax.set_ylabel("Board size (nodes)", fontsize=10)
    ax.set_title(GRAPH_LABELS[gtype], fontsize=10, fontweight="bold")

cb1b = fig1b.colorbar(im1b, ax=axes1b, fraction=0.03, pad=0.02)
cb1b.set_label("Extinction rate (fraction of runs, weakest species)", fontsize=9)
cb1b.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb1b.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

out1b = FIGURES_DIR / "fig1b_weakest_occ_heatmap.pdf"
fig1b.savefig(out1b, dpi=150)
print(f"Saved {out1b}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Jain fairness index vs k
#            rows = grid sizes (4),  cols = ER / SBM
#            lines = policy (4 colours)
# ─────────────────────────────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(len(GRID_SIZES), 2,
                            figsize=(10, 3.0 * len(GRID_SIZES)),
                            sharex=True, sharey=False,
                            constrained_layout=True)
fig2.suptitle("Jain fairness index vs mean degree k  (RPS, Gillespie)", fontsize=11)

for ri, n in enumerate(GRID_SIZES):
    for ci, gtype in enumerate(GRAPH_TYPES):
        ax = axes2[ri, ci]
        for policy in POLICIES:
            xs, ys, es = [], [], []
            for k in K_VALUES:
                subset = [r for r in runs
                          if r["_graph_type"] == gtype
                          and r["_n_nodes"] == n
                          and r["_target_k"] == k
                          and r["_policy"] == policy]
                if subset:
                    vals = [r["jain_fairness_final"] for r in subset]
                    xs.append(k)
                    ys.append(np.mean(vals))
                    es.append(np.std(vals) / np.sqrt(len(vals)))  # SE
            if xs:
                st = POLICY_STYLE[policy]
                ax.errorbar(xs, ys, yerr=es,
                            color=st["color"], ls=st["ls"],
                            marker=st["marker"], markersize=5,
                            linewidth=1.4, capsize=3,
                            label=st["label"])

        ax.set_ylim(0.2, 1.05)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.axhline(1/3, color="gray", lw=0.8, ls=":", alpha=0.6)  # random baseline

        if ri == 0:
            ax.set_title(GRAPH_LABELS[gtype], fontsize=10, fontweight="bold")
        if ri == len(GRID_SIZES) - 1:
            ax.set_xlabel("Mean degree k", fontsize=9)
        if ci == 0:
            ax.set_ylabel(f"n={n}\nJain index", fontsize=9)
        if ri == 0 and ci == 1:
            ax.legend(fontsize=7.5, loc="lower left", framealpha=0.7)

out2 = FIGURES_DIR / "fig2_jain_vs_k.pdf"
fig2.savefig(out2, dpi=150)
print(f"Saved {out2}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Inter-block mixing rate (SBM only), EXTINCTION RUNS ONLY
#
#   Restricted to n=81 and n=225 where extinction rates are 20-54%.
#   For each (n, policy, k) condition, runs are split:
#     - extinction runs : survivors_count < 3
#     - coexistence runs: survivors_count == 3
#   Only extinction-run series are plotted (this is where the asymmetry lives).
#   Each panel annotates the extinction fraction at each k value.
# ─────────────────────────────────────────────────────────────────────────────

sbm_runs = [r for r in runs
            if r["_graph_type"] == "sbm"
            and r.get("interblock_rate_by_color") is not None]

SPECIES = [1, -1, 2]   # +1 black, -1 white, +2 red
SPECIES_MARKER = {1: "o", -1: "s", 2: "^"}

# Restrict to the two sizes where extinction signal is strongest
FIG3_SIZES = [n for n in GRID_SIZES if n in (81, 225)]
if not FIG3_SIZES:
    FIG3_SIZES = GRID_SIZES  # fallback if sizes differ

fig3, axes3 = plt.subplots(len(FIG3_SIZES), len(POLICIES),
                            figsize=(3.0 * len(POLICIES), 3.2 * len(FIG3_SIZES)),
                            sharex=True, sharey=False,
                            constrained_layout=True)
fig3.suptitle(
    "Inter-block mixing rate — extinction runs only (SBM, Gillespie)\n"
    "n=81 & n=225; number above x-axis = extinction fraction at that k",
    fontsize=11,
)

# Make axes3 always 2-D
if len(FIG3_SIZES) == 1:
    axes3 = axes3[np.newaxis, :]

for ri, n in enumerate(FIG3_SIZES):
    for ci, policy in enumerate(POLICIES):
        ax = axes3[ri, ci]

        # Collect all SBM runs for this (n, policy, k), split by outcome
        ext_frac_at_k = {}  # k -> fraction that were extinction runs
        for k in K_VALUES:
            all_k = [r for r in sbm_runs
                     if r["_n_nodes"] == n
                     and r["_target_k"] == k
                     and r["_policy"] == policy]
            if all_k:
                n_ext = sum(1 for r in all_k if r.get("survivors_count", 3) < 3)
                ext_frac_at_k[k] = n_ext / len(all_k)

        for sp in SPECIES:
            xs, ys, es = [], [], []
            for k in K_VALUES:
                ext_subset = [r for r in sbm_runs
                              if r["_n_nodes"] == n
                              and r["_target_k"] == k
                              and r["_policy"] == policy
                              and r.get("survivors_count", 3) < 3]
                if ext_subset:
                    vals = [r["interblock_rate_by_color"][sp]
                            for r in ext_subset
                            if sp in r["interblock_rate_by_color"]]
                    if vals:
                        xs.append(k)
                        ys.append(np.mean(vals))
                        es.append(np.std(vals) / np.sqrt(len(vals)))
            if xs:
                ax.errorbar(xs, ys, yerr=es,
                            color=SPECIES_COLOR[sp],
                            marker=SPECIES_MARKER[sp],
                            markersize=5, linewidth=1.4, capsize=3,
                            label=f"sp {COLOR_LABEL[sp]}")

        # Annotate extinction fraction above the x-axis ticks
        for k, frac in ext_frac_at_k.items():
            ki = K_VALUES.index(k)
            ax.annotate(f"{frac:.0%}", xy=(k, -0.04), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=6, color="0.4")

        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        # reference line at 1/3 (equal mixing)
        ax.axhline(1/3, color="gray", lw=0.6, ls=":", alpha=0.5)

        if ri == 0:
            ax.set_title(policy, fontsize=10, fontweight="bold")
        if ri == len(FIG3_SIZES) - 1:
            ax.set_xlabel("Mean degree k", fontsize=9)
        if ci == 0:
            ax.set_ylabel(f"n={n}\nInter-block rate\n(extinction runs)", fontsize=9)
        if ri == 0 and ci == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

out3 = FIGURES_DIR / "fig3_interblock_mixing.pdf"
fig3.savefig(out3, dpi=150)
print(f"Saved {out3}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3b — Scatter: Jain index vs capture rate, coloured by inter-block mixing
#             SBM + RPS only.  Panels = k value.
#             x = capture_rate_per_move (per-run)
#             y = jain_fairness_final   (per-run)
#             colour = mean inter-block mixing across species
#             marker size = board size (n_nodes)
#             marker shape = policy
# ─────────────────────────────────────────────────────────────────────────────

# Map n_nodes → marker edge size (area in scatter points²)
SIZE_MAP = {81: 30, 225: 60, 361: 100, 625: 160}
SIZE_LABEL = {81: "9×9 (81)", 225: "15×15 (225)", 361: "19×19 (361)", 625: "25×25 (625)"}
POLICY_MARKER = {"BBB": "o", "BAB": "s", "BAA": "^", "AAA": "D"}

rps_sbm_scatter = [
    r for r in runs
    if r["_graph_type"] == "sbm"
    and r.get("interblock_rate_by_color") is not None
    and r.get("capture_rate_per_move") is not None
    and r.get("jain_fairness_final") is not None
]

K_VALS_3B = sorted({r["_target_k"] for r in rps_sbm_scatter})
ncols_3b = len(K_VALS_3B)

fig3b, axes3b = plt.subplots(1, ncols_3b, figsize=(3.2 * ncols_3b, 3.8),
                              constrained_layout=True, sharey=True)
fig3b.suptitle(
    "RPS · SBM ecology: coexistence vs capture rate\n"
    "fill = inter-block mixing (mean across species),  "
    "size = board size,  panels = target degree",
    fontsize=10,
)

# Shared colour scale across all panels
all_mix = [np.mean(list(r["interblock_rate_by_color"].values()))
           for r in rps_sbm_scatter]
vmin_mix, vmax_mix = 0.0, np.percentile(all_mix, 97)  # clip top outliers

cmap_mix = plt.cm.viridis
norm_mix = mcolors.Normalize(vmin=vmin_mix, vmax=vmax_mix)

for ax, k in zip(axes3b, K_VALS_3B):
    subset_k = [r for r in rps_sbm_scatter if r["_target_k"] == k]

    for policy in POLICIES:
        subset_p = [r for r in subset_k if r["_policy"] == policy]
        if not subset_p:
            continue
        xs = [r["capture_rate_per_move"] for r in subset_p]
        ys = [r["jain_fairness_final"]   for r in subset_p]
        cs = [np.mean(list(r["interblock_rate_by_color"].values())) for r in subset_p]
        sz = [SIZE_MAP.get(r["_n_nodes"], 50) for r in subset_p]

        sc = ax.scatter(xs, ys, c=cs, s=sz,
                        cmap=cmap_mix, norm=norm_mix,
                        marker=POLICY_MARKER[policy],
                        edgecolors="0.3", linewidths=0.4, alpha=0.75,
                        label=policy)

    ax.set_title(f"k = {k}", fontsize=10)
    ax.set_xlabel("Capture rate per move", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.25, linewidth=0.5)

axes3b[0].set_ylabel("Jain fairness index", fontsize=9)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap_mix, norm=norm_mix)
sm.set_array([])
cb3b = fig3b.colorbar(sm, ax=axes3b, fraction=0.025, pad=0.02)
cb3b.set_label("Inter-block mixing (mean across species)", fontsize=8)

# Legend: size → board size, marker → policy
from matplotlib.lines import Line2D
size_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="0.4",
           markeredgecolor="0.3", markersize=np.sqrt(s),
           label=SIZE_LABEL[n])
    for n, s in SIZE_MAP.items()
]
policy_handles = [
    Line2D([0], [0], marker=POLICY_MARKER[p], color="w",
           markerfacecolor="0.5", markeredgecolor="0.3",
           markersize=7, label=p)
    for p in POLICIES
]
axes3b[-1].legend(
    handles=size_handles + policy_handles,
    fontsize=6.5, loc="upper right", framealpha=0.7,
    title="size / policy", title_fontsize=6.5,
)

out3b = FIGURES_DIR / "fig3b_jain_vs_capture_scatter.pdf"
fig3b.savefig(out3b, dpi=150)
print(f"Saved {out3b}")

print("\nAll figures saved to figures/")
