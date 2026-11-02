"""
plot_occ_trajectories.py — Occupancy trajectory analysis for three competition modes.

Q1 (RPS, fig_occ_trajectories_rps.pdf):
  BAA policy runs only. Split into extinction (survivors<3) and coexistence (survivors=3).
  For extinction runs, identify which A-strategy color went extinct and which survived.
  Plot mean trajectories ± 89% HDI: B / A-extinct / A-surviving across both outcomes.

Q2 (AVA BBB, fig_occ_trajectories_ava_bbb.pdf):
  BBB policy. Split by survivors_count: 1 (monopoly), 2 (partial), 3 (coexistence).
  For each run, rank colors by final occupancy → winner / 2nd / loser roles.
  Monopoly (n=15): individual spaghetti traces + bold mean. Others: mean + 89% HDI.

Q3 (DC-SBM AVA, fig_occ_trajectories_dcsbm_eps.pdf):
  All policies, dc_sbm topology only. Compare ε=0.02 vs ε=0.15.
  Left panel: survival fraction over time (fraction of runs still 3-way coexisting)
  for both ε — directly shows early vs late divergence.
  Right panel: mean occupancy trajectories split by ε and outcome (extinction vs coexistence).

Time axis: 250 points × occ_every=20 → turns 0, 20, …, 4980.
Occupancy is already stored as fraction of n_nodes — no further normalisation needed.
"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

COLORS_3P = [+1, -1, +2]           # black, white, red
N_TURNS   = 250
OCC_EVERY = 20
TURNS     = np.arange(N_TURNS) * OCC_EVERY   # 0, 20, …, 4980
HDI_LO, HDI_HI = 5.5, 94.5          # 89% HDI

def hdi_band(arr2d):
    """arr2d shape (n_runs, 250) → lo (250,), hi (250,)."""
    lo = np.percentile(arr2d, HDI_LO, axis=0)
    hi = np.percentile(arr2d, HDI_HI, axis=0)
    return lo, hi

def get_ts(r, color):
    return np.array(r["occ_ts_by_color"][color])

def stack_ts(runs, color):
    """Stack occ_ts for a color across runs → (n_runs, 250)."""
    return np.array([get_ts(r, color) for r in runs])

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

print("Loading pkl files …")
with open("results/eps_sweep_rps_v1.pkl", "rb") as fh:
    rps = pickle.load(fh)
with open("results/eps_sweep_ava_v1.pkl", "rb") as fh:
    ava = pickle.load(fh)

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1 — RPS BAA extinction vs coexistence trajectories
# ═══════════════════════════════════════════════════════════════════════

# Policy BAA: color +1 = B, -1 = A, +2 = A
baa_ext  = [r for r in rps if r["_policy"] == "BAA"
            and r.get("survivors_count", 3) < 3]
baa_coex = [r for r in rps if r["_policy"] == "BAA"
            and r.get("survivors_count", 3) == 3]

print(f"\nQ1 RPS BAA: {len(baa_ext)} extinction, {len(baa_coex)} coexistence runs")

# For extinction runs: identify which A color went extinct and which survived
b_ext, a_extc, a_surv = [], [], []
for r in baa_ext:
    ts = r["occ_ts_by_color"]
    # Extinct A color is whichever of {-1, +2} ends near zero
    if ts[-1][-1] < ts[+2][-1]:
        c_extinct, c_survive = -1, +2
    else:
        c_extinct, c_survive = +2, -1
    b_ext.append(get_ts(r, +1))
    a_extc.append(get_ts(r, c_extinct))
    a_surv.append(get_ts(r, c_survive))

b_ext  = np.array(b_ext)    # (160, 250)
a_extc = np.array(a_extc)
a_surv = np.array(a_surv)

# For coexistence runs: mean of both A colors per run
b_coex = stack_ts(baa_coex, +1)
a_coex = np.array([(get_ts(r, -1) + get_ts(r, +2)) / 2 for r in baa_coex])

# Subsample coexistence to 300 for equal visual weight
rng = np.random.default_rng(42)
idx = rng.choice(len(baa_coex), size=min(300, len(baa_coex)), replace=False)
b_coex  = b_coex[idx]
a_coex  = a_coex[idx]
print(f"  Coexistence sample: {len(idx)} runs shown")

fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
fig1.suptitle(
    "RPS BAA policy — mean occupancy trajectories ± 89% HDI\n"
    "Policy: color +1 = B (invasive),  colors −1 & +2 = A (coexist)",
    fontsize=10,
)

# Extinction panel
ax = axes1[0]
ax.set_title(f"Extinction runs  (survivors < 3,  n={len(baa_ext)})",
             fontsize=10, fontweight="bold")
for arr, label, color in [
    (b_ext,  "B (invasive)",     "#e41a1c"),
    (a_extc, "A extinct",        "#984ea3"),
    (a_surv, "A surviving",      "#377eb8"),
]:
    m   = arr.mean(axis=0)
    lo, hi = hdi_band(arr)
    ax.plot(TURNS, m, color=color, lw=1.8, label=label)
    ax.fill_between(TURNS, lo, hi, color=color, alpha=0.18)

ax.axhline(1/3, color="0.6", lw=0.8, ls="--", label="1/3 equal share")
ax.set_xlabel("Game turn", fontsize=9)
ax.set_ylabel("Occupancy fraction", fontsize=9)
ax.set_xlim(0, 4980); ax.set_ylim(0, 1)
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.85)
ax.grid(alpha=0.2)

# Coexistence panel
ax = axes1[1]
ax.set_title(f"Coexistence runs  (survivors = 3,  sample n={len(idx)})",
             fontsize=10, fontweight="bold")
for arr, label, color in [
    (b_coex, "B (invasive)",  "#e41a1c"),
    (a_coex, "A (mean of both)", "#377eb8"),
]:
    m   = arr.mean(axis=0)
    lo, hi = hdi_band(arr)
    ax.plot(TURNS, m, color=color, lw=1.8, label=label)
    ax.fill_between(TURNS, lo, hi, color=color, alpha=0.18)

ax.axhline(1/3, color="0.6", lw=0.8, ls="--", label="1/3 equal share")
ax.set_xlabel("Game turn", fontsize=9)
ax.set_ylabel("Occupancy fraction", fontsize=9)
ax.set_xlim(0, 4980); ax.set_ylim(0, 1)
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.85)
ax.grid(alpha=0.2)

out1 = FIGURES_DIR / "fig_occ_trajectories_rps.pdf"
fig1.savefig(out1, bbox_inches="tight")
fig1.savefig(out1.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"Saved {out1}")
plt.close(fig1)

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2 — AVA BBB trajectories by final outcome
# ═══════════════════════════════════════════════════════════════════════

bbb_mono  = [r for r in ava if r["_policy"] == "BBB" and r.get("survivors_count") == 1]
bbb_part  = [r for r in ava if r["_policy"] == "BBB" and r.get("survivors_count") == 2]
bbb_coex  = [r for r in ava if r["_policy"] == "BBB" and r.get("survivors_count") == 3]

print(f"\nQ2 AVA BBB: monopoly={len(bbb_mono)}, partial={len(bbb_part)}, coexistence={len(bbb_coex)}")

def rank_colors(r, n_colors=3):
    """Rank colors by final occupancy: index 0 = winner (highest)."""
    ts  = r["occ_ts_by_color"]
    final = {c: ts[c][-1] for c in COLORS_3P}
    ordered = sorted(COLORS_3P, key=lambda c: final[c], reverse=True)
    return [np.array(ts[c]) for c in ordered]  # [winner_ts, 2nd_ts, 3rd_ts]

# Build role-stacked arrays for each outcome
def stack_by_role(runs):
    """Returns (winner, second, third) each (n_runs, 250)."""
    ranked = [rank_colors(r) for r in runs]
    return (np.array([x[0] for x in ranked]),
            np.array([x[1] for x in ranked]),
            np.array([x[2] for x in ranked]))

mono_W, mono_2, mono_3 = stack_by_role(bbb_mono)
part_W, part_2, part_3 = stack_by_role(bbb_part)
coex_W, coex_2, coex_3 = stack_by_role(bbb_coex)

ROLE_COLORS = ["#e41a1c", "#ff7f00", "#377eb8"]   # winner, 2nd, 3rd
ROLE_LABELS = ["Winner / dominant", "2nd place", "Loser / extinct"]

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), sharey=True,
                            constrained_layout=True)
fig2.suptitle(
    "AVA BBB policy — occupancy trajectories by final outcome\n"
    "Colors ranked by final occupancy: red=winner, orange=2nd, blue=loser",
    fontsize=10,
)

panels = [
    (axes2[0], bbb_mono, mono_W, mono_2, mono_3,
     f"Monopoly  (survivors=1,  n={len(bbb_mono)})"),
    (axes2[1], bbb_part, part_W, part_2, part_3,
     f"Partial extinction  (survivors=2,  n={len(bbb_part)})"),
    (axes2[2], bbb_coex, coex_W, coex_2, coex_3,
     f"Coexistence  (survivors=3,  n={len(bbb_coex)})"),
]

for ax, runs, W, S, L, title in panels:
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    is_mono = len(runs) == len(bbb_mono)

    for arr, label, color in zip([W, S, L], ROLE_LABELS, ROLE_COLORS):
        m = arr.mean(axis=0)
        if is_mono:
            # Spaghetti for small n
            for trace in arr:
                ax.plot(TURNS, trace, color=color, lw=0.6, alpha=0.35)
            ax.plot(TURNS, m, color=color, lw=2.2, label=label)
        else:
            lo, hi = hdi_band(arr)
            ax.plot(TURNS, m, color=color, lw=1.8, label=label)
            ax.fill_between(TURNS, lo, hi, color=color, alpha=0.18)

    ax.axhline(1/3, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("Game turn", fontsize=9)
    if ax is axes2[0]:
        ax.set_ylabel("Occupancy fraction", fontsize=9)
    ax.set_xlim(0, 4980); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.grid(alpha=0.2)

out2 = FIGURES_DIR / "fig_occ_trajectories_ava_bbb.pdf"
fig2.savefig(out2, bbox_inches="tight")
fig2.savefig(out2.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"Saved {out2}")
plt.close(fig2)

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3 — DC-SBM AVA ε=0.02 vs ε=0.15
# ═══════════════════════════════════════════════════════════════════════

dc_02 = [r for r in ava if r["_graph_type"] == "dc_sbm" and r["_eps"] == 0.02]
dc_15 = [r for r in ava if r["_graph_type"] == "dc_sbm" and r["_eps"] == 0.15]

print(f"\nQ3 DC-SBM: ε=0.02 n={len(dc_02)}, ε=0.15 n={len(dc_15)}")

def survival_fraction(runs, threshold=0.001):
    """At each time point, fraction of runs where all 3 colors have occ > threshold."""
    n = len(runs)
    alive = np.zeros(N_TURNS)
    for r in runs:
        ts = r["occ_ts_by_color"]
        alive += np.array([
            int(all(ts[c][t] > threshold for c in COLORS_3P))
            for t in range(N_TURNS)
        ])
    return alive / n

def mean_trajectories_by_outcome(runs):
    """Returns (coex_mean, coex_lo, coex_hi, ext_mean, ext_lo, ext_hi, n_coex, n_ext)
    where each array is (250,); averages across all 3 colors and all runs."""
    coex_runs = [r for r in runs if r.get("survivors_count", 0) == 3]
    ext_runs  = [r for r in runs if r.get("survivors_count", 0) < 3]
    results = []
    for group in [coex_runs, ext_runs]:
        if not group:
            results += [np.zeros(N_TURNS)] * 3 + [0]
            continue
        # Mean over all 3 colors per run, then stack runs
        avg_per_run = np.array([
            np.mean([get_ts(r, c) for c in COLORS_3P], axis=0)
            for r in group
        ])   # (n_group, 250)
        m = avg_per_run.mean(axis=0)
        lo, hi = hdi_band(avg_per_run)
        results += [m, lo, hi, len(group)]
    return results

sf_02 = survival_fraction(dc_02)
sf_15 = survival_fraction(dc_15)

coex_m02, coex_lo02, coex_hi02, n_coex02, \
ext_m02,  ext_lo02,  ext_hi02,  n_ext02  = mean_trajectories_by_outcome(dc_02)

coex_m15, coex_lo15, coex_hi15, n_coex15, \
ext_m15,  ext_lo15,  ext_hi15,  n_ext15  = mean_trajectories_by_outcome(dc_15)

fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
fig3.suptitle(
    "DC-SBM AVA — ε=0.02 vs ε=0.15  (all policies pooled,  n=416 per ε)\n"
    "Left: fraction of runs with all three species present over time  ·  "
    "Right: mean occupancy (all 3 colors averaged) split by outcome",
    fontsize=9.5,
)

# Left: survival fraction
ax = axes3[0]
ax.plot(TURNS, sf_02, color="#e41a1c", lw=2.0, label=f"ε=0.02  ({sf_02[-1]*100:.1f}% at turn {TURNS[-1]:,})")
ax.plot(TURNS, sf_15, color="#377eb8", lw=2.0, label=f"ε=0.15  ({sf_15[-1]*100:.1f}% at turn {TURNS[-1]:,})")
ax.fill_between(TURNS, sf_02, sf_15,
                where=sf_15 > sf_02, alpha=0.15, color="#377eb8",
                label="ε=0.15 advantage")
ax.set_xlabel("Game turn", fontsize=9)
ax.set_ylabel("Fraction of runs with all three species present", fontsize=9)
ax.set_xlim(0, 4980); ax.set_ylim(0, 1.02)
ax.legend(fontsize=8.5, framealpha=0.88)
ax.grid(alpha=0.2)
ax.set_title("Survival fraction over time", fontsize=10, fontweight="bold")

# Right: mean trajectories by outcome, both ε
ax = axes3[1]

for label, m_c, lo_c, hi_c, m_e, lo_e, hi_e, color_eps, n_c, n_e in [
    ("ε=0.02", coex_m02, coex_lo02, coex_hi02,
               ext_m02,  ext_lo02,  ext_hi02,  "#e41a1c", n_coex02, n_ext02),
    ("ε=0.15", coex_m15, coex_lo15, coex_hi15,
               ext_m15,  ext_lo15,  ext_hi15,  "#377eb8", n_coex15, n_ext15),
]:
    # Coexistence outcome
    ax.plot(TURNS, m_c, color=color_eps, lw=2.0,
            label=f"{label} all present (n={n_c})")
    ax.fill_between(TURNS, lo_c, hi_c, color=color_eps, alpha=0.15)
    # Extinction outcome (dashed)
    if n_e > 0:
        ax.plot(TURNS, m_e, color=color_eps, lw=1.4, ls="--",
                label=f"{label} species lost (n={n_e})")
        ax.fill_between(TURNS, lo_e, hi_e, color=color_eps, alpha=0.08)

ax.axhline(1/3, color="0.6", lw=0.8, ls=":", label="1/3 equal share")
ax.set_xlabel("Game turn", fontsize=9)
ax.set_ylabel("Mean occupancy (all 3 colors averaged)", fontsize=9)
ax.set_xlim(0, 4980); ax.set_ylim(0, 0.7)
ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
ax.grid(alpha=0.2)
ax.set_title("Mean occupancy by ε and outcome", fontsize=10, fontweight="bold")

out3 = FIGURES_DIR / "fig_occ_trajectories_dcsbm_eps.pdf"
fig3.savefig(out3, bbox_inches="tight")
fig3.savefig(out3.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"Saved {out3}")
plt.close(fig3)
