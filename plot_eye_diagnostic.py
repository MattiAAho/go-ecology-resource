"""
plot_eye_diagnostic.py — Rough diagnostic: eye / group structure by regime.

Figure 1 — Two-player (fig_eye_diag_2p.png)
  Uses stored black_groups / white_groups (stone group counts).
  Eye counts require graph reconstruction; not available here.
  2×2: rows = winner groups / loser groups
       cols = dominance regime (ci_2p<0.10) / segregation regime (ci_2p>0.85)
  Violin by graph type (ER, SBM, DC-SBM); all k, grid sizes, schedulers pooled.

Figure 2 — Three-player (fig_eye_diag_3p.png)
  Uses eye_records_3p.pkl.
  2×2: rows = winner eye count / loser alive groups
       cols = extinction regime (survivors<3) / coexistence regime (J>=0.85, surv=3)
  Each panel: two violin sets (RPS, AVA) × three graph types.
"""
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis_helpers import ci_2p

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ── colour / style ────────────────────────────────────────────────────────────
GC = {"er": "#e41a1c", "sbm": "#377eb8", "dc_sbm": "#ff7f00"}
GTYPES = ["er", "sbm", "dc_sbm"]
GLABEL = {"er": "ER", "sbm": "SBM", "dc_sbm": "DC-SBM"}

def vplot(ax, datasets, positions, color, width=0.35, label=None):
    """Draw violins from a list of arrays at given positions."""
    for i, (data, pos) in enumerate(zip(datasets, positions)):
        clean = [v for v in data if np.isfinite(v)]
        if len(clean) < 2:
            ax.plot(pos, np.mean(clean) if clean else 0,
                    "o", color=color, ms=5, alpha=0.7)
            continue
        parts = ax.violinplot([clean], positions=[pos], widths=width,
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            # Clip KDE density at y=0 (counts cannot be negative)
            verts = pc.get_paths()[0].vertices
            verts[:, 1] = np.maximum(verts[:, 1], 0.0)
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmedians"].set_color(color)
        parts["cmedians"].set_linewidth(2)
    # invisible line for legend
    if label:
        ax.plot([], [], color=color, lw=6, alpha=0.6, label=label)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Two-player
# ══════════════════════════════════════════════════════════════════════════════
print("Loading two-player data …")
with open("results/two_p_all_v1.pkl", "rb") as fh:
    twop = pickle.load(fh)

irr2p = [r for r in twop
         if not r.get("connectivity_failed")
         and r["_graph_type"] in GTYPES
         and r.get("black_occupancy") is not None]
for r in irr2p:
    r["_ci2p"] = ci_2p(r)
    # identify winner/loser
    if r["black_occupancy"] >= r["white_occupancy"]:
        r["_winner_groups"] = r.get("black_groups", 0)
        r["_loser_groups"]  = r.get("white_groups", 0)
    else:
        r["_winner_groups"] = r.get("white_groups", 0)
        r["_loser_groups"]  = r.get("black_groups", 0)

dom2p = [r for r in irr2p if r["_ci2p"] < 0.10]
seg2p = [r for r in irr2p if r["_ci2p"] > 0.85]
print(f"  dominance n={len(dom2p)}  segregation n={len(seg2p)}")

PANEL_SPECS_2P = [
    # (row, col, field, regime_data, regime_label)
    (0, 0, "_winner_groups", dom2p, "Dominance\n(ci < 0.10)"),
    (0, 1, "_winner_groups", seg2p, "Segregation\n(ci > 0.85)"),
    (1, 0, "_loser_groups",  dom2p, "Dominance\n(ci < 0.10)"),
    (1, 1, "_loser_groups",  seg2p, "Segregation\n(ci > 0.85)"),
]
ROW_LABEL_2P = {0: "Winner stone groups", 1: "Loser stone groups"}

fig1, axes1 = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
fig1.suptitle(
    "Two-player stone group structure by regime  (diagnostic)\n"
    "ER / SBM / DC-SBM — all k, grid sizes, schedulers pooled\n"
    "[Eye counts not available without graph reconstruction]",
    fontsize=10
)

xs_base = np.arange(len(GTYPES)) * 1.2   # space the three graph types

for row, col, field, regime_data, regime_label in PANEL_SPECS_2P:
    ax = axes1[row, col]
    for gi, gtype in enumerate(GTYPES):
        vals = [r[field] for r in regime_data if r["_graph_type"] == gtype]
        vplot(ax, [vals], [xs_base[gi]], GC[gtype], width=0.5, label=GLABEL[gtype])

    ax.set_xticks(xs_base)
    ax.set_xticklabels([GLABEL[g] for g in GTYPES], fontsize=10)
    ax.set_title(regime_label, fontsize=10)
    ax.set_ylim(bottom=-0.2)
    ax.grid(axis="y", alpha=0.3)
    if col == 0:
        ax.set_ylabel(ROW_LABEL_2P[row], fontsize=10)
    if row == 0 and col == 0:
        ax.legend(fontsize=8, loc="upper right")

out1 = FIGURES_DIR / "fig_eye_diag_2p.png"
fig1.savefig(out1, dpi=150)
print(f"Saved {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Three-player
# ══════════════════════════════════════════════════════════════════════════════
print("Loading three-player eye records …")
with open("results/eye_records_3p.pkl", "rb") as fh:
    eye3 = pickle.load(fh)

irr3p = [r for r in eye3 if r["graph_type"] in GTYPES]
ext3p  = [r for r in irr3p if r["survivors"] < 3]
coex3p = [r for r in irr3p if r["survivors"] == 3 and r["jain"] >= 0.85]
print(f"  extinction n={len(ext3p)}  coexistence n={len(coex3p)}")

CONTESTS = ["rps", "all_vs_all"]
CLABEL   = {"rps": "RPS", "all_vs_all": "AVA"}
C_COLOR  = {"rps": "#1f77b4", "all_vs_all": "#ff7f0e"}

PANEL_SPECS_3P = [
    (0, 0, "winner_n_eyes",       ext3p,  "Extinction (surv<3)"),
    (0, 1, "winner_n_eyes",       coex3p, "Coexistence (J≥0.85, surv=3)"),
    (1, 0, "loser_n_alive_groups",ext3p,  "Extinction (surv<3)"),
    (1, 1, "loser_n_alive_groups",coex3p, "Coexistence (J≥0.85, surv=3)"),
]
ROW_LABEL_3P = {0: "Winner eye count", 1: "Loser alive groups"}

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
fig2.suptitle(
    "Three-player eye / group structure by regime  (diagnostic)\n"
    "ER / SBM / DC-SBM — RPS (blue) vs AVA (orange) — Gillespie-like, all k and grid sizes pooled",
    fontsize=10
)

# x positions: for each graph type, two adjacent violins (RPS, AVA)
SPACING   = 0.45   # gap between RPS and AVA within a group
GRP_WIDTH = 1.4    # gap between graph type groups
offsets   = {"rps": -SPACING/2, "all_vs_all": +SPACING/2}

for row, col, field, regime_data, regime_label in PANEL_SPECS_3P:
    ax = axes2[row, col]
    xtick_pos, xtick_lab = [], []

    for gi, gtype in enumerate(GTYPES):
        grp_center = gi * GRP_WIDTH
        for contest in CONTESTS:
            pos = grp_center + offsets[contest]
            vals = [r[field] for r in regime_data
                    if r["graph_type"] == gtype and r["contest"] == contest]
            lab = f"{GLABEL[gtype]}\n{CLABEL[contest]}" if gi == 0 else None
            vplot(ax, [vals], [pos], C_COLOR[contest], width=0.35, label=lab)

        xtick_pos.append(grp_center)
        xtick_lab.append(GLABEL[gtype])

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, fontsize=10)
    ax.set_title(regime_label, fontsize=10)
    ax.set_ylim(bottom=-0.2)
    ax.grid(axis="y", alpha=0.3)
    if col == 0:
        ax.set_ylabel(ROW_LABEL_3P[row], fontsize=10)
    if row == 0 and col == 0:
        ax.legend(fontsize=8, loc="upper right")

out2 = FIGURES_DIR / "fig_eye_diag_3p.png"
fig2.savefig(out2, dpi=150)
print(f"Saved {out2}")
print("Done.")
