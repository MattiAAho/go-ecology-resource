"""Model schematic (candidate Figure 1) for manuscript-1.

Two bands:
  TOP  — the per-turn process pipeline: graph -> place stone -> form groups ->
         resource gate -> capture / outcome, with a "repeat each turn" loop.
  BOTTOM — the swept design space ("knobs"): topology family (5 thumbnails,
         regular->heterogeneous), competition structure (2P / RPS / AVA),
         scheduler (sync / async), and the scalar sweep parameters.
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, GREY, GREEN, RED, INK = "#1f77b4", "#666666", "#2ca02c", "#d62728", "#222222"
ORANGE, PURPLE = "#ff7f0e", "#9467bd"
PANEL = "#f4f6f8"

fig = plt.figure(figsize=(13, 7))
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def box(x, y, w, h, text, fc="white", ec=INK, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1.4, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=4, color=INK)


def arrow(x0, y0, x1, y1, color=INK, lw=1.8, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=16,
                                 lw=lw, color=color, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))


# ───────────────────────── TOP: process pipeline ─────────────────────────
ax.text(0.5, 0.965, "A  Per-turn process", ha="center", fontsize=12, fontweight="bold", color=INK)
# two-level flow: middle row = main spine (place → form groups → outcome);
# lower row = the capture branch (capture attempt → resource gate → capture).
y_mid, mh = 0.875, 0.072
y_low, lh = 0.745, 0.066
y_loop = 0.672


def lbox(cx, cy, w, h, text, fc="white", fs=8.5):
    box(cx - w / 2, cy - h / 2, w, h, text, fc=fc, fs=fs)


def connect(pts, color=GREY, lw=1.6):
    """Angular (elbow) connector through waypoints; arrowhead on the final segment."""
    for a_, b_ in zip(pts[:-2], pts[1:-1]):
        ax.plot([a_[0], b_[0]], [a_[1], b_[1]], color=color, lw=lw, zorder=2, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=15,
                                 lw=lw, color=color, zorder=2, connectionstyle="arc3,rad=0"))


cP, wP = 0.1375, 0.185      # place or pass
cF, wF = 0.3750, 0.150      # form groups
cO, wO = 0.8950, 0.140      # outcome
cA, wA = 0.3750, 0.140      # capture attempt
cG, wG = 0.5450, 0.150      # resource gate
cC, wC = 0.7075, 0.115      # capture
xJ = 0.778                  # branch point (turn end)

lbox(cP, y_mid, wP, mh, "Place stone or pass\n(reactive heuristic; ε random)")
lbox(cF, y_mid, wF, mh, "Form groups\n(stones + liberties)")
lbox(cO, y_mid, wO, mh, "Outcome\ncoexistence / dominance", fc="#f0eef7")
lbox(cA, y_low, wA, lh, "Capture attempt\n(last liberty removed)", fs=8)
lbox(cG, y_low, wG, lh, "Resource gate\n$p=S_A/(S_A{+}\\delta S_D)$", fc="#eaf2fb", fs=8)
lbox(cC, y_low, wC, lh, "Capture\n(remove dead group)", fc="#eafaef", fs=8)

# main spine (middle): place -> form groups, then the no-capture line out to the branch point
arrow(cP + wP / 2, y_mid, cF - wF / 2, y_mid)
ax.plot([cF + wF / 2, xJ], [y_mid, y_mid], color=INK, lw=1.7, zorder=2, solid_capstyle="round")
ax.text((cF + wF / 2 + xJ) / 2, y_mid + 0.030, "no capture", ha="center", va="bottom",
        fontsize=7.5, color=GREY, style="italic")
ax.scatter([xJ], [y_mid], s=20, color=INK, zorder=4)
arrow(xJ, y_mid, cO - wO / 2, y_mid)
ax.text((xJ + cO - wO / 2) / 2, y_mid + 0.046, "after 5,000 turns", ha="center", va="bottom",
        fontsize=6.8, color=GREY, style="italic")

# capture branch (lower): form groups down -> chain -> back up onto the main line
arrow(cF, y_mid - mh / 2, cA, y_low + lh / 2)
ax.text(cF + 0.015, (y_mid - mh / 2 + y_low + lh / 2) / 2, "capture", ha="left", va="center",
        fontsize=7.5, color=GREY, style="italic")
arrow(cA + wA / 2, y_low, cG - wG / 2, y_low)
arrow(cG + wG / 2, y_low, cC - wC / 2, y_low)
arrow(cC, y_low + lh / 2, cC, y_mid)

# repeat-each-turn loop, carrying the scheduler that orders the moves
connect([(xJ, y_mid), (xJ, y_loop), (cP, y_loop), (cP, y_mid - mh / 2)])
ax.text((cP + xJ) / 2, y_loop - 0.018, "repeat each turn  (up to 5,000 turns)",
        ha="center", va="top", fontsize=8, color=GREY, style="italic")
ax.text((cP + xJ) / 2, y_loop - 0.040,
        "scheduler orders the moves:  synchronous (round-robin)  /  energy-weighted asynchronous (Gillespie-like)",
        ha="center", va="top", fontsize=7.2, color=GREY, style="italic")


# ──────────────────── BOTTOM: swept design space panel ────────────────────
ax.add_patch(FancyBboxPatch((0.03, 0.045), 0.94, 0.555, boxstyle="round,pad=0.005,rounding_size=0.02",
                            fc=PANEL, ec=GREY, lw=1.2, zorder=1))
ax.text(0.05, 0.565, "B  Swept design space", ha="left", fontsize=12, fontweight="bold", color=INK)

ax.text(0.05, 0.50, "Topology  (regular → heterogeneous)", ha="left", fontsize=10,
        fontweight="bold", color=INK)
ax.annotate("", xy=(0.93, 0.485), xytext=(0.06, 0.485),
            arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2))


# --- thumbnail drawers ---
def thumb(rect, G, pos, node_color=BLUE, sizes=34, ew=0.6):
    a = fig.add_axes(rect); a.axis("off")
    nx.draw_networkx_edges(G, pos, ax=a, edge_color=GREY, alpha=0.45, width=ew)
    nx.draw_networkx_nodes(G, pos, ax=a, node_color=node_color, node_size=sizes,
                           linewidths=0, edgecolors="none")
    xs_ = [p[0] for p in pos.values()]; ys_ = [p[1] for p in pos.values()]
    pad = 0.12 * (max(xs_) - min(xs_) + 1e-9)
    a.set_xlim(min(xs_) - pad, max(xs_) + pad)
    a.set_ylim(min(ys_) - pad, max(ys_) + pad)


def thumb_torus(rect, nu=16, nv=8, R=1.0, r=0.42, tilt_x=0.78, tilt_y=0.42):
    """Genuine 3-D donut: torus surface under two rotations — tilt_x sets the vertical
    foreshortening (smaller = more upright/rounder), tilt_y turns it about the vertical axis
    for an oblique 3/4 view. Tube mesh + nodes shaded by depth (near solid+large, far faint+small)."""
    a = fig.add_axes(rect); a.axis("off"); a.set_aspect("equal")
    cx_, sx_ = np.cos(tilt_x), np.sin(tilt_x)
    cy_, sy_ = np.cos(tilt_y), np.sin(tilt_y)

    def proj(u, v):                                         # (screen_x, screen_y, depth)
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        y1, z1 = y * cx_ - z * sx_, y * sx_ + z * cx_       # rotate about x-axis (foreshorten)
        return x * cy_ + z1 * sy_, y1, -x * sy_ + z1 * cy_  # then about y-axis (turn), drop depth

    us = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    vs = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    pts = {(i, j): proj(u, v) for i, u in enumerate(us) for j, v in enumerate(vs)}
    zd = np.array([p[2] for p in pts.values()])
    z0, z1 = zd.min(), zd.max()
    shade = lambda z: (z - z0) / (z1 - z0 + 1e-9)           # 0 = far/back, 1 = near/front

    segs = []                                               # tube mesh: minor + major circles
    for i in range(nu):
        for j in range(nv):
            segs.append((pts[(i, j)], pts[(i, (j + 1) % nv)]))      # around the tube
            segs.append((pts[(i, j)], pts[((i + 1) % nu, j)]))      # around the ring
    for p, q in sorted(segs, key=lambda s: s[0][2] + s[1][2]):      # far first
        d = shade((p[2] + q[2]) / 2)
        a.plot([p[0], q[0]], [p[1], q[1]], color=GREY, lw=0.55, alpha=0.18 + 0.55 * d, zorder=1)

    order = sorted(pts.values(), key=lambda p: p[2])               # far nodes first
    base = np.array(mcolors.to_rgba(BLUE))
    cols = np.tile(base, (len(order), 1))
    cols[:, 3] = [0.45 + 0.55 * shade(p[2]) for p in order]
    a.scatter([p[0] for p in order], [p[1] for p in order],
              s=[5 + 9 * shade(p[2]) for p in order], c=cols, zorder=2)
    ax_, ay_ = [p[0] for p in pts.values()], [p[1] for p in pts.values()]
    m = 1.08 * max(max(ax_) - min(ax_), max(ay_) - min(ay_)) / 2
    a.set_xlim((min(ax_) + max(ax_)) / 2 - m, (min(ax_) + max(ax_)) / 2 + m)
    a.set_ylim((min(ay_) + max(ay_)) / 2 - m, (min(ay_) + max(ay_)) / 2 + m)


def layout_blocks(G, sizes, seed):
    """Lay out each block independently and translate to a triangle vertex → 3 clear clumps."""
    centers = [(0.0, 1.35), (-1.25, -0.8), (1.25, -0.8)]
    pos = {}; start = 0
    for b, n in enumerate(sizes):
        nodes = list(range(start, start + n))
        sp = nx.spring_layout(G.subgraph(nodes), seed=seed + b, k=0.7, iterations=120)
        cx, cy = centers[b]
        for nd in nodes:
            pos[nd] = (sp[nd][0] * 0.85 + cx, sp[nd][1] * 0.85 + cy)
        start += n
    return pos


rng = 7
Gl = nx.grid_2d_graph(4, 4); posl = {n: n for n in Gl}                    # lattice
Ge = nx.gnp_random_graph(24, 0.12, seed=rng); pose = nx.spring_layout(Ge, seed=rng, k=0.9)  # ER

# SBM — 3 ER-like blocks (roughly uniform degree, no hubs)
sizes = [8, 8, 8]
Gs = nx.stochastic_block_model(sizes, [[0.5, 0.02, 0.02], [0.02, 0.5, 0.02], [0.02, 0.02, 0.5]], seed=rng)
poss = layout_blocks(Gs, sizes, rng)
cols3 = [BLUE] * 8 + [ORANGE] * 8 + [GREEN] * 8

# DC-SBM — 3 degree-corrected blocks. Each block has a DEGREE GRADIENT (a couple of
# semi-hubs + lower-degree nodes, not a single star), and inter-block edges land on a
# MIX of nodes (hubs and leaves alike), degree-weighted but not hub-to-hub only.
Gd = nx.Graph(); Gd.add_nodes_from(range(24))
intra = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),       # node 0 = primary semi-hub
         (1, 2), (1, 6), (1, 7), (3, 6), (4, 5)]        # node 1 = secondary semi-hub
for b in range(3):
    base = b * 8
    for u, v in intra:
        Gd.add_edge(base + u, base + v)
inter = [(0, 8), (5, 9), (8, 16), (13, 18), (10, 17), (1, 16), (6, 20)]  # hubs + leaves
for u, v in inter:
    Gd.add_edge(u, v)
posd = layout_blocks(Gd, sizes, rng + 3)
deg = np.array([Gd.degree(n) for n in range(24)])
sizes_d = 12 + 7 * deg

labels = ["lattice", "torus", "ER", "SBM (3 blocks)", "DC-SBM (3 blocks)"]
tw, ty = 0.15, 0.345
txs = [0.065 + i * 0.176 for i in range(5)]
thumb([txs[0], ty, tw, 0.13], Gl, posl)
thumb_torus([txs[1], ty, tw, 0.13])
thumb([txs[2], ty, tw, 0.13], Ge, pose, sizes=28)
thumb([txs[3], ty, tw, 0.13], Gs, poss, node_color=cols3, sizes=30)
thumb([txs[4], ty, tw, 0.13], Gd, posd, node_color=cols3, sizes=sizes_d)
for x, lab in zip(txs, labels):
    ax.text(x + tw / 2, ty - 0.02, lab, ha="center", fontsize=8.5, color=INK)

# --- competition structure (left) and scheduler (right) ---
ax.text(0.05, 0.275, "Competition structure", ha="left", fontsize=10, fontweight="bold", color=INK)
for x, s in zip([0.06, 0.18, 0.30], ["2-player\ndirect", "3-player\nRPS (cyclic)", "3-player\nAVA (symmetric)"]):
    box(x, 0.175, 0.11, 0.07, s, fc="white", fs=8)

ax.text(0.52, 0.275, "Scheduler", ha="left", fontsize=10, fontweight="bold", color=INK)
for x, s in zip([0.52, 0.70], ["synchronous\n(round-robin)", "energy-weighted asynchronous\n(Gillespie-like)"]):
    box(x, 0.175, 0.155, 0.07, s, fc="white", fs=8)

# --- scalar sweep parameters ---
ax.text(0.05, 0.115, "Other swept parameters:", ha="left", fontsize=10, fontweight="bold", color=INK)
ax.text(0.27, 0.115,
        "mean degree k ∈ {8, 10, 12, 14, 19}   ·   board size n ∈ {81, 225, 361, 625}   ·   exploration ε = 0.15",
        ha="left", fontsize=9.5, color=INK)
ax.text(0.05, 0.075,
        "ε fixed at 0.15 in production, also swept {0.02, 0.04, 0.07, 0.15} for sensitivity "
        "on a reduced grid (n ∈ {225, 625}, k ∈ {10, 14})   ·   ≈ 50,000 simulation runs in total",
        ha="left", fontsize=8.7, color=GREY, style="italic")

fig.savefig("figures/final/fig_schematic.png", dpi=150, bbox_inches="tight")
try:
    fig.savefig("figures/final/fig_schematic.pdf", bbox_inches="tight")
    print("saved figures/final/fig_schematic.pdf / .png")
except PermissionError:
    fig.savefig("/tmp/fig_schematic.pdf", bbox_inches="tight")
    print("PDF locked (open in a viewer?); wrote PNG + /tmp/fig_schematic.pdf")
print("DC-SBM degrees:", sorted(deg, reverse=True))
plt.close(fig)
