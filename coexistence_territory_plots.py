"""
coexistence_territory_plots.py — Genuine coexistence analysis on irregular graphs.

Filters the combined eye-analysis dataset (RPS + AVA) to:
  - graph_type ∈ {er, sbm, dc_sbm}   (irregular topologies only)
  - survivors  ≥ 3                    (all three species alive at game end)
  - jain       ≥ 0.85                 (near-equal coexistence, not nominal survival)

Then produces three figures:

  Plot 1  coexistence_loser_alive_groups.pdf
          Loser's alive stone groups (≥2 eyes) — violin by graph_type × contest.
          Directly tests whether the third-ranked species holds defended territory
          in genuine coexistence runs.

  Plot 2  coexistence_eye_structure_by_rank.pdf
          Eye count (n_eyes) and eye density (eye/stone ratio) for all three
          ranks, by graph type.  Reveals how territorial structure differs across
          ER, SBM, and DC-SBM within the high-Jain coexistence regime.

  Plot 3  coexistence_filter_rate.pdf
          Fraction of all runs per graph_type × contest that pass the filter
          (survivors ≥ 3 AND Jain ≥ 0.85).  Contextualises how common genuine
          coexistence is on each topology.

Loads:  results/eye_records_3p.pkl
Saves:  results/coexistence_*.pdf
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR

# ── constants ─────────────────────────────────────────────────────────────────
IRREGULAR_GRAPHS = ["er", "sbm", "dc_sbm"]
GRAPH_LABELS     = {"er": "ER", "sbm": "SBM", "dc_sbm": "DC-SBM"}
CONTEST_LABELS   = {"rps": "RPS", "all_vs_all": "AVA"}
CONTEST_COLORS   = {"rps": "#1f77b4", "all_vs_all": "#ff7f0e"}  # blue / orange
RANK_LABEL       = {0: "winner", 1: "2nd", 2: "loser"}
RANK_DISPLAY     = {0: "Winner", 1: "2nd", 2: "Loser"}
RANK_COLOR       = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}

JAIN_THRESHOLD    = 0.85
SURVIVORS_MIN     = 3

# ── filter helpers ─────────────────────────────────────────────────────────────

def _passes_filter(r: dict) -> bool:
    return (
        r.get("graph_type") in IRREGULAR_GRAPHS
        and (r.get("survivors") or 0) >= SURVIVORS_MIN
        and not np.isnan(r.get("jain", float("nan")))
        and r.get("jain", 0.0) >= JAIN_THRESHOLD
    )

def _in_irregular(r: dict) -> bool:
    return r.get("graph_type") in IRREGULAR_GRAPHS


def _violin(ax, data: list, position: float, color: str, width: float = 0.35) -> None:
    """Draw a single violin at `position`; silently skip if < 2 points."""
    clean = [v for v in data if not np.isnan(v)]
    if len(clean) < 2:
        # Draw a dot for single-point data
        if clean:
            ax.plot(position, clean[0], "o", color=color, ms=4, alpha=0.7)
        return
    parts = ax.violinplot([clean], positions=[position], widths=width,
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        # Clip KDE density at y=0 (counts cannot be negative)
        verts = pc.get_paths()[0].vertices
        verts[:, 1] = np.maximum(verts[:, 1], 0.0)
        pc.set_facecolor(color)
        pc.set_alpha(0.65)
        pc.set_edgecolor("none")
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.2)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Loser alive groups, violin by graph_type × contest
# ─────────────────────────────────────────────────────────────────────────────

def plot1_loser_alive_groups(records: list[dict], out_path: Path) -> None:
    """
    Rows: 1
    Cols: one per contest (rps / ava)
    x-axis: graph_type (er, sbm, dc_sbm)
    y-axis: loser_n_alive_groups

    Each contest panel shows: for each graph_type, the distribution of alive
    stone groups held by the third-ranked species in high-Jain coexistence runs.
    """
    sub    = [r for r in records if _passes_filter(r)]
    if not sub:
        print("  Plot 1: no records pass the filter — skipping")
        return

    contests = [c for c in ["rps", "all_vs_all"]
                if any(r["contest"] == c for r in sub)]
    n_c = len(contests)

    fig, axes = plt.subplots(1, n_c, figsize=(3.5 * n_c + 0.5, 4.2), sharey=True)
    if n_c == 1:
        axes = [axes]

    field = "loser_n_alive_groups"
    positions = [1, 2, 3]

    for ax, contest in zip(axes, contests):
        csub = [r for r in sub if r["contest"] == contest]
        color = CONTEST_COLORS[contest]

        for pos, gt in zip(positions, IRREGULAR_GRAPHS):
            vals = [r[field] for r in csub
                    if r["graph_type"] == gt
                    and not np.isnan(r.get(field, float("nan")))]
            _violin(ax, vals, pos, color)

            # Overlay individual points (jittered)
            if vals:
                jitter = np.random.default_rng(42 + pos).uniform(-0.12, 0.12, len(vals))
                ax.scatter(np.full(len(vals), pos) + jitter, vals,
                           s=6, alpha=0.25, color=color, zorder=2)

        ax.set_xticks(positions)
        ax.set_xticklabels([GRAPH_LABELS[g] for g in IRREGULAR_GRAPHS], fontsize=9)
        ax.set_title(CONTEST_LABELS[contest], fontsize=10)
        ax.grid(axis="y", alpha=0.3, lw=0.6)
        ax.set_xlabel("Graph type", fontsize=9)
        n = len(csub)
        ax.text(0.97, 0.97, f"n = {n}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="#555")

    axes[0].set_ylabel("Loser alive groups (≥ 2 eyes)", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Eye count + eye density for all three ranks, by graph type
# ─────────────────────────────────────────────────────────────────────────────

def plot2_eye_structure_by_rank(records: list[dict], out_path: Path) -> None:
    """
    2 rows × 3 cols.
    Rows: metric  (n_eyes  |  eye_stone_ratio)
    Cols: rank    (winner  |  2nd  |  loser)

    Within each cell: 6 violins — {er, sbm, dc_sbm} × {rps (blue), ava (orange)}.
    x positions: 1,2 (er rps/ava), 3.5,4.5 (sbm), 6,7 (dc_sbm).
    """
    sub = [r for r in records if _passes_filter(r)]
    if not sub:
        print("  Plot 2: no records pass the filter — skipping")
        return

    contests = [c for c in ["rps", "all_vs_all"]
                if any(r["contest"] == c for r in sub)]

    metrics = [
        ("n_eyes",          "Eye count"),
        ("eye_stone_ratio", "Eye / stone ratio"),
    ]
    n_rows = len(metrics)
    n_cols = 3  # ranks

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.8 * n_rows),
                             sharey="row")

    # x-position layout: gaps of 1.5 between graph_type groups
    # within each group: two contests side by side (offset ±0.25)
    group_centers = {gt: 1.0 + i * 2.5 for i, gt in enumerate(IRREGULAR_GRAPHS)}
    contest_offsets = {c: j * 0.55 - 0.275 for j, c in enumerate(contests)}
    violin_width = 0.45

    for row, (metric_suffix, ylabel) in enumerate(metrics):
        for col, (rank, rl) in enumerate(RANK_LABEL.items()):
            ax = axes[row, col]
            field = f"{rl}_{metric_suffix}"

            for gt in IRREGULAR_GRAPHS:
                cx = group_centers[gt]
                for contest in contests:
                    pos = cx + contest_offsets[contest]
                    vals = [r[field] for r in sub
                            if r["graph_type"] == gt
                            and r["contest"] == contest
                            and not np.isnan(r.get(field, float("nan")))]
                    _violin(ax, vals, pos, CONTEST_COLORS[contest], width=violin_width)

            # Vertical separators between graph_type groups
            for i in range(1, len(IRREGULAR_GRAPHS)):
                ax.axvline(group_centers[IRREGULAR_GRAPHS[i]] - 1.25 + contest_offsets[contests[0]],
                           color="gray", lw=0.5, alpha=0.4, ls="--")

            # x tick at group center, labelled with graph_type
            ax.set_xticks([group_centers[gt] + (max(contest_offsets.values()) + min(contest_offsets.values())) / 2
                           for gt in IRREGULAR_GRAPHS])
            ax.set_xticklabels([GRAPH_LABELS[g] for g in IRREGULAR_GRAPHS], fontsize=8)

            ax.grid(axis="y", alpha=0.3, lw=0.6)
            ax.set_title(f"{RANK_DISPLAY[rank]}", fontsize=9)

        axes[row, 0].set_ylabel(ylabel, fontsize=9)

    # Row labels on the right
    for row, (_, ylabel) in enumerate(metrics):
        axes[row, -1].yaxis.set_label_position("right")
        axes[row, -1].set_ylabel(ylabel, fontsize=8, rotation=270, labelpad=12)

    # Legend
    legend_patches = [
        mpatches.Patch(color=CONTEST_COLORS[c], alpha=0.7, label=CONTEST_LABELS[c])
        for c in contests
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(contests), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Eye structure by rank — genuine coexistence runs\n"
        f"(irregular graphs, survivors ≥ {SURVIVORS_MIN}, Jain ≥ {JAIN_THRESHOLD})",
        fontsize=10, y=1.01,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Filter pass rate per graph_type × contest
# ─────────────────────────────────────────────────────────────────────────────

def plot3_filter_rate(records: list[dict], out_path: Path) -> None:
    """
    Grouped bar chart: fraction of runs per graph_type × contest that pass
    survivors ≥ 3 AND Jain ≥ 0.85.

    Denominator = all runs in the eye_records for that graph_type × contest
    (i.e., completed runs with valid final state — connectivity failures excluded
    upstream).

    The y-axis is the genuine-coexistence rate.  Each group on the x-axis is a
    graph type; bars within each group are contest type.
    """
    # Only irregular graphs
    irr = [r for r in records if _in_irregular(r)]
    if not irr:
        print("  Plot 3: no irregular-graph records — skipping")
        return

    contests = [c for c in ["rps", "all_vs_all"]
                if any(r["contest"] == c for r in irr)]

    # Count total and passing per (graph_type, contest)
    from collections import defaultdict
    totals   = defaultdict(int)
    passing  = defaultdict(int)
    for r in irr:
        key = (r["graph_type"], r["contest"])
        totals[key]  += 1
        if (r.get("survivors") or 0) >= SURVIVORS_MIN and \
           not np.isnan(r.get("jain", float("nan"))) and \
           r.get("jain", 0.0) >= JAIN_THRESHOLD:
            passing[key] += 1

    # Build arrays
    rates:  dict[tuple, float] = {}
    errors: dict[tuple, float] = {}  # Wilson interval half-width
    for gt in IRREGULAR_GRAPHS:
        for contest in contests:
            key = (gt, contest)
            n = totals[key]
            k = passing[key]
            if n == 0:
                rates[key]  = float("nan")
                errors[key] = 0.0
            else:
                p = k / n
                rates[key]  = p
                # Wilson score 95% CI half-width
                z = 1.96
                denom = 1 + z**2 / n
                centre = (p + z**2 / (2 * n)) / denom
                half   = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
                errors[key] = half

    n_gt = len(IRREGULAR_GRAPHS)
    n_c  = len(contests)
    bar_w = 0.32
    group_gap = 1.0
    x_centers = np.arange(n_gt) * (n_c * bar_w + group_gap)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for j, contest in enumerate(contests):
        xs    = x_centers + j * bar_w - (n_c - 1) * bar_w / 2
        ys    = [rates.get((gt, contest), float("nan")) for gt in IRREGULAR_GRAPHS]
        yerrs = [errors.get((gt, contest), 0.0) for gt in IRREGULAR_GRAPHS]
        # Replace nan with 0 for bar, keep nan markers invisible
        ys_plot = [y if not np.isnan(y) else 0.0 for y in ys]

        bars = ax.bar(xs, ys_plot, width=bar_w,
                      color=CONTEST_COLORS[contest], alpha=0.8,
                      label=CONTEST_LABELS[contest], zorder=2)
        ax.errorbar(xs, ys_plot, yerr=yerrs,
                    fmt="none", ecolor="black", capsize=3, lw=1.0, zorder=3)

        # Annotate with count
        for x_, y_, gt in zip(xs, ys, IRREGULAR_GRAPHS):
            key = (gt, contest)
            n   = totals.get(key, 0)
            k   = passing.get(key, 0)
            if not np.isnan(y_) and n > 0:
                ax.text(x_, y_ + 0.012, f"{k}/{n}",
                        ha="center", va="bottom", fontsize=6.5, color="#333")

    ax.set_xticks(x_centers)
    ax.set_xticklabels([GRAPH_LABELS[g] for g in IRREGULAR_GRAPHS], fontsize=10)
    ax.set_ylabel(f"Fraction passing filter\n(survivors ≥ {SURVIVORS_MIN}, Jain ≥ {JAIN_THRESHOLD})", fontsize=9)
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.25))
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.35, lw=0.6)
    ax.axhline(0, color="black", lw=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")

    # Print a summary table to stdout
    print(f"\n  {'graph':>8}  {'contest':>12}  {'pass':>5}  {'total':>6}  {'rate':>6}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*5}  {'-'*6}  {'-'*6}")
    for gt in IRREGULAR_GRAPHS:
        for contest in contests:
            key = (gt, contest)
            n = totals.get(key, 0)
            k = passing.get(key, 0)
            r = rates.get(key, float("nan"))
            print(f"  {gt:>8}  {contest:>12}  {k:>5}  {n:>6}  {r:>6.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Genuine coexistence territory analysis — three plots"
    )
    parser.add_argument(
        "--records",
        default=str(RESULTS_DIR / "eye_records_3p.pkl"),
        help="Path to eye_records_3p.pkl produced by eye_analysis_3p.py",
    )
    parser.add_argument(
        "--out", default=str(FIGURES_DIR),
        help="Output directory for PDF figures",
    )
    args = parser.parse_args()

    rec_path = Path(args.records)
    if not rec_path.exists():
        print(f"ERROR: records file not found: {rec_path}")
        print("Run eye_analysis_3p.py first to generate eye_records_3p.pkl")
        sys.exit(1)

    print(f"Loading eye records from {rec_path.name} …", flush=True)
    with open(rec_path, "rb") as fh:
        records: list[dict] = pickle.load(fh)
    print(f"  {len(records)} total records loaded.")

    # Summarise the filter
    irr = [r for r in records if _in_irregular(r)]
    passing = [r for r in irr if _passes_filter(r)]
    print(f"\n  Irregular-graph records : {len(irr)}")
    print(f"  Passing filter           : {len(passing)}")
    print(f"    (graph ∈ {{er,sbm,dc_sbm}}, survivors≥{SURVIVORS_MIN}, Jain≥{JAIN_THRESHOLD})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots …", flush=True)

    plot1_loser_alive_groups(
        records,
        out_dir / "coexistence_loser_alive_groups.pdf",
    )
    plot2_eye_structure_by_rank(
        records,
        out_dir / "coexistence_eye_structure_by_rank.pdf",
    )
    plot3_filter_rate(
        records,
        out_dir / "coexistence_filter_rate.pdf",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
