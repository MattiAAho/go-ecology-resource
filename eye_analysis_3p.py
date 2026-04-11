"""
eye_analysis_3p.py — Eye analysis extended to three-player RPS and all-vs-all sweeps.

The two-player `analyse_eyes` in analysis_helpers.py calls make_game_from_graph with
default colors=(+1,-1) and no prey_of.  is_true_eye_like iterates gs.colors for the
opponent legality check, so the +2 species is invisible and eye detection is incorrect
for three-player games.  This script fixes that and produces analogous plots with
winner / 2nd / loser ranked by n_stones.

Loads:
    results/full_all_v2.pkl   — RPS sweep  (rps contest)
    results/ava_all_v1.pkl    — AVA sweep  (all_vs_all contest)

Saves plots to results/eye_3p_*.png
"""

from __future__ import annotations

import itertools
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from run_one_game import make_game_from_graph
from go_engine import is_true_eye_like, groups_and_liberties
from rps_sweep import (
    FULL_PARAMS, DCSBM_PARAMS, FIXED_TOPO_PARAMS,
    AVA_PARAMS, AVA_DCSBM_PARAMS, AVA_FIXED_TOPO_PARAMS,
    GRAPH_MAKERS, PREY_OF, COLORS,
)

RESULTS_DIR = Path(__file__).parent / "results"
SEED_BASE   = 42
ALL_COLORS  = (+1, -1, +2)
COLOR_LABEL = {+1: "black", -1: "white", +2: "red"}
RANK_LABEL  = {0: "winner", 1: "2nd", 2: "loser"}
RANK_COLOR  = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}   # green / orange / red


# ─────────────────────────────────────────────────────────────────────────────
# 3-player eye analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_eyes_3p(
    G: nx.Graph,
    state_dict: dict,
    color: int,
    prey_of: dict | None,
    all_colors: tuple = ALL_COLORS,
) -> dict:
    """Analyse eye-like structures for `color` in a three-player board state.

    Fixes the two-player limitation: passes correct colors=(+1,-1,+2) and
    prey_of to make_game_from_graph so that is_true_eye_like checks all three
    opponents for the legality test.

    Parameters
    ----------
    G          : the graph (NetworkX)
    state_dict : node → color int (0=empty, +1/-1/+2 = occupied)
    color      : the species to analyse
    prey_of    : predation dict for this contest, or None (all-vs-all)
    all_colors : tuple of all species colors in the game
    """
    gs = make_game_from_graph(G, state=state_dict, colors=all_colors, prey_of=prey_of)

    eye_nodes   = [v for v in G.nodes() if is_true_eye_like(gs, v, color)]
    eye_set     = set(eye_nodes)
    eye_subgraph = G.subgraph(eye_set)
    components  = list(nx.connected_components(eye_subgraph))
    comp_sizes  = sorted([len(c) for c in components], reverse=True)

    stone_groups = groups_and_liberties(gs, color)
    eyes_per_group: list[int] = []
    n_alive_groups = 0
    for group, _ in stone_groups:
        # eye nodes adjacent to this stone group
        group_eyes = {v for v in eye_set if any(u in group for u in gs.neigh[v])}
        local_eye_graph = G.subgraph(group_eyes)
        n_local_comps   = nx.number_connected_components(local_eye_graph)
        eyes_per_group.append(n_local_comps)
        if n_local_comps >= 2:
            n_alive_groups += 1

    n_stones = sum(1 for v in G.nodes() if state_dict.get(v, 0) == color)

    return {
        "color":                color,
        "label":                COLOR_LABEL[color],
        "n_eyes":               len(eye_nodes),
        "n_stones":             n_stones,
        "eye_stone_ratio":      len(eye_nodes) / n_stones if n_stones else 0.0,
        "n_eye_components":     len(components),
        "max_eye_component":    comp_sizes[0] if comp_sizes else 0,
        "n_stone_groups":       len(stone_groups),
        "n_alive_groups":       n_alive_groups,
        "alive_group_fraction": n_alive_groups / len(stone_groups) if stone_groups else 0.0,
        "mean_eyes_per_group":  sum(eyes_per_group) / len(eyes_per_group) if eyes_per_group else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graph reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _build_seed_map(params_list: list[dict], seed_base: int = SEED_BASE) -> dict:
    """Map (contest, graph_type, grid_size, scheduler, policy, target_k) → cond_seed.

    Each params dict was used as an independent run_sweep call with the same
    seed_base=42.  Condition index i gives cond_seed = seed_base + (i+1)*10_000.
    Graph types are mutually exclusive across sub-sweeps so no key collisions occur.
    """
    seed_map: dict[tuple, int] = {}
    for params in params_list:
        conditions = list(itertools.product(
            params["contests"],
            params["graph_types"],
            params["grid_sizes"],
            params["schedulers"],
            params["policies"],
            params["k_values"],
        ))
        for i, cond in enumerate(conditions):
            cond_seed = seed_base + (i + 1) * 10_000
            seed_map.setdefault(cond, cond_seed)
    return seed_map


# Build once at module level — used by reconstruct_graph
_SEED_MAP: dict[tuple, int] = _build_seed_map([
    FULL_PARAMS, DCSBM_PARAMS, FIXED_TOPO_PARAMS,
    AVA_PARAMS, AVA_DCSBM_PARAMS, AVA_FIXED_TOPO_PARAMS,
])


def reconstruct_graph(r: dict) -> nx.Graph | None:
    """Reconstruct the graph used for result `r` using the same seed formula as run_sweep."""
    key = (
        r["_contest"],
        r["_graph_type"],
        r["_grid_size"],
        r["_scheduler"],
        r["_policy"],
        r["_target_k"],
    )
    cond_seed = _SEED_MAP.get(key)
    if cond_seed is None:
        print(f"  WARNING: no seed map entry for key {key}", flush=True)
        return None
    graph_seed = cond_seed + r["_rep"] * 1000
    maker = GRAPH_MAKERS[r["_graph_type"]]
    return maker(r["_grid_size"], r["_target_k"], seed=graph_seed)


# ─────────────────────────────────────────────────────────────────────────────
# Main collection loop
# ─────────────────────────────────────────────────────────────────────────────

def collect_eye_records(all_results: list[dict]) -> list[dict]:
    """Iterate results, reconstruct graphs, run eye analysis for all three species.

    For each valid result:
    - Rank the three colors by n_stones (desc) → rank 0=winner, 1=2nd, 2=loser
    - Run analyse_eyes_3p for each color
    - Store one record per run with fields for all three ranks
    """
    valid = [r for r in all_results if not r.get("connectivity_failed")]
    n_total = len(valid)
    print(f"  Analysing {n_total} valid runs …", flush=True)

    records: list[dict] = []
    skipped = 0
    graph_cache: dict[tuple, nx.Graph | None] = {}   # (graph_type, grid_size, seed) → G

    for idx, r in enumerate(valid):
        if (idx + 1) % max(1, n_total // 20) == 0:
            print(f"  {idx + 1}/{n_total}", flush=True)

        final_state = r.get("final_state")
        if final_state is None:
            skipped += 1
            continue

        n_stones_dict: dict[int, int] = r.get("n_stones", {})
        if not n_stones_dict:
            # Fall back to counting from final_state
            n_stones_dict = {c: sum(1 for v in final_state if final_state[v] == c)
                             for c in ALL_COLORS}

        # Rank colors by stone count (descending); ties broken by color value for reproducibility
        ranked_colors = sorted(ALL_COLORS, key=lambda c: (-n_stones_dict.get(c, 0), c))

        # Reconstruct graph (cache by unique graph seed key)
        cache_key = (r["_graph_type"], r["_grid_size"], r["_target_k"],
                     r["_contest"], r["_scheduler"], r["_policy"], r["_rep"])
        if cache_key not in graph_cache:
            graph_cache[cache_key] = reconstruct_graph(r)
        G = graph_cache[cache_key]

        if G is None:
            skipped += 1
            continue

        contest  = r["_contest"]
        prey_of  = PREY_OF.get(contest)   # None for all_vs_all; dict for rps

        # Run eye analysis for each rank
        eye_by_rank: dict[int, dict] = {}
        for rank, color in enumerate(ranked_colors):
            eye_by_rank[rank] = analyse_eyes_3p(G, final_state, color, prey_of)

        rec = dict(
            contest      = contest,
            graph_type   = r["_graph_type"],
            grid_size    = r["_grid_size"],
            n_nodes      = r["_grid_size"] ** 2,
            scheduler    = r["_scheduler"],
            policy       = r["_policy"],
            target_k     = r["_target_k"],
            rep          = r["_rep"],
            # run-level outcome metrics
            jain         = r.get("jain_fairness_final",  np.nan),
            survivors    = r.get("survivors_count",       np.nan),
            largest_share= r.get("largest_share",         np.nan),
            moves        = r.get("moves",                 np.nan),
        )
        # Add per-rank eye fields
        for rank in range(3):
            e = eye_by_rank[rank]
            rl = RANK_LABEL[rank]
            rec[f"{rl}_color"]           = e["label"]
            rec[f"{rl}_n_stones"]        = e["n_stones"]
            rec[f"{rl}_n_eyes"]          = e["n_eyes"]
            rec[f"{rl}_eye_stone_ratio"] = e["eye_stone_ratio"]
            rec[f"{rl}_n_alive_groups"]  = e["n_alive_groups"]
            rec[f"{rl}_alive_frac"]      = e["alive_group_fraction"]
            rec[f"{rl}_mean_eyes_per_group"] = e["mean_eyes_per_group"]
            rec[f"{rl}_n_eye_components"]    = e["n_eye_components"]
            rec[f"{rl}_max_eye_component"]   = e["max_eye_component"]

        records.append(rec)

    if skipped:
        print(f"  Skipped {skipped} runs (missing final_state or graph reconstruction failed)")
    print(f"  Collected {len(records)} records.")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _violin_by_rank(
    records: list[dict],
    field: str,
    ylabel: str,
    title: str,
    out_path: Path,
    contests: list[str] = ("rps", "all_vs_all"),
    graph_types: list[str] | None = None,
) -> None:
    """Violin plot: field by rank (winner/2nd/loser), one column per contest."""
    if graph_types is None:
        graph_types = sorted({r["graph_type"] for r in records})
    n_contests = len(contests)

    fig, axes = plt.subplots(1, n_contests, figsize=(6 * n_contests, 5), sharey=True)
    if n_contests == 1:
        axes = [axes]

    positions = [1, 2, 3]

    for ax, contest in zip(axes, contests):
        sub = [r for r in records if r["contest"] == contest]
        if not sub:
            ax.set_title(contest)
            continue

        data_by_rank = []
        for rank, rl in RANK_LABEL.items():
            vals = [r[f"{rl}_{field}"] for r in sub if not np.isnan(r.get(f"{rl}_{field}", np.nan))]
            data_by_rank.append(vals)

        parts = ax.violinplot(data_by_rank, positions=positions,
                              showmedians=True, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(RANK_COLOR[i])
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")

        ax.set_xticks(positions)
        ax.set_xticklabels([RANK_LABEL[r] for r in range(3)])
        ax.set_xlabel("Rank by stone count")
        ax.set_title(contest.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def _violin_by_rank_graphtype(
    records: list[dict],
    field: str,
    ylabel: str,
    title: str,
    out_path: Path,
    contests: list[str] = ("rps", "all_vs_all"),
) -> None:
    """Grid of violin plots: rows=contest, columns=rank, violins=graph_type."""
    graph_types = sorted({r["graph_type"] for r in records})
    n_gt = len(graph_types)
    gt_colors = plt.cm.tab10(np.linspace(0, 0.9, n_gt))

    n_rows = len(contests)
    n_cols = 3   # one per rank
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             sharey=True, sharex=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, contest in enumerate(contests):
        for col, (rank, rl) in enumerate(RANK_LABEL.items()):
            ax = axes[row, col]
            sub = [r for r in records if r["contest"] == contest]

            data = []
            labels = []
            colors = []
            for i_gt, gt in enumerate(graph_types):
                vals = [r[f"{rl}_{field}"] for r in sub
                        if r["graph_type"] == gt
                        and not np.isnan(r.get(f"{rl}_{field}", np.nan))]
                if vals:
                    data.append(vals)
                    labels.append(gt)
                    colors.append(gt_colors[i_gt])

            if not data:
                continue

            positions = list(range(1, len(data) + 1))
            parts = ax.violinplot(data, positions=positions,
                                  showmedians=True, showextrema=False)
            for pc, col_c in zip(parts["bodies"], colors):
                pc.set_facecolor(col_c)
                pc.set_alpha(0.7)
            parts["cmedians"].set_color("black")

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_title(f"{contest.replace('_',' ')} — {rl}", fontsize=9)
            ax.grid(axis="y", alpha=0.3)

    axes[-1, 0].set_xlabel("Graph type")
    for row in range(n_rows):
        axes[row, 0].set_ylabel(ylabel)

    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def _phase_space_plot(
    records: list[dict],
    out_path: Path,
    contests: list[str] = ("rps", "all_vs_all"),
) -> None:
    """Scatter: winner eye_stone_ratio vs Jain fairness, coloured by survivors."""
    n_contests = len(contests)
    fig, axes = plt.subplots(1, n_contests, figsize=(6 * n_contests, 5))
    if n_contests == 1:
        axes = [axes]

    for ax, contest in zip(axes, contests):
        sub = [r for r in records
               if r["contest"] == contest
               and not np.isnan(r.get("winner_eye_stone_ratio", np.nan))
               and not np.isnan(r.get("jain", np.nan))]
        if not sub:
            ax.set_title(contest)
            continue

        x = np.array([r["jain"] for r in sub])
        y = np.array([r["winner_eye_stone_ratio"] for r in sub])
        c = np.array([r["survivors"] for r in sub])

        sc = ax.scatter(x, y, c=c, cmap="RdYlGn", alpha=0.4, s=15,
                        vmin=1, vmax=3)
        plt.colorbar(sc, ax=ax, label="survivors")
        ax.set_xlabel("Jain fairness index")
        ax.set_ylabel("Winner eye/stone ratio")
        ax.set_title(contest.replace("_", " ").title())
        ax.grid(alpha=0.3)

    fig.suptitle("Phase space: Jain fairness vs winner eye density\n"
                 "(colour = number of surviving species)", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def _loser_metrics_plot(
    records: list[dict],
    out_path: Path,
    contests: list[str] = ("rps", "all_vs_all"),
) -> None:
    """2×2 grid: loser metrics (n_stones, n_eyes, alive_frac, mean_eyes_per_group)
    vs. Jain fairness, for each contest side-by-side."""
    metrics = [
        ("loser_n_stones",         "Loser stone count"),
        ("loser_n_eyes",           "Loser eye count"),
        ("loser_alive_frac",       "Loser alive-group fraction"),
        ("loser_mean_eyes_per_group", "Loser mean eyes/group"),
    ]
    n_contests = len(contests)
    n_metrics  = len(metrics)
    fig, axes = plt.subplots(n_metrics, n_contests,
                             figsize=(5 * n_contests, 4 * n_metrics),
                             sharex=False)
    if n_contests == 1:
        axes = axes[:, np.newaxis]

    for row, (field, ylabel) in enumerate(metrics):
        for col, contest in enumerate(contests):
            ax = axes[row, col]
            sub = [r for r in records
                   if r["contest"] == contest
                   and not np.isnan(r.get(field, np.nan))
                   and not np.isnan(r.get("jain", np.nan))]
            if not sub:
                continue
            x = [r["jain"] for r in sub]
            y = [r[field] for r in sub]
            ax.scatter(x, y, alpha=0.25, s=10, color="#d62728")
            ax.set_xlabel("Jain fairness")
            ax.set_ylabel(ylabel)
            ax.set_title(contest.replace("_", " ").title() if row == 0 else "")
            ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def _stone_rank_stability_plot(
    records: list[dict],
    out_path: Path,
    contests: list[str] = ("rps", "all_vs_all"),
) -> None:
    """Bar: fraction of runs where each species color ends up winner/2nd/loser.
    Shows whether the RPS cycle keeps species symmetric or if one color dominates."""
    n_contests = len(contests)
    fig, axes = plt.subplots(1, n_contests, figsize=(6 * n_contests, 4))
    if n_contests == 1:
        axes = [axes]

    for ax, contest in zip(axes, contests):
        sub = [r for r in records if r["contest"] == contest]
        if not sub:
            ax.set_title(contest)
            continue

        # Count (color_label, rank) occurrences
        from collections import Counter
        counts: dict[str, Counter] = defaultdict(Counter)
        for r in sub:
            for rank, rl in RANK_LABEL.items():
                counts[RANK_LABEL[rank]][r[f"{rl}_color"]] += 1

        color_labels = sorted({r[f"{RANK_LABEL[0]}_color"] for r in sub}
                               | {r[f"{RANK_LABEL[1]}_color"] for r in sub}
                               | {r[f"{RANK_LABEL[2]}_color"] for r in sub})
        x  = np.arange(len(color_labels))
        w  = 0.25
        for i, (rank, rl) in enumerate(RANK_LABEL.items()):
            vals = [counts[rl].get(cl, 0) / len(sub) for cl in color_labels]
            ax.bar(x + i * w, vals, width=w, label=rl,
                   color=RANK_COLOR[rank], alpha=0.8)

        ax.set_xticks(x + w)
        ax.set_xticklabels(color_labels)
        ax.set_ylabel("Fraction of runs")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title(contest.replace("_", " ").title())
        ax.axhline(1 / 3, color="k", ls="--", lw=0.8, label="uniform 1/3")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Species rank distribution (symmetry check)", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="3-player eye analysis")
    parser.add_argument("--rps",  default=str(RESULTS_DIR / "full_all_v2.pkl"),
                        help="Path to RPS all-result pkl")
    parser.add_argument("--ava",  default=str(RESULTS_DIR / "ava_all_v1.pkl"),
                        help="Path to AVA all-result pkl")
    parser.add_argument("--out",  default=str(RESULTS_DIR),
                        help="Output directory for PNG files")
    parser.add_argument("--limit", type=int, default=0,
                        help="Analyse only the first N valid results (0 = all; for testing)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load results ──────────────────────────────────────────────────────────
    all_results: list[dict] = []
    for path_str, label in [(args.rps, "RPS"), (args.ava, "AVA")]:
        p = Path(path_str)
        if not p.exists():
            print(f"  WARNING: {p} not found — skipping {label}")
            continue
        print(f"Loading {label} from {p.name} …", flush=True)
        with open(p, "rb") as fh:
            data = pickle.load(fh)
        print(f"  {len(data)} total records.")
        all_results.extend(data)

    if not all_results:
        print("No results loaded.  Exiting.")
        return

    # Optional: subsample for fast testing
    if args.limit > 0:
        valid = [r for r in all_results if not r.get("connectivity_failed")]
        invalid = [r for r in all_results if r.get("connectivity_failed")]
        all_results = valid[:args.limit] + invalid
        print(f"  (limit={args.limit}: using {min(args.limit, len(valid))} valid runs)")

    # ── Collect eye records ───────────────────────────────────────────────────
    print("\nRunning eye analysis …", flush=True)
    records = collect_eye_records(all_results)

    if not records:
        print("No records produced.  Check final_state availability.")
        return

    # Save records as pkl for further notebook use
    rec_path = out_dir / "eye_records_3p.pkl"
    with open(rec_path, "wb") as fh:
        pickle.dump(records, fh, protocol=4)
    print(f"\nSaved {len(records)} eye records → {rec_path.name}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …", flush=True)
    contests = ["rps", "all_vs_all"]
    contests_present = sorted({r["contest"] for r in records})

    _violin_by_rank(
        records, "n_eyes", "Eye count",
        "Eye count by rank (winner / 2nd / loser)",
        out_dir / "eye_3p_n_eyes_by_rank.png",
        contests=contests_present,
    )
    _violin_by_rank(
        records, "eye_stone_ratio", "Eye / stone ratio",
        "Eye density (eyes per stone) by rank",
        out_dir / "eye_3p_eye_stone_ratio_by_rank.png",
        contests=contests_present,
    )
    _violin_by_rank(
        records, "n_alive_groups", "Alive groups (≥2 eyes)",
        "Alive stone groups by rank",
        out_dir / "eye_3p_alive_groups_by_rank.png",
        contests=contests_present,
    )
    _violin_by_rank(
        records, "mean_eyes_per_group", "Mean eye components / group",
        "Mean eyes per stone group by rank",
        out_dir / "eye_3p_mean_eyes_per_group_by_rank.png",
        contests=contests_present,
    )

    _violin_by_rank_graphtype(
        records, "eye_stone_ratio", "Eye / stone ratio",
        "Eye density by rank and graph type",
        out_dir / "eye_3p_eye_stone_by_graphtype.png",
        contests=contests_present,
    )

    _phase_space_plot(
        records, out_dir / "eye_3p_phase_space.png",
        contests=contests_present,
    )
    _loser_metrics_plot(
        records, out_dir / "eye_3p_loser_metrics.png",
        contests=contests_present,
    )
    _stone_rank_stability_plot(
        records, out_dir / "eye_3p_rank_stability.png",
        contests=contests_present,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
