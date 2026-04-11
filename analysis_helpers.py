"""
analysis_helpers.py — stable analysis/plotting helpers for the Go-ecology sweep notebooks.

Import this module from notebooks instead of redefining functions inline.
All functions that existed in multiple notebook versions are resolved here
to their canonical (correct, most recent) form.

Usage in notebook:
    from analysis_helpers import (
        graph_descriptors, print_ecology_table,
        plot_snapshot_pair, plot_plot_snapshot_pairs,
        classify_regime, analyse_eyes, collect_sweep_eye_data,
        rebuild_graphs, get_results_for,
        _pad_to, _nanmean_std,
    )
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from run_one_game import make_game_from_graph
from go_engine import is_true_eye_like, groups_and_liberties


# ─────────────────────────────────────────────
# Graph construction helpers
# (canonical: notebook cell 4 + rps_sweep.py integer-size fix)
# ─────────────────────────────────────────────

def er_p_for_degree(grid_size: int, target_k: float) -> float:
    n = grid_size ** 2
    return target_k / (n - 1)


def sbm_P_for_degree(
    grid_size: int,
    target_k: float,
    modularity_ratio: float = 8.0,
) -> list[list[float]]:
    """Return the 3×3 probability matrix P for nx.stochastic_block_model.

    Block sizes are 3:2:1 (returned separately by sbm_sizes).
    Note: rps_sweep.py uses a version that returns (sizes, P) — this version
    returns only P to match the notebook's rebuild_graphs() interface.
    """
    n = grid_size ** 2
    unit = n / 6
    s = [3 * unit, 2 * unit, unit]
    r = 1.0 / modularity_ratio

    def mean_deg(pw: float) -> float:
        pb = pw * r
        P = [
            [pw,        pb,        pb * 0.5],
            [pb,        pw * 0.75, pb      ],
            [pb * 0.5,  pb,        pw * 0.875],
        ]
        return sum(s[i] * s[j] * P[i][j] for i in range(3) for j in range(3)) / n

    pw = brentq(lambda pw: mean_deg(pw) - target_k, 1e-6, 1.0 - 1e-9)
    pb = pw * r
    return [
        [pw,        pb,        pb * 0.5],
        [pb,        pw * 0.75, pb      ],
        [pb * 0.5,  pb,        pw * 0.875],
    ]


def sbm_sizes(grid_size: int) -> list[int]:
    """Integer block sizes [s1, s2, s3] summing exactly to grid_size²."""
    n = grid_size ** 2
    unit = n / 6
    s1 = round(3 * unit)
    s2 = round(2 * unit)
    s3 = n - s1 - s2
    assert s3 > 0, f"Block s3 non-positive for grid_size={grid_size}"
    return [s1, s2, s3]


_DEFAULT_GRID_SIZES    = [9, 15, 19, 25]
_DEFAULT_TARGET_DEGREES = {"deg08": 8, "deg10": 10, "deg12": 12, "deg14": 14, "deg19": 19}


def rebuild_graphs(
    grid_sizes: list[int] | None = None,
    target_degrees: dict[str, float] | None = None,
    topos: list[str] = ("er", "sbm"),
    seed_range: range = range(1, 20),
) -> dict[int, dict[str, nx.Graph]]:
    """Reconstruct ER and SBM graphs by searching for the first connected instance.

    Returns {grid_size: {tag_full: G}} where tag_full = f"{topo}_{tag}".

    Note: this is the historical reconstruction path. For new sweeps prefer storing
    graph_seed in the result dict and calling make_er_graph/make_sbm_graph directly.
    """
    if grid_sizes is None:
        grid_sizes = _DEFAULT_GRID_SIZES
    if target_degrees is None:
        target_degrees = _DEFAULT_TARGET_DEGREES
    out: dict[int, dict[str, nx.Graph]] = {gs: {} for gs in grid_sizes}
    for gs in grid_sizes:
        n = gs ** 2
        sizes = sbm_sizes(gs)
        for tag, target_k in target_degrees.items():
            if "er" in topos:
                p = er_p_for_degree(gs, target_k)
                for seed in seed_range:
                    G = nx.erdos_renyi_graph(n, p, seed=seed)
                    if nx.is_connected(G):
                        out[gs][f"er_{tag}"] = G
                        break
                else:
                    print(f"  WARNING: no connected ER graph found for gs={gs} {tag}")

            if "sbm" in topos:
                P = sbm_P_for_degree(gs, target_k)
                for seed in seed_range:
                    G = nx.stochastic_block_model(sizes, P, seed=seed)
                    if nx.is_connected(G):
                        out[gs][f"sbm_{tag}"] = G
                        break
                else:
                    print(f"  WARNING: no connected SBM graph found for gs={gs} {tag}")
    return out


def _rebuild_one(
    grid_size: int,
    tag: str,
    target_degrees: dict[str, float] | None = None,
) -> dict[str, nx.Graph]:
    """Rebuild ER and SBM graphs for one (grid_size, tag) condition."""
    if target_degrees is None:
        target_degrees = {"deg08": 8, "deg10": 10, "deg12": 12, "deg14": 14, "deg19": 19}
    target_k = target_degrees[tag]
    n = grid_size ** 2
    sizes = sbm_sizes(grid_size)
    graphs: dict[str, nx.Graph] = {}

    p = er_p_for_degree(grid_size, target_k)
    for seed in range(1, 20):
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        if nx.is_connected(G):
            graphs["er"] = G
            break

    P = sbm_P_for_degree(grid_size, target_k)
    for seed in range(1, 20):
        G = nx.stochastic_block_model(sizes, P, seed=seed)
        if nx.is_connected(G):
            graphs["sbm"] = G
            break

    return graphs


def get_results_for(
    all_results: dict,
    grid_size: int,
    tag: str,
    graphs_sweep: dict | None = None,
) -> tuple[dict, dict]:
    """Extract a (grid_size, tag) slice into {topo: [runs]} and {topo: G} dicts."""
    results: dict[str, list] = {}
    for topo in ("er", "sbm"):
        tag_full = f"{topo}_{tag}"
        entry = all_results.get((grid_size, tag_full), {})
        runs = entry.get(topo, [])
        if runs:
            results[topo] = runs

    if graphs_sweep is not None:
        graphs = {
            topo: graphs_sweep[grid_size][f"{topo}_{tag}"]
            for topo in results
            if f"{topo}_{tag}" in graphs_sweep.get(grid_size, {})
        }
    else:
        graphs = _rebuild_one(grid_size, tag)

    return results, graphs


# ─────────────────────────────────────────────
# Two-player coexistence index (corrected formula)
# ─────────────────────────────────────────────

def ci_2p(r: dict) -> float:
    """Stones-normalised coexistence index for a two-player result dict.

    Formula: 1 - |black - white| / (black + white)
      = 0  when one player holds all stones (monopoly)
      = 1  when both hold equal stones (coexistence)

    Use this instead of r["coexistence_index"] for two-player pkl files
    produced before 2026-04-05, which store the old board-normalised
    formula  1 - |Δ| / n_nodes  (runs from two_p_*_v1.pkl).

    Falls back to r["coexistence_index"] if occupancy fields are absent
    (e.g. three-player records, which store Jain's index there).
    """
    b = r.get("black_occupancy")
    w = r.get("white_occupancy")
    if b is None or w is None:
        return r.get("coexistence_index", float("nan"))
    total = b + w
    if total <= 0:
        return float("nan")
    return float(1.0 - abs(b - w) / total)


# ─────────────────────────────────────────────
# Regime classification
# ─────────────────────────────────────────────

def classify_regime(
    r: dict,
    cap_rate_thresh: float = 0.05,
    ci_thresh: float = 0.80,   # matches notebook CAP_RATE_THRESH / CI_THRESH constants
) -> str:
    cap = r.get("capture_rate_per_move", 1.0)
    ci  = r.get("coexistence_index", 0.0)
    if cap < cap_rate_thresh and ci > ci_thresh:
        return "segregation"
    return "dominance"


# ─────────────────────────────────────────────
# Eye analysis
# ─────────────────────────────────────────────

def analyse_eyes(G: nx.Graph, state_dict: dict, color: int) -> dict:
    """Analyse eye-like structures for `color` in the given board state."""
    gs = make_game_from_graph(G, state=state_dict)

    eye_nodes = [v for v in G.nodes() if is_true_eye_like(gs, v, color)]
    eye_set = set(eye_nodes)
    eye_subgraph = G.subgraph(eye_set)
    components = list(nx.connected_components(eye_subgraph))
    comp_sizes = sorted([len(c) for c in components], reverse=True)

    stone_groups = groups_and_liberties(gs, color)
    eyes_per_group = []
    n_alive_groups = 0
    for group, _ in stone_groups:
        group_eyes = {v for v in eye_set if any(u in group for u in gs.neigh[v])}
        local_eye_graph = G.subgraph(group_eyes)
        n_local_comps = nx.number_connected_components(local_eye_graph)
        eyes_per_group.append(n_local_comps)
        if n_local_comps >= 2:
            n_alive_groups += 1

    n_stones = sum(1 for v in G.nodes() if state_dict.get(v, 0) == color)
    label = "black" if color == +1 else "white"

    return {
        "color":               label,
        "n_eyes":              len(eye_nodes),
        "n_stones":            n_stones,
        "eye_stone_ratio":     len(eye_nodes) / n_stones if n_stones else 0.0,
        "n_eye_components":    len(components),
        "eye_comp_sizes":      comp_sizes[:5],
        "max_eye_component":   comp_sizes[0] if comp_sizes else 0,
        "n_stone_groups":      len(stone_groups),
        "n_alive_groups":      n_alive_groups,
        "alive_group_fraction": n_alive_groups / len(stone_groups) if stone_groups else 0.0,
        "mean_eyes_per_group": sum(eyes_per_group) / len(eyes_per_group) if eyes_per_group else 0.0,
    }


def collect_sweep_eye_data(
    all_results: dict,
    graphs_by_gs: dict,
    grid_sizes: list[int] | None = None,
    target_degrees: dict[str, float] | None = None,
    topos: list[str] = ("er", "sbm"),
    cap_rate_thresh: float = 0.05,
    ci_thresh: float = 0.80,
) -> list[dict]:
    """Return a flat list of per-run dicts with regime + eye analysis fields."""
    if grid_sizes is None:
        grid_sizes = _DEFAULT_GRID_SIZES
    if target_degrees is None:
        target_degrees = _DEFAULT_TARGET_DEGREES
    records = []
    for gs in grid_sizes:
        for topo in topos:
            for tag, target_k in target_degrees.items():
                tag_full = f"{topo}_{tag}"
                G = graphs_by_gs.get(gs, {}).get(tag_full)
                if G is None:
                    print(f"  missing graph: gs={gs} {tag_full}")
                    continue
                runs = all_results.get((gs, tag_full), {}).get(topo, [])
                for r in runs:
                    state = r.get("sat_state") or r.get("final_state")
                    if state is None:
                        continue

                    regime = classify_regime(r, cap_rate_thresh, ci_thresh)
                    b = analyse_eyes(G, state, +1)
                    w = analyse_eyes(G, state, -1)

                    b_alive = b["alive_group_fraction"] == 1.0
                    w_alive = w["alive_group_fraction"] == 1.0
                    if b_alive and not w_alive:
                        winner, loser = b, w
                    elif w_alive and not b_alive:
                        winner, loser = w, b
                    else:
                        winner = loser = None

                    records.append(dict(
                        grid_size        = gs,
                        topo             = topo,
                        tag              = tag,
                        target_k         = target_k,
                        regime           = regime,
                        capture_rate     = r.get("capture_rate_per_move", np.nan),
                        ci               = r.get("coexistence_index", np.nan),
                        interblock       = r.get("interblock_mix") or np.nan,
                        winner_eyes      = winner["n_eyes"]              if winner else np.nan,
                        winner_stones    = winner["n_stones"]            if winner else np.nan,
                        eye_ratio        = winner["eye_stone_ratio"]     if winner else np.nan,
                        loser_stones     = loser["n_stones"]             if loser  else np.nan,
                        loser_groups     = loser["n_stone_groups"]       if loser  else np.nan,
                        loser_eyes       = loser["n_eyes"]               if loser  else np.nan,
                        loser_alive      = loser["n_alive_groups"]       if loser  else np.nan,
                        loser_alive_frac = loser["alive_group_fraction"] if loser  else np.nan,
                    ))
    return records


# ─────────────────────────────────────────────
# Numerical utilities
# ─────────────────────────────────────────────

def _pad_to(T: int, arr, fill: float = np.nan) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.size >= T:
        return arr[:T]
    out = np.full(T, fill, dtype=float)
    out[:arr.size] = arr
    return out


def _nanmean_std(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)


# ─────────────────────────────────────────────
# Graph descriptors
# (canonical: cell 47's more robust version)
# ─────────────────────────────────────────────

def graph_descriptors(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    deg = np.fromiter((d for _, d in G.degree()), dtype=float, count=n)
    return {
        "n":          int(n),
        "m":          int(m),
        "deg_mean":   float(deg.mean()) if n else 0.0,
        "deg_std":    float(deg.std(ddof=0)) if n else 0.0,
        "clustering": float(nx.average_clustering(G)) if n else 0.0,
    }


# ─────────────────────────────────────────────
# Tabular display
# (canonical: cell 10)
# ─────────────────────────────────────────────

def print_ecology_table(
    table_rows: list[dict],
    show: list[tuple[str, str]] | None = None,
) -> None:
    """Pretty-print mean ± sd table without pandas."""
    if show is None:
        show = [
            ("coexistence_index",    "1 - Dominance"),
            ("capture_rate_per_move","CapRate"),
            ("black_occupancy",      "Occ(B)"),
            ("white_occupancy",      "Occ(W)"),
            ("occ_mean",             "Biomass"),
            ("occ_diff_abs",         "Dominance"),
            ("largest_frac_mean",    "LargestFrac"),
            ("entropy_mean",         "Entropy"),
            ("frag_mean",            "Frag"),
            ("ataris_mean",          "Atari"),
            ("moves",                "Moves"),
            ("passes",               "Passes"),
        ]

    def fmt(mu, sd):
        if np.isnan(mu):
            return "   nan"
        return f"{mu:7.3f} ± {sd:6.3f}"

    hdr = "Topology  n  " + "  ".join(f"{lbl:>14s}" for _, lbl in show)
    print(hdr)
    print("-" * len(hdr))
    for row in table_rows:
        line = f"{row['topology']:<8s} {row['n_runs']:>2d} "
        for key, _lbl in show:
            mu = row.get(key + "_mean", np.nan)
            sd = row.get(key + "_sd",   0.0)
            line += "  " + f"{fmt(mu, sd):>12s}"
        print(line)


# ─────────────────────────────────────────────
# Snapshot plotting
# (canonical: cell 41 — the working version without the free-variable topo bug)
# ─────────────────────────────────────────────

def plot_snapshot_pair(
    r: dict,
    G: nx.Graph,
    *,
    title: str | None = None,
    node_size: float = 22,
    edge_alpha: float = 0.05,
    edge_sample: float = 0.2,
    seed: int = 0,
) -> None:
    """Plot occupancy and resource-field snapshots side by side.

    Requires coords, final_state, and final_R to be stored in the result dict r.
    (coords and final_state are always present since the recent fix;
     final_R requires store_replay=True.)
    """
    coords     = r.get("coords")
    state      = r.get("final_state")
    R          = r.get("final_R")

    missing = [k for k, v in [("coords", coords), ("final_state", state), ("final_R", R)] if v is None]
    if missing:
        print(f"plot_snapshot_pair: missing {missing} in result dict — skipping.")
        return

    nodes = list(coords.keys())
    xs = np.array([coords[v][0] for v in nodes], float)
    ys = np.array([coords[v][1] for v in nodes], float)
    rv = np.array([R.get(v, np.nan) for v in nodes], float)

    occ_c = []
    for v in nodes:
        s = state.get(v, 0)
        if s == +1:   occ_c.append("black")
        elif s == -1: occ_c.append("white")
        else:         occ_c.append("0.7")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    rng = np.random.default_rng(seed)
    edges = list(G.edges())
    if edge_sample < 1.0:
        m = int(len(edges) * edge_sample)
        if m > 0:
            edges = [edges[i] for i in rng.choice(len(edges), size=m, replace=False)]

    for ax in axes:
        for u, v in edges:
            x1, y1 = coords[u]
            x2, y2 = coords[v]
            ax.plot([x1, x2], [y1, y2], alpha=edge_alpha, linewidth=0.25)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")

    axes[0].scatter(xs, ys, c=occ_c, s=node_size, linewidths=0.2, edgecolors="0.2")
    axes[0].set_title("Occupancy (black / white / empty)")

    sc = axes[1].scatter(xs, ys, c=rv, s=node_size, linewidths=0.2, edgecolors="0.2")
    fig.colorbar(sc, ax=axes[1], label="R")
    axes[1].set_title("Resource field (R)")

    fig.suptitle(title or "Snapshot pair", y=0.98)
    plt.tight_layout()
    plt.show()


def plot_plot_snapshot_pairs(
    results: dict,
    graphs: dict,
    topologies: list[str],
    n_runs: int = 2,
) -> None:
    """Plot snapshot pairs for the first n_runs of each topology."""
    for topo in topologies:
        runs = results.get(topo, [])
        G    = graphs.get(topo)
        if not runs or G is None:
            continue
        for j in range(min(n_runs, len(runs))):
            plot_snapshot_pair(
                runs[j], G,
                title=f"{topo} — run {j} — Snapshot pair",
            )
