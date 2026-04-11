"""
rps_sweep.py — Three-species RPS / all-vs-all sweep on ER, SBM, DC-SBM, lattice, torus.

Species:  +1 (black), -1 (white), +2 (red)
Contests: rps       → prey_of = {+1:(-1,), -1:(+2,), +2:(+1,)}
          all_vs_all→ prey_of = None
Schedulers: sync (round-robin), gillespie_occ (energy-weighted async)
Graphs:
  er      – Erdős-Rényi G(n,p), p scaled to target mean degree k
  sbm     – Stochastic block model, 3 asymmetric blocks, modularity_ratio=8
  dc_sbm  – Degree-corrected SBM (theta_std=0.3), same block structure as sbm
  lattice – gs×gs 2D grid, mean degree ≈ 3.6–3.8 (fixed topology, k arg ignored)
  torus   – gs×gs toroidal grid, all nodes degree 4 (fixed topology, k arg ignored)

Connectivity: all graph generators log a warning and return None on failure.
  _run_single records None graphs as connectivity_failed=True in results.
  Fixed-topology graphs (lattice, torus) are always connected.

Sub-sweeps:
  FULL_PARAMS       – ER + SBM,    k ∈ {8,10,12,14,19}, 26 reps → 4 160 runs
  DCSBM_PARAMS      – DC-SBM only, k ∈ {10,12,14,19},   26 reps → 1 664 runs  (k=8 excluded)
  FIXED_TOPO_PARAMS – lattice+torus, k_nominal=4,        26 reps →   832 runs
"""

from __future__ import annotations

import math
import itertools
import time
import os
import pickle
import concurrent.futures
from pathlib import Path
from typing import Literal

import numpy as np
import networkx as nx
from scipy.optimize import brentq

from run_one_game import run_one_game

# ─────────────────────────────────────────────
# Species / contest definitions
# ─────────────────────────────────────────────

COLORS = (+1, -1, +2)          # black, white, red

PREY_OF = {
    "rps":       {+1: (-1,), -1: (+2,), +2: (+1,)},
    "all_vs_all": None,
}

COLOR_LABEL = {+1: "black", -1: "white", +2: "red"}

# ─────────────────────────────────────────────
# Policy definitions
# B = invasive (competitive); A = coexist (niche-seeking)
# ─────────────────────────────────────────────

_policy_balanced = dict(
    avoid_eye_filling=True,
    w_own_adj=3.0, w_libs=2.0, w_size=0.5,
    w_captures=10.0, w_opp_pressure=6.0,
    self_atari_penalty=20.0, w_degree=2.0, w_center=0.0,
)
_policy_invasive = dict(
    w_overlap=+1.5, w_opp_pressure=8.0, w_captures=12.0,
    self_atari_penalty=15.0, w_territory=0.05, territory_depth=2,
    w_center=0.0, normalize_degree=True,
)
_policy_coexist = dict(
    w_overlap=-1.5, w_territory=0.2, territory_depth=2,
    w_center=0.0, normalize_degree=True,
)

policyB = {**_policy_balanced, **_policy_invasive}   # invasive
policyA = {**_policy_balanced, **_policy_coexist}    # coexist

# Letters map to COLORS in order: (+1, -1, +2)
POLICY_CONFIGS: dict[str, dict[int, dict]] = {
    "BBB": {+1: policyB, -1: policyB, +2: policyB},
    "BAB": {+1: policyB, -1: policyA, +2: policyB},
    "BAA": {+1: policyB, -1: policyA, +2: policyA},
    "AAA": {+1: policyA, -1: policyA, +2: policyA},
}

# ─────────────────────────────────────────────
# Two-player definitions
# ─────────────────────────────────────────────

COLORS_2P = (+1, -1)    # black, white

# Letters map to (+1, -1) in order
POLICY_CONFIGS_2P: dict[str, dict[int, dict]] = {
    "BB": {+1: policyB, -1: policyB},
    "BA": {+1: policyB, -1: policyA},
    "AB": {+1: policyA, -1: policyB},
    "AA": {+1: policyA, -1: policyA},
}

# ─────────────────────────────────────────────
# Graph builders
# ─────────────────────────────────────────────

def er_p_for_degree(grid_size: int, target_k: float) -> float:
    n = grid_size ** 2
    return target_k / (n - 1)


def sbm_P_for_degree(
    grid_size: int,
    target_k: float,
    modularity_ratio: float = 8.0,
):
    """Return (sizes, P) for nx.stochastic_block_model with mean degree ≈ target_k."""
    n = grid_size ** 2

    # Integer block sizes summing exactly to n (3:2:1 ratio)
    unit = n / 6
    s = [round(3 * unit), round(2 * unit), round(unit)]
    s[-1] = n - s[0] - s[1]          # force exact sum
    assert sum(s) == n and all(si > 0 for si in s), f"Bad sizes: {s}"

    r = 1.0 / modularity_ratio

    def mean_deg(pw: float) -> float:
        pb = pw * r
        P = [
            [pw,         pb,        pb * 0.5 ],
            [pb,         pw * 0.75, pb       ],
            [pb * 0.5,   pb,        pw * 0.875],
        ]
        return sum(s[i] * s[j] * P[i][j] for i in range(3) for j in range(3)) / n

    pw = brentq(lambda pw: mean_deg(pw) - target_k, 1e-6, 1.0 - 1e-9)
    pb = pw * r
    P = [
        [pw,         pb,        pb * 0.5 ],
        [pb,         pw * 0.75, pb       ],
        [pb * 0.5,   pb,        pw * 0.875],
    ]
    return s, P


def make_er_graph(
    grid_size: int, target_k: float, seed: int, max_tries: int = 20
) -> nx.Graph:
    """Build a connected ER graph; raises RuntimeError if connectivity not achieved."""
    n = grid_size ** 2
    p = er_p_for_degree(grid_size, target_k)
    for attempt in range(max_tries):
        G = nx.erdos_renyi_graph(n, p, seed=seed + attempt)
        if nx.is_connected(G):
            G.graph["regen_attempts"] = attempt + 1
            if attempt > 0:
                print(
                    f"    ER connected after {attempt + 1} attempts "
                    f"(n={n}, k={target_k}, p={p:.5f})",
                    flush=True,
                )
            return G
    print(
        f"  SKIP er n={n} k={target_k} seed={seed}: "
        f"no connected graph after {max_tries} tries (p={p:.5f})",
        flush=True,
    )
    return None


def make_sbm_graph(
    grid_size: int, target_k: float, seed: int, max_tries: int = 20
) -> nx.Graph:
    """Build a connected SBM graph; raises RuntimeError if connectivity not achieved."""
    sizes, P = sbm_P_for_degree(grid_size, target_k)
    for attempt in range(max_tries):
        G = nx.stochastic_block_model(sizes, P, seed=seed + attempt)
        if nx.is_connected(G):
            G.graph["regen_attempts"] = attempt + 1
            if attempt > 0:
                print(
                    f"    SBM connected after {attempt + 1} attempts "
                    f"(n={grid_size ** 2}, k={target_k})",
                    flush=True,
                )
            return G
    print(
        f"  SKIP sbm n={grid_size ** 2} k={target_k} seed={seed}: "
        f"no connected graph after {max_tries} tries",
        flush=True,
    )
    return None


def make_lattice_graph(grid_size: int, target_k: float, seed: int) -> nx.Graph:
    """Build a gs×gs 2D grid graph. Always connected; target_k and seed are ignored.

    Actual mean degree = (4*(gs-2)² + 3*4*(gs-2) + 2*4) / n, which is ≈ 3.56 (gs=9)
    to ≈ 3.84 (gs=25).  Interior nodes have degree 4; edges 3; corners 2.
    """
    G_raw = nx.grid_2d_graph(grid_size, grid_size)
    G = nx.relabel_nodes(G_raw, {node: i for i, node in enumerate(sorted(G_raw.nodes()))})
    G.graph["regen_attempts"] = 1
    return G


def make_torus_graph(grid_size: int, target_k: float, seed: int) -> nx.Graph:
    """Build a gs×gs toroidal lattice (periodic boundary conditions). Always connected;
    target_k and seed are ignored.  Every node has degree 4 exactly.
    """
    G_raw = nx.grid_2d_graph(grid_size, grid_size, periodic=True)
    G = nx.relabel_nodes(G_raw, {node: i for i, node in enumerate(sorted(G_raw.nodes()))})
    G.graph["regen_attempts"] = 1
    return G


def sbm_with_degree_correction(
    grid_size: int,
    target_k: float,
    seed: int,
    theta_std: float = 0.3,
    max_tries: int = 50,
) -> nx.Graph:
    """
    DC-SBM: same 3-block structure as make_sbm_graph but with per-node degree
    heterogeneity parameterised by theta_std.

    For each node u in block i, draw theta_u ~ Lognormal(mu, theta_std) where
    mu is set so E[theta_u] = 1 within each block.  Edge probability between
    u (block i) and v (block j) is min(1, theta_u * theta_v * P[i][j]).

    Connectivity is verified; regenerated up to max_tries times (no edge bridging).

    G.graph stores:
        theta_cv          – overall CV of the theta vector
        theta_std_param   – the theta_std parameter used
        block_deg_stats   – dict[block_idx] -> {mean, std, min, max} of actual degrees
        dc_sbm_attempts   – number of attempts needed to achieve connectivity
    """
    n = grid_size ** 2
    sizes, P = sbm_P_for_degree(grid_size, target_k)
    P_arr = np.array(P)

    # Precompute node-to-block mapping
    block_assign = np.repeat(np.arange(3), sizes)

    # Node index ranges per block
    block_starts = [0, sizes[0], sizes[0] + sizes[1]]
    node_lists = [
        np.arange(block_starts[b], block_starts[b] + sizes[b]) for b in range(3)
    ]

    rng = np.random.default_rng(seed)

    for attempt in range(max_tries):
        # Draw and normalise theta per block so within-block mean = 1
        sigma = theta_std
        mu = -0.5 * sigma ** 2           # ensures E[lognormal] = 1
        theta = np.empty(n)
        for b, nodes in enumerate(node_lists):
            raw = rng.lognormal(mu, sigma, size=len(nodes))
            raw /= raw.mean()
            theta[nodes] = raw

        # Build graph with Bernoulli edges scaled by theta_u * theta_v
        G = nx.Graph()
        G.add_nodes_from(range(n))
        nx.set_node_attributes(G, {int(i): int(block_assign[i]) for i in range(n)}, "block")

        for bi in range(3):
            for bj in range(bi, 3):
                p_base = P_arr[bi, bj]
                u_arr = node_lists[bi]
                v_arr = node_lists[bj]

                if bi == bj:
                    # Within-block: upper triangle only, no self-loops
                    probs = np.minimum(
                        1.0, np.outer(theta[u_arr], theta[u_arr]) * p_base
                    )
                    rand_mat = rng.random(probs.shape)
                    mask = np.triu(rand_mat < probs, k=1)
                    ei_arr, ej_arr = np.where(mask)
                    for ei, ej in zip(ei_arr, ej_arr):
                        G.add_edge(int(u_arr[ei]), int(u_arr[ej]))
                else:
                    # Between-block: all pairs
                    probs = np.minimum(
                        1.0, np.outer(theta[u_arr], theta[v_arr]) * p_base
                    )
                    rand_mat = rng.random(probs.shape)
                    mask = rand_mat < probs
                    ei_arr, ej_arr = np.where(mask)
                    for ei, ej in zip(ei_arr, ej_arr):
                        G.add_edge(int(u_arr[ei]), int(v_arr[ej]))

        if not nx.is_connected(G):
            continue

        # Connected — record metadata
        block_deg_stats: dict[int, dict] = {}
        for b, nodes in enumerate(node_lists):
            degs = np.array([G.degree(int(v)) for v in nodes], dtype=float)
            block_deg_stats[b] = {
                "mean": float(degs.mean()),
                "std":  float(degs.std()),
                "min":  int(degs.min()),
                "max":  int(degs.max()),
            }

        G.graph["theta_cv"]        = float(theta.std() / theta.mean())
        G.graph["theta_std_param"] = theta_std
        G.graph["block_deg_stats"] = block_deg_stats
        G.graph["dc_sbm_attempts"] = attempt + 1
        G.graph["regen_attempts"]  = attempt + 1

        if attempt > 0:
            print(
                f"    DC-SBM connected after {attempt + 1} attempts "
                f"(n={n}, k={target_k}, theta_std={theta_std})",
                flush=True,
            )
        return G

    print(
        f"  SKIP dc_sbm n={n} k={target_k} theta_std={theta_std} seed={seed}: "
        f"no connected graph after {max_tries} tries",
        flush=True,
    )
    return None


GRAPH_MAKERS = {
    "er":      make_er_graph,
    "sbm":     make_sbm_graph,
    "lattice": make_lattice_graph,
    "torus":   make_torus_graph,
    "dc_sbm":  sbm_with_degree_correction,
}

# Graph types that carry "block" node attributes (needed for store_interblock)
_BLOCK_GRAPH_TYPES = {"sbm", "dc_sbm"}

# ─────────────────────────────────────────────
# Single condition runner
# ─────────────────────────────────────────────

def run_condition(
    contest:    Literal["rps", "all_vs_all"],
    graph_type: Literal["er", "sbm"],
    grid_size:  int,
    scheduler:  Literal["sync", "gillespie_occ"],
    target_k:   float = 10,
    n_moves:    int   = 2000,
    n_reps:     int   = 3,
    seed_base:  int   = 0,
    store_occ_ts: bool = True,
    occ_every:  int   = 20,
) -> list[dict]:
    """Run n_reps replicates for one (contest × graph_type × grid_size × scheduler) cell."""
    prey_of = PREY_OF[contest]
    make_graph = GRAPH_MAKERS[graph_type]
    results = []

    for rep in range(n_reps):
        graph_seed = seed_base + rep * 1000
        game_seed  = seed_base + rep * 1000 + 1

        G = make_graph(grid_size, target_k, graph_seed)

        if G is None:
            results.append({
                "_contest":            contest,
                "_graph_type":         graph_type,
                "_grid_size":          grid_size,
                "_scheduler":          scheduler,
                "_rep":                rep,
                "_n_nodes":            grid_size ** 2,
                "connectivity_failed": True,
            })
            continue

        r = run_one_game(
            G,
            seed=game_seed,
            n_moves=n_moves,
            colors=COLORS,
            prey_of=prey_of,
            scheduler=scheduler,
            use_gate=True,
            store_replay=False,
            store_metrics=False,       # hardcoded black/white, skip for 3-species
            store_timeseries=True,
            store_occ_ts=store_occ_ts,
            occ_every=occ_every,
            store_eff_ts=False,        # hardcoded black/white
            store_final_libs=False,
            store_interblock=(graph_type in _BLOCK_GRAPH_TYPES),
            store_R_ts=False,
            store_sat_state=False,
        )

        r["_contest"]    = contest
        r["_graph_type"] = graph_type
        r["_grid_size"]  = grid_size
        r["_scheduler"]  = scheduler
        r["_rep"]        = rep
        r["_n_nodes"]    = grid_size ** 2
        results.append(r)

    return results


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────

def _mean_std(vals):
    a = np.array(vals, dtype=float)
    return float(np.mean(a)), float(np.std(a))


def summary_table(all_results: list[dict]) -> None:
    """Print per-condition mean ± std of key metrics."""
    from collections import defaultdict

    skipped = [r for r in all_results if r.get("connectivity_failed")]
    if skipped:
        print(f"  {len(skipped)} run(s) skipped (connectivity_failed):")
        skip_counts: dict[tuple, int] = defaultdict(int)
        for r in skipped:
            skip_counts[(r["_graph_type"], r["_grid_size"],
                         r.get("_target_k", "?"))  ] += 1
        for key, cnt in sorted(skip_counts.items()):
            print(f"    graph={key[0]}  n={key[1]**2}  k={key[2]}  skipped={cnt}")
        print()

    valid = [r for r in all_results if not r.get("connectivity_failed")]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in valid:
        key = (r["_contest"], r["_graph_type"], r["_grid_size"],
               r["_scheduler"], r.get("_policy", "?"), r.get("_target_k", "?"))
        groups[key].append(r)

    header = (
        f"{'contest':<12} {'graph':<5} {'n':>5} {'sched':<14} "
        f"{'policy':<5} {'k':>4} "
        f"{'survivors':>10} {'jain':>8} {'largest':>9} {'moves':>7}"
    )
    print(header)
    print("-" * len(header))

    for key in sorted(groups):
        contest, graph_type, grid_size, scheduler, policy, target_k = key
        rs = groups[key]
        surv_m, surv_s   = _mean_std([r["survivors_count"]     for r in rs])
        jain_m, jain_s   = _mean_std([r["jain_fairness_final"] for r in rs])
        lrg_m,  lrg_s    = _mean_std([r["largest_share"]       for r in rs])
        mov_m,  _        = _mean_std([r["moves"]               for r in rs])
        n = grid_size ** 2
        print(
            f"{contest:<12} {graph_type:<5} {n:>5} {scheduler:<14} "
            f"{str(policy):<5} {str(target_k):>4} "
            f"{surv_m:>6.2f}±{surv_s:.2f} "
            f"{jain_m:>5.3f}±{jain_s:.3f} "
            f"{lrg_m:>6.3f}±{lrg_s:.3f} "
            f"{mov_m:>7.0f}"
        )


# ─────────────────────────────────────────────
# Sweeps
# ─────────────────────────────────────────────

# 2-rep smoke test: 1 grid × 1 k × 4 policies × 2 graphs = 16 runs
SANITY_PARAMS = dict(
    grid_sizes  = [9],
    k_values    = [10],
    n_moves     = 5_000,
    n_reps      = 2,
    contests    = ["rps"],
    graph_types = ["er", "sbm"],
    schedulers  = ["gillespie_occ"],
    policies    = ["BBB", "BAB", "BAA", "AAA"],
)

# Full production sweep: 4 grid × 5 k × 4 policies × 2 graphs × 26 reps = 4160 runs
FULL_PARAMS = dict(
    grid_sizes  = [9, 15, 19, 25],
    k_values    = [8, 10, 12, 14, 19],
    n_moves     = 5_000,
    n_reps      = 26,
    contests    = ["rps"],
    graph_types = ["er", "sbm"],
    schedulers  = ["gillespie_occ"],
    policies    = ["BBB", "BAB", "BAA", "AAA"],
)

# DC-SBM sweep: k=8 excluded (near-connectivity-threshold at n=625; see sbm_with_degree_correction)
# 4 k × 4 grid × 1 contest × 1 scheduler × 4 policies × 1 graph × 26 reps = 1 664 runs
DCSBM_PARAMS = dict(
    grid_sizes  = [9, 15, 19, 25],
    k_values    = [10, 12, 14, 19],
    n_moves     = 5_000,
    n_reps      = 26,
    contests    = ["rps"],
    graph_types = ["dc_sbm"],
    schedulers  = ["gillespie_occ"],
    policies    = ["BBB", "BAB", "BAA", "AAA"],
)

# Fixed-topology sweep: lattice and torus — k is a nominal placeholder (topology determines degree)
# Lattice: mean degree ≈ 3.56–3.84 by grid size.  Torus: always k=4.
# 1 k × 4 grid × 1 contest × 1 scheduler × 4 policies × 2 graphs × 26 reps = 832 runs
FIXED_TOPO_PARAMS = dict(
    grid_sizes  = [9, 15, 19, 25],
    k_values    = [4],           # nominal only; ignored by lattice/torus makers
    n_moves     = 5_000,
    n_reps      = 26,
    contests    = ["rps"],
    graph_types = ["lattice", "torus"],
    schedulers  = ["gillespie_occ"],
    policies    = ["BBB", "BAB", "BAA", "AAA"],
)

# ── All-vs-all mirrors ────────────────────────────────────────────────────────
# Identical structure to RPS params; contest="all_vs_all" means prey_of=None:
# every species can capture every other symmetrically (no cyclic hierarchy).
# 4 grid × 5 k × 4 policies × 2 graphs × 26 reps = 4 160 runs
AVA_PARAMS = dict(
    grid_sizes  = [9, 15, 19, 25],
    k_values    = [8, 10, 12, 14, 19],
    n_moves     = 5_000,
    n_reps      = 26,
    contests    = ["all_vs_all"],
    graph_types = ["er", "sbm"],
    schedulers  = ["gillespie_occ"],
    policies    = ["BBB", "BAB", "BAA", "AAA"],
)

# k=8 excluded for DC-SBM (connectivity threshold; see DCSBM_PARAMS)
# 4 k × 4 grid × 4 policies × 1 graph × 26 reps = 1 664 runs
AVA_DCSBM_PARAMS = dict(
    grid_sizes  = [9, 15, 19, 25],
    k_values    = [10, 12, 14, 19],
    n_moves     = 5_000,
    n_reps      = 26,
    contests    = ["all_vs_all"],
    graph_types = ["dc_sbm"],
    schedulers  = ["gillespie_occ"],
    policies    = ["BBB", "BAB", "BAA", "AAA"],
)

# 1 k × 4 grid × 4 policies × 2 graphs × 26 reps = 832 runs
AVA_FIXED_TOPO_PARAMS = dict(
    grid_sizes  = [9, 15, 19, 25],
    k_values    = [4],
    n_moves     = 5_000,
    n_reps      = 26,
    contests    = ["all_vs_all"],
    graph_types = ["lattice", "torus"],
    schedulers  = ["gillespie_occ"],
    policies    = ["BBB", "BAB", "BAA", "AAA"],
)

# ── Two-player sweep params ───────────────────────────────────────────────────
# Two players (+1, -1), prey_of=None (symmetric competition), use_gate=True.
# Both schedulers included: sync (round-robin) and gillespie_occ (energy-weighted).
# Policies: BB BA AB AA  (B=invasive, A=coexist; left letter = black/+1)

# Test run: 1 grid × 1 k × 2 sched × 4 policies × 2 graphs × 3 reps = 48 runs
TWO_P_TEST_PARAMS = dict(
    grid_sizes  = [15],
    k_values    = [10],
    n_moves     = 5_000,
    n_reps      = 3,
    contests    = ["two_player"],   # label only; prey_of=None set in _run_single_2p
    graph_types = ["er", "sbm"],
    schedulers  = ["sync", "gillespie_occ"],
    policies    = ["BB", "BA", "AB", "AA"],
)

# Full ER + SBM: 4 grid × 5 k × 2 sched × 4 policies × 2 graphs × 26 reps = 8 320 runs
TWO_P_FULL_PARAMS = dict(
    grid_sizes  = [9, 15, 19, 25],
    k_values    = [8, 10, 12, 14, 19],
    n_moves     = 5_000,
    n_reps      = 26,
    contests    = ["two_player"],
    graph_types = ["er", "sbm"],
    schedulers  = ["sync", "gillespie_occ"],
    policies    = ["BB", "BA", "AB", "AA"],
)

# DC-SBM: k=8 excluded — 4 k × 4 grid × 2 sched × 4 policies × 1 graph × 26 reps = 3 328 runs
TWO_P_DCSBM_PARAMS = dict(
    grid_sizes  = [9, 15, 19, 25],
    k_values    = [10, 12, 14, 19],
    n_moves     = 5_000,
    n_reps      = 26,
    contests    = ["two_player"],
    graph_types = ["dc_sbm"],
    schedulers  = ["sync", "gillespie_occ"],
    policies    = ["BB", "BA", "AB", "AA"],
)

# Fixed topology: 1 k × 4 grid × 2 sched × 4 policies × 2 graphs × 26 reps = 1 664 runs
TWO_P_FIXED_TOPO_PARAMS = dict(
    grid_sizes  = [9, 15, 19, 25],
    k_values    = [4],
    n_moves     = 5_000,
    n_reps      = 26,
    contests    = ["two_player"],
    graph_types = ["lattice", "torus"],
    schedulers  = ["sync", "gillespie_occ"],
    policies    = ["BB", "BA", "AB", "AA"],
)

# Legacy / exploratory — kept for notebook use
TEST_PARAMS = dict(
    grid_sizes  = [9],
    k_values    = [10],
    n_moves     = {"sync": 2_000, "gillespie_occ": 2_000},
    n_reps      = 3,
    contests    = ["rps", "all_vs_all"],
    graph_types = ["er", "sbm"],
    schedulers  = ["sync", "gillespie_occ"],
    policies    = ["BBB"],
)

SCALE_PARAMS = dict(
    grid_sizes  = [9, 15, 19, 25],
    k_values    = [10],
    n_moves     = 15_000,
    n_reps      = 3,
    contests    = ["rps", "all_vs_all"],
    graph_types = ["er", "sbm"],
    schedulers  = ["sync", "gillespie_occ"],
    policies    = ["BBB"],
)


# ─────────────────────────────────────────────
# Parallel runner  (individual-run granularity for best load balancing)
# ─────────────────────────────────────────────

def _resolve_n_moves(params: dict, scheduler: str) -> int:
    nm = params["n_moves"]
    return nm[scheduler] if isinstance(nm, dict) else nm


def _run_single(args):
    """Top-level (picklable) wrapper: run exactly one replicate."""
    (contest, graph_type, grid_size, scheduler,
     target_k, n_moves, rep, seed_base, policy_name) = args

    prey_of    = PREY_OF[contest]
    policy_kw  = POLICY_CONFIGS[policy_name]
    G = GRAPH_MAKERS[graph_type](grid_size, target_k, seed=seed_base + rep * 1000)

    if G is None:
        return {
            "_contest":            contest,
            "_graph_type":         graph_type,
            "_grid_size":          grid_size,
            "_scheduler":          scheduler,
            "_target_k":           target_k,
            "_policy":             policy_name,
            "_rep":                rep,
            "_n_nodes":            grid_size ** 2,
            "connectivity_failed": True,
        }

    r = run_one_game(
        G,
        seed=seed_base + rep * 1000 + 1,
        n_moves=n_moves,
        colors=COLORS,
        prey_of=prey_of,
        scheduler=scheduler,
        policy_kwargs=policy_kw,
        use_gate=True,
        store_replay=False,
        store_metrics=False,
        store_timeseries=True,
        store_occ_ts=True,
        occ_every=20,
        store_eff_ts=False,
        store_final_libs=False,
        store_interblock=(graph_type in _BLOCK_GRAPH_TYPES),
        store_R_ts=False,
        store_sat_state=False,
    )
    r["_contest"]    = contest
    r["_graph_type"] = graph_type
    r["_grid_size"]  = grid_size
    r["_scheduler"]  = scheduler
    r["_target_k"]   = target_k
    r["_policy"]     = policy_name
    r["_rep"]        = rep
    r["_n_nodes"]    = grid_size ** 2
    return r


def _run_single_2p(args):
    """Two-player picklable wrapper: run exactly one replicate.

    Two players (+1 black, -1 white), prey_of=None (each captures the other),
    resource gate active.  Policy names are two-letter: BB, BA, AB, AA.
    """
    (contest, graph_type, grid_size, scheduler,
     target_k, n_moves, rep, seed_base, policy_name) = args

    policy_kw = POLICY_CONFIGS_2P[policy_name]
    G = GRAPH_MAKERS[graph_type](grid_size, target_k, seed=seed_base + rep * 1000)

    if G is None:
        return {
            "_contest":            contest,
            "_graph_type":         graph_type,
            "_grid_size":          grid_size,
            "_scheduler":          scheduler,
            "_target_k":           target_k,
            "_policy":             policy_name,
            "_rep":                rep,
            "_n_nodes":            grid_size ** 2,
            "connectivity_failed": True,
        }

    r = run_one_game(
        G,
        seed=seed_base + rep * 1000 + 1,
        n_moves=n_moves,
        colors=COLORS_2P,
        prey_of=None,           # symmetric: each player captures the other
        scheduler=scheduler,
        policy_kwargs=policy_kw,
        use_gate=True,
        store_replay=False,
        store_metrics=False,
        store_timeseries=False,  # per-turn event logs not needed; occ_ts is independent
        store_occ_ts=True,
        occ_every=20,
        store_eff_ts=False,
        store_final_libs=False,
        store_interblock=(graph_type in _BLOCK_GRAPH_TYPES),
        store_R_ts=False,
        store_sat_state=False,
    )
    r["_contest"]    = contest
    r["_graph_type"] = graph_type
    r["_grid_size"]  = grid_size
    r["_scheduler"]  = scheduler
    r["_target_k"]   = target_k
    r["_policy"]     = policy_name
    r["_rep"]        = rep
    r["_n_nodes"]    = grid_size ** 2
    return r


def run_sweep(
    params: dict,
    seed_base: int = 0,
    n_workers: int | None = None,
    worker_fn=None,
) -> list[dict]:
    """Run a full parameter sweep, parallelising at individual-run granularity.

    worker_fn: picklable callable with the same signature as _run_single.
               Defaults to _run_single (3-player).  Pass _run_single_2p for
               two-player sweeps.
    """
    if worker_fn is None:
        worker_fn = _run_single
    conditions = list(itertools.product(
        params["contests"],
        params["graph_types"],
        params["grid_sizes"],
        params["schedulers"],
        params["policies"],
        params["k_values"],
    ))
    n_reps  = params["n_reps"]
    n_total = len(conditions) * n_reps
    if n_workers is None:
        n_workers = min(n_total, os.cpu_count() or 1)

    nm_display = params["n_moves"]
    print(
        f"Sweep: {len(conditions)} conditions × {n_reps} reps "
        f"= {n_total} runs  "
        f"(n_moves={nm_display}, k_values={params['k_values']}, "
        f"policies={params['policies']}, workers={n_workers})"
    )

    # one task per replicate
    tasks = []
    for i, (contest, graph_type, grid_size, scheduler, policy_name, target_k) in enumerate(conditions):
        n_moves = _resolve_n_moves(params, scheduler)
        cond_seed = seed_base + (i + 1) * 10_000
        for rep in range(n_reps):
            tasks.append((
                contest, graph_type, grid_size, scheduler,
                target_k, n_moves, rep, cond_seed, policy_name,
            ))

    all_results: list[dict] = []
    t0 = time.perf_counter()
    done = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(worker_fn, t): t for t in tasks}
        for fut in concurrent.futures.as_completed(futures):
            all_results.append(fut.result())
            done += 1
            if done % max(1, n_total // 20) == 0 or done == n_total:
                elapsed = time.perf_counter() - t0
                print(
                    f"  {done}/{n_total} runs done  "
                    f"({elapsed:.0f}s elapsed, "
                    f"~{elapsed/done*(n_total-done):.0f}s remaining)",
                    flush=True,
                )

    elapsed = time.perf_counter() - t0
    print(f"\nTotal: {elapsed:.1f}s  ({elapsed/len(all_results):.2f}s/run)\n")
    return all_results


# ─────────────────────────────────────────────
# Scaling analysis
# ─────────────────────────────────────────────

def scaling_table(all_results: list[dict]) -> None:
    """
    For each (grid_size, scheduler, contest, graph_type) show:
      - moves_mean   : actual moves placed (stones on board)
      - passes_mean  : pass events
      - total_mean   : moves + passes (= turns consumed of n_moves budget)
      - pass_frac    : passes / turns  — if >0.5 the game has run cold
      - survivors    : mean survivors at end
    This tells us how large n_moves needs to be to let games run to completion.
    """
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        if r.get("connectivity_failed"):
            continue
        key = (r["_grid_size"], r["_scheduler"], r["_contest"], r["_graph_type"])
        groups[key].append(r)

    header = (
        f"{'n':>5} {'sched':<14} {'contest':<12} {'graph':<5} "
        f"{'moves':>8} {'passes':>8} {'total':>8} {'pass%':>7} {'surv':>6}"
    )
    print(header)
    print("-" * len(header))

    for key in sorted(groups):
        grid_size, scheduler, contest, graph_type = key
        rs = groups[key]
        n = grid_size ** 2
        moves_m,  _ = _mean_std([r["moves"]          for r in rs])
        passes_m, _ = _mean_std([r["passes"]          for r in rs])
        total_m,  _ = _mean_std([r["turns"]           for r in rs])
        surv_m,   _ = _mean_std([r["survivors_count"] for r in rs])
        pass_frac = passes_m / max(1, total_m)
        print(
            f"{n:>5} {scheduler:<14} {contest:<12} {graph_type:<5} "
            f"{moves_m:>8.0f} {passes_m:>8.0f} {total_m:>8.0f} "
            f"{pass_frac:>6.1%} {surv_m:>6.2f}"
        )


# ─────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────

RESULTS_DIR = Path("results")


def save_results(results: list[dict], label: str) -> Path:
    """Pickle results to results/<label>.pkl and return the path."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{label}.pkl"
    # Strip bulky occ_ts_by_color arrays before saving to keep file small;
    # keep the scalar summary fields which are enough for all downstream analysis.
    # (set strip_ts=False below if you need the full trajectories)
    with open(path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = path.stat().st_size / 1e6
    print(f"Saved {len(results)} runs → {path}  ({size_mb:.1f} MB)")
    return path


def load_results(label: str) -> list[dict]:
    path = RESULTS_DIR / f"{label}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def results_to_dataframe(results: list[dict]):
    """Return a tidy pandas DataFrame with one row per run (scalar fields only)."""
    import pandas as pd
    SKIP = {
        "history", "final_state", "coords", "final_R",
        "metrics_history", "atari_counts", "capture_sizes", "captures_per_turn",
        "occ_b_ts", "occ_w_ts", "occ_t_ts", "occ_ts_by_color",
        "eff_b_ts", "eff_w_ts",
        "R_mean_ts", "R_occ_mean_ts", "R_empty_mean_ts", "R_occ_mass_ts",
        "pass_ts", "pass_with_legal_ts", "pass_no_legal_ts",
        "passes_failed_gate_ts", "attempt_ts", "color_ts", "illegal_ts",
        "passes_failed_gate_ts_by_color", "passes_failed_gate_ts_black",
        "passes_failed_gate_ts_white",
        "final_group_liberties_black", "final_group_liberties_white",
        "sat_state", "sat_R",
        # dict fields → expanded explicitly below
        "n_stones", "events_by_color", "moves_by_color", "passes_true_by_color",
        "passes_failed_gate_by_color", "cap_total_by_color", "cap_moves_by_color",
        "event_share_by_color",
        "interblock_rate_by_color", "occ_avg_last30p",
    }
    rows = []
    for r in results:
        if r.get("connectivity_failed"):
            # Preserve metadata so skips are visible in the dataframe
            rows.append({k: v for k, v in r.items() if not isinstance(v, (list, dict))})
            continue
        row = {k: v for k, v in r.items() if k not in SKIP and not isinstance(v, (list, dict))}
        # flatten n_stones per color
        for c, label in COLOR_LABEL.items():
            row[f"n_stones_{label}"] = r.get("n_stones", {}).get(c, 0)
        # flatten interblock_rate_by_color
        ib = r.get("interblock_rate_by_color")
        if isinstance(ib, dict):
            for c, label in COLOR_LABEL.items():
                row[f"interblock_rate_{label}"] = ib.get(c, float("nan"))
        # flatten occ_avg_last30p
        occ30 = r.get("occ_avg_last30p")
        if isinstance(occ30, dict):
            for c, label in COLOR_LABEL.items():
                row[f"occ_avg_last30p_{label}"] = occ30.get(c, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# DC-SBM verification
# ─────────────────────────────────────────────

def verify_dc_sbm(
    grid_sizes: tuple = (9, 15, 19, 25),
    target_k: float = 10.0,
    theta_std: float = 0.3,
    seed: int = 42,
    max_tries: int = 20,
    out_path: str = "dc_sbm_verify.png",
) -> None:
    """
    Generate one standard SBM and one DC-SBM per grid size, print a diagnostic
    table, and save a side-by-side degree-distribution plot.

    Checks:
      - both graphs connected
      - mean degree within 5 % of target_k
      - DC-SBM std(degree) visibly > SBM std(degree)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_sizes = len(grid_sizes)
    fig, axes = plt.subplots(2, n_sizes, figsize=(4 * n_sizes, 8))

    header = (
        f"{'grid':>6} {'n':>5} {'graph':>8} {'connected':>10} "
        f"{'mean_k':>8} {'std_k':>7} {'min_k':>6} {'max_k':>6} "
        f"{'theta_cv':>10} {'attempts':>9}"
    )
    print(f"\nDC-SBM Verification  (k={target_k}, theta_std={theta_std})\n")
    print(header)
    print("-" * len(header))

    all_ok = True

    for col, gs in enumerate(grid_sizes):
        n = gs ** 2

        G_sbm = make_sbm_graph(gs, target_k, seed=seed, max_tries=max_tries)
        G_dc  = sbm_with_degree_correction(
            gs, target_k, seed=seed, theta_std=theta_std, max_tries=max_tries
        )

        degs_sbm = np.array([d for _, d in G_sbm.degree()])
        degs_dc  = np.array([d for _, d in G_dc.degree()])

        for name, G, degs in [("sbm", G_sbm, degs_sbm), ("dc_sbm", G_dc, degs_dc)]:
            connected = nx.is_connected(G)
            mean_k    = float(degs.mean())
            std_k     = float(degs.std())
            theta_cv  = G.graph.get("theta_cv", float("nan"))
            attempts  = G.graph.get("regen_attempts", 1)

            pct_err = abs(mean_k - target_k) / target_k * 100
            flag = "  ***>5%" if pct_err > 5 else ""
            if not connected or pct_err > 5:
                all_ok = False

            print(
                f"{gs:>6} {n:>5} {name:>8} {str(connected):>10} "
                f"{mean_k:>8.2f} {std_k:>7.2f} {int(degs.min()):>6} "
                f"{int(degs.max()):>6} {theta_cv:>10.4f} {attempts:>9}{flag}"
            )

        # Check heterogeneity: DC-SBM should have higher std
        if degs_dc.std() <= degs_sbm.std():
            print(
                f"  WARNING n={n}: DC-SBM std ({degs_dc.std():.2f}) "
                f"<= SBM std ({degs_sbm.std():.2f}) — theta_std may be too small"
            )
            all_ok = False
        print()

        # ── Plots ──────────────────────────────────────────────
        kmax = int(max(degs_sbm.max(), degs_dc.max())) + 1
        bins = np.arange(0, kmax + 2) - 0.5

        ax_sbm = axes[0, col]
        ax_dc  = axes[1, col]

        ax_sbm.hist(degs_sbm, bins=bins, color="#4878CF", alpha=0.85, density=True)
        ax_sbm.axvline(target_k, color="k", ls="--", lw=1.5, label=f"k={target_k}")
        ax_sbm.set_title(
            f"n={n}  SBM\nmean={degs_sbm.mean():.1f}  std={degs_sbm.std():.2f}"
        )
        ax_sbm.set_xlabel("degree")
        if col == 0:
            ax_sbm.set_ylabel("density")
        ax_sbm.legend(fontsize=8)

        bds = G_dc.graph.get("block_deg_stats", {})
        block_info = "  ".join(
            f"b{b}: μ={bds[b]['mean']:.1f}" for b in sorted(bds)
        ) if bds else ""
        ax_dc.hist(degs_dc, bins=bins, color="#D65F5F", alpha=0.85, density=True)
        ax_dc.axvline(target_k, color="k", ls="--", lw=1.5, label=f"k={target_k}")
        ax_dc.set_title(
            f"n={n}  DC-SBM (θ_std={theta_std})\n"
            f"mean={degs_dc.mean():.1f}  std={degs_dc.std():.2f}  "
            f"CV={G_dc.graph.get('theta_cv', float('nan')):.3f}"
        )
        ax_dc.set_xlabel("degree")
        if col == 0:
            ax_dc.set_ylabel("density")
        ax_dc.legend(fontsize=8)
        if block_info:
            ax_dc.annotate(
                block_info, xy=(0.02, 0.97), xycoords="axes fraction",
                va="top", fontsize=7, color="0.4",
            )

    plt.suptitle(
        f"Degree distributions: SBM vs DC-SBM  (k={target_k}, θ_std={theta_std})",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plot saved → {out_path}")
    print(f"\nVerification {'PASSED' if all_ok else 'FAILED — see warnings above'}.")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    mode  = sys.argv[1] if len(sys.argv) > 1 else "sanity"
    label = sys.argv[2] if len(sys.argv) > 2 else mode

    if mode == "sanity":
        print("=== SANITY RUN (2 reps, n=81, k=10, ER+SBM, all 4 policies) ===\n")
        results = run_sweep(SANITY_PARAMS, seed_base=42)
        summary_table(results)
        save_results(results, f"sanity_{label}")

    elif mode == "full":
        print("=== FULL SWEEP — ER + SBM (4 160 runs) ===\n")
        results = run_sweep(FULL_PARAMS, seed_base=42)
        summary_table(results)
        save_results(results, f"full_{label}")

    elif mode == "full_dcsbm":
        print("=== FULL SWEEP — DC-SBM (1 664 runs, k∈{10,12,14,19}) ===\n")
        results = run_sweep(DCSBM_PARAMS, seed_base=42)
        summary_table(results)
        save_results(results, f"full_dcsbm_{label}")

    elif mode == "full_fixed":
        print("=== FULL SWEEP — lattice + torus (832 runs) ===\n")
        results = run_sweep(FIXED_TOPO_PARAMS, seed_base=42)
        summary_table(results)
        save_results(results, f"full_fixed_{label}")

    elif mode == "full_all":
        print("=== FULL SWEEP — ER + SBM (4 160 runs) ===\n")
        r1 = run_sweep(FULL_PARAMS, seed_base=42)
        summary_table(r1)
        save_results(r1, f"full_er_sbm_{label}")

        print("=== FULL SWEEP — DC-SBM (1 664 runs) ===\n")
        r2 = run_sweep(DCSBM_PARAMS, seed_base=42)
        summary_table(r2)
        save_results(r2, f"full_dcsbm_{label}")

        print("=== FULL SWEEP — lattice + torus (832 runs) ===\n")
        r3 = run_sweep(FIXED_TOPO_PARAMS, seed_base=42)
        summary_table(r3)
        save_results(r3, f"full_fixed_{label}")

        all_results = r1 + r2 + r3
        print(f"\n=== COMBINED SUMMARY ({len(all_results)} runs) ===\n")
        summary_table(all_results)
        save_results(all_results, f"full_all_{label}")

    elif mode == "full_ava":
        print("=== AVA SWEEP — ER + SBM (4 160 runs) ===\n")
        r1 = run_sweep(AVA_PARAMS, seed_base=42)
        summary_table(r1)
        save_results(r1, f"ava_er_sbm_{label}")

        print("=== AVA SWEEP — DC-SBM (1 664 runs) ===\n")
        r2 = run_sweep(AVA_DCSBM_PARAMS, seed_base=42)
        summary_table(r2)
        save_results(r2, f"ava_dcsbm_{label}")

        print("=== AVA SWEEP — lattice + torus (832 runs) ===\n")
        r3 = run_sweep(AVA_FIXED_TOPO_PARAMS, seed_base=42)
        summary_table(r3)
        save_results(r3, f"ava_fixed_{label}")

        all_results = r1 + r2 + r3
        print(f"\n=== COMBINED SUMMARY ({len(all_results)} runs) ===\n")
        summary_table(all_results)
        save_results(all_results, f"ava_all_{label}")

    elif mode == "test_2p":
        print("=== TWO-PLAYER TEST (48 runs: n=225, k=10, ER+SBM, sync+gillespie) ===\n")
        results = run_sweep(TWO_P_TEST_PARAMS, seed_base=42, worker_fn=_run_single_2p)
        summary_table(results)
        save_results(results, f"two_p_test_{label}")

    elif mode == "full_2p":
        print("=== TWO-PLAYER SWEEP — ER + SBM (8 320 runs) ===\n")
        r1 = run_sweep(TWO_P_FULL_PARAMS, seed_base=42, worker_fn=_run_single_2p)
        summary_table(r1)
        save_results(r1, f"two_p_er_sbm_{label}")

        print("=== TWO-PLAYER SWEEP — DC-SBM (3 328 runs) ===\n")
        r2 = run_sweep(TWO_P_DCSBM_PARAMS, seed_base=42, worker_fn=_run_single_2p)
        summary_table(r2)
        save_results(r2, f"two_p_dcsbm_{label}")

        print("=== TWO-PLAYER SWEEP — lattice + torus (1 664 runs) ===\n")
        r3 = run_sweep(TWO_P_FIXED_TOPO_PARAMS, seed_base=42, worker_fn=_run_single_2p)
        summary_table(r3)
        save_results(r3, f"two_p_fixed_{label}")

        all_results = r1 + r2 + r3
        print(f"\n=== COMBINED SUMMARY ({len(all_results)} runs) ===\n")
        summary_table(all_results)
        save_results(all_results, f"two_p_all_{label}")

    elif mode == "test":
        print("=== LEGACY TEST RUN (serial, n=81 only) ===\n")
        results = run_sweep(TEST_PARAMS, seed_base=42, n_workers=1)
        summary_table(results)
        save_results(results, f"test_{label}")

    elif mode == "scale":
        print("=== LEGACY SCALING TEST (parallel, all grid sizes) ===\n")
        results = run_sweep(SCALE_PARAMS, seed_base=42)
        print("\n--- move/pass budget analysis ---\n")
        scaling_table(results)
        print("\n--- outcome summary ---\n")
        summary_table(results)
        save_results(results, f"scale_{label}")

    elif mode == "verify_dc_sbm":
        print("=== DC-SBM VERIFICATION (k=10, theta_std=0.3, all grid sizes) ===\n")
        verify_dc_sbm(
            grid_sizes=(9, 15, 19, 25),
            target_k=10.0,
            theta_std=0.3,
            seed=42,
            out_path=f"dc_sbm_verify_{label}.png",
        )

    else:
        print(
            f"Unknown mode: {mode!r}. "
            "Use: sanity | full | full_dcsbm | full_fixed | full_all | "
            "test | scale | verify_dc_sbm"
        )
