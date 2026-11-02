"""
run_eps_sweep.py — Epsilon sensitivity diagnostic sweep.

Scientific question:
  Does policy insensitivity persist at lower stochastic exploration rates, or
  is it an artefact of the high ε=0.15 used in production sweeps?  Secondary
  question: how does ε affect eye formation and coexistence in two-player games?

This is supplementary diagnostic work, outside the current manuscript.

Design:
  Irregular graphs (ER, SBM, DC-SBM):  ε ∈ {0.02, 0.04, 0.07, 0.15}
  Regular graphs (lattice, torus):       ε ∈ {0.02, 0.15}  (two extremes)
  Scheduler: gillespie_occ only
  k:          10, 14
  Grid sizes: 15×15 (n=225), 25×25 (n=625)
  Reps:       26
  use_gate=True, seed_base=42

Competition modes (separate runs, separate output files):
  2p  — two-player;       policies: BB, BA, AB, AA
  rps — three-player RPS; policies: BBB, BAB, BAA, AAA
  ava — three-player AVA; policies: BBB, BAB, BAA, AAA

Run count (per sweep):
  Irregular: 4ε × 3 graphs × 2k × 2n × 4pol × 26reps = 4,992
  Regular:   2ε × 2 graphs × 2k × 2n × 4pol × 26reps = 1,664
  Total per sweep: 6,656 runs  (~19,968 across all three sweeps)

Seed design:
  seed_base=42 (same as production).  The eps axis sits outside the base-condition
  loop, so for a given (contest, graph_type, grid_size, scheduler, policy, k, rep)
  the graph_seed is identical across all ε values.  This enables clean within-sweep
  cross-ε comparison on the same graph instances.

  NOTE: condition indices in this sweep differ from production sub-sweeps because the
  eps sweep covers fewer grid_sizes and k_values.  The ε=0.15 rows therefore use
  different graph seeds than the matching production runs.  Cross-sweep comparison
  should be distributional (aggregate statistics per matched condition cell), not
  record-by-record.

Output files:
  results/eps_sweep_2p_v1.pkl
  results/eps_sweep_rps_v1.pkl
  results/eps_sweep_ava_v1.pkl

Each record stores:
  _eps        — ε value used
  _graph_seed — graph seed (for post-hoc eye analysis / graph reconstruction)
  All standard scalar fields from run_one_game.

Usage:
  python run_eps_sweep.py 2p v1
  python run_eps_sweep.py rps v1
  python run_eps_sweep.py ava v1
"""

from __future__ import annotations

import concurrent.futures
import itertools
import os
import time
from collections import defaultdict

import numpy as np

from run_one_game import run_one_game
from rps_sweep import (
    _BLOCK_GRAPH_TYPES,
    COLORS,
    COLORS_2P,
    GRAPH_MAKERS,
    POLICY_CONFIGS,
    POLICY_CONFIGS_2P,
    PREY_OF,
    save_results,
)

# ─────────────────────────────────────────────
# ε values
# ─────────────────────────────────────────────

EPS_IRREGULAR = [0.02, 0.04, 0.07, 0.15]   # ER, SBM, DC-SBM
EPS_REGULAR   = [0.02, 0.15]               # lattice, torus (two extremes)

_REGULAR_GRAPHS = {"lattice", "torus"}


def _eps_list(graph_type: str) -> list[float]:
    return EPS_REGULAR if graph_type in _REGULAR_GRAPHS else EPS_IRREGULAR


# ─────────────────────────────────────────────
# Sweep parameter dicts
# ─────────────────────────────────────────────

_COMMON = dict(
    grid_sizes = [15, 25],
    k_values   = [10, 14],
    n_moves    = 5_000,
    n_reps     = 26,
    schedulers = ["gillespie_occ"],
)

EPS_2P_PARAMS = dict(
    **_COMMON,
    contests    = ["two_player"],
    graph_types = ["er", "sbm", "dc_sbm", "lattice", "torus"],
    policies    = ["BB", "BA", "AB", "AA"],
)

EPS_RPS_PARAMS = dict(
    **_COMMON,
    contests    = ["rps"],
    graph_types = ["er", "sbm", "dc_sbm", "lattice", "torus"],
    policies    = ["BBB", "BAB", "BAA", "AAA"],
)

EPS_AVA_PARAMS = dict(
    **_COMMON,
    contests    = ["all_vs_all"],
    graph_types = ["er", "sbm", "dc_sbm", "lattice", "torus"],
    policies    = ["BBB", "BAB", "BAA", "AAA"],
)


# ─────────────────────────────────────────────
# Worker functions
# ─────────────────────────────────────────────

def _run_single_eps(args):
    """Three-player picklable worker for the eps sweep.

    Signature matches _run_single from rps_sweep.py, extended with eps at end.
    Stores _eps and _graph_seed in the result record.

    args: (contest, graph_type, grid_size, scheduler, target_k, n_moves,
           rep, cond_seed, policy_name, eps)
    """
    (contest, graph_type, grid_size, scheduler,
     target_k, n_moves, rep, cond_seed, policy_name, eps) = args

    prey_of    = PREY_OF[contest]
    policy_kw  = POLICY_CONFIGS[policy_name]
    graph_seed = cond_seed + rep * 1000

    G = GRAPH_MAKERS[graph_type](grid_size, target_k, seed=graph_seed)

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
            "_eps":                eps,
            "_graph_seed":         graph_seed,
            "connectivity_failed": True,
        }

    r = run_one_game(
        G,
        seed=cond_seed + rep * 1000 + 1,
        n_moves=n_moves,
        eps=eps,
        colors=COLORS,
        prey_of=prey_of,
        scheduler=scheduler,
        policy_kwargs=policy_kw,
        use_gate=True,
        store_replay=False,
        store_metrics=False,
        store_timeseries=False,   # scalars + final_state sufficient for eps analysis
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
    r["_eps"]        = eps
    r["_graph_seed"] = graph_seed
    return r


def _run_single_2p_eps(args):
    """Two-player picklable worker for the eps sweep.

    Two players (+1 black, -1 white), prey_of=None, use_gate=True.
    store_occ_ts=True so ci_2p() can be computed from occ_avg_last30p.

    args: (contest, graph_type, grid_size, scheduler, target_k, n_moves,
           rep, cond_seed, policy_name, eps)
    """
    (contest, graph_type, grid_size, scheduler,
     target_k, n_moves, rep, cond_seed, policy_name, eps) = args

    policy_kw  = POLICY_CONFIGS_2P[policy_name]
    graph_seed = cond_seed + rep * 1000

    G = GRAPH_MAKERS[graph_type](grid_size, target_k, seed=graph_seed)

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
            "_eps":                eps,
            "_graph_seed":         graph_seed,
            "connectivity_failed": True,
        }

    r = run_one_game(
        G,
        seed=cond_seed + rep * 1000 + 1,
        n_moves=n_moves,
        eps=eps,
        colors=COLORS_2P,
        prey_of=None,             # symmetric: each captures the other
        scheduler=scheduler,
        policy_kwargs=policy_kw,
        use_gate=True,
        store_replay=False,
        store_metrics=False,
        store_timeseries=False,   # matches production 2P (store_occ_ts is independent)
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
    r["_eps"]        = eps
    r["_graph_seed"] = graph_seed
    return r


# ─────────────────────────────────────────────
# Sweep runner
# ─────────────────────────────────────────────

def run_eps_sweep(
    params: dict,
    seed_base: int = 42,
    n_workers: int | None = None,
    worker_fn=None,
) -> list[dict]:
    """Run a full epsilon-sensitivity sweep.

    The eps axis sits outside the base-condition loop.  For a given
    (contest, graph_type, grid_size, scheduler, policy, k, rep), the
    graph_seed = cond_seed + rep*1000 is identical across all ε values,
    enabling clean within-sweep cross-ε comparisons on the same graph.

    Irregular graphs (ER, SBM, DC-SBM) get EPS_IRREGULAR (4 values).
    Regular graphs (lattice, torus) get EPS_REGULAR (2 values).
    """
    if worker_fn is None:
        worker_fn = _run_single_eps

    # Base conditions without the eps axis — determines graph_seed
    base_conditions = list(itertools.product(
        params["contests"],
        params["graph_types"],
        params["grid_sizes"],
        params["schedulers"],
        params["policies"],
        params["k_values"],
    ))

    n_reps  = params["n_reps"]
    n_moves = params["n_moves"]

    n_total = sum(
        len(_eps_list(graph_type)) * n_reps
        for (_, graph_type, *_rest) in base_conditions
    )

    if n_workers is None:
        n_workers = min(n_total, os.cpu_count() or 1)

    print(
        f"Eps sweep: {len(base_conditions)} base conditions × (2 or 4 ε) × {n_reps} reps "
        f"= {n_total} total runs  (n_moves={n_moves}, workers={n_workers})"
    )
    print(f"  Irregular graphs: ε ∈ {EPS_IRREGULAR}")
    print(f"  Regular graphs:   ε ∈ {EPS_REGULAR}")

    tasks = []
    for i, (contest, graph_type, grid_size, scheduler, policy_name, target_k) in enumerate(base_conditions):
        cond_seed = seed_base + (i + 1) * 10_000
        for eps in _eps_list(graph_type):
            for rep in range(n_reps):
                tasks.append((
                    contest, graph_type, grid_size, scheduler,
                    target_k, n_moves, rep, cond_seed, policy_name, eps,
                ))

    all_results: list[dict] = []
    t0   = time.perf_counter()
    done = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(worker_fn, t): t for t in tasks}
        for fut in concurrent.futures.as_completed(futures):
            all_results.append(fut.result())
            done += 1
            if done % max(1, n_total // 20) == 0 or done == n_total:
                elapsed    = time.perf_counter() - t0
                remaining  = elapsed / done * (n_total - done) if done < n_total else 0.0
                print(
                    f"  {done}/{n_total} runs done  "
                    f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)",
                    flush=True,
                )

    elapsed = time.perf_counter() - t0
    print(f"\nTotal: {elapsed:.1f}s  ({elapsed / len(all_results):.2f}s/run)\n")
    return all_results


# ─────────────────────────────────────────────
# Summary table (eps-aware)
# ─────────────────────────────────────────────

def _mean_std(vals):
    a = np.array(vals, dtype=float)
    return float(np.mean(a)), float(np.std(a))


def eps_summary_table(all_results: list[dict], is_2p: bool = False) -> None:
    """Print per-(eps × condition) mean ± std of key metrics."""
    skipped = [r for r in all_results if r.get("connectivity_failed")]
    if skipped:
        counts: dict[tuple, int] = defaultdict(int)
        for r in skipped:
            counts[(r["_graph_type"], r["_grid_size"], r.get("_target_k", "?"))] += 1
        print(f"  {len(skipped)} skipped (connectivity_failed):")
        for key, cnt in sorted(counts.items()):
            print(f"    graph={key[0]}  n={key[1]**2}  k={key[2]}  skipped={cnt}")
        print()

    valid  = [r for r in all_results if not r.get("connectivity_failed")]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in valid:
        key = (
            r.get("_eps"),
            r["_contest"],
            r["_graph_type"],
            r["_grid_size"],
            r["_scheduler"],
            r.get("_policy", "?"),
            r.get("_target_k", "?"),
        )
        groups[key].append(r)

    if is_2p:
        header = (
            f"{'eps':>6} {'graph':<8} {'n':>5} {'sched':<14} "
            f"{'policy':<5} {'k':>4} {'ci_raw':>9} {'moves':>7}"
        )
    else:
        header = (
            f"{'eps':>6} {'contest':<10} {'graph':<8} {'n':>5} {'sched':<14} "
            f"{'policy':<5} {'k':>4} {'survivors':>10} {'jain':>8} {'moves':>7}"
        )
    print(header)
    print("-" * len(header))

    for key in sorted(groups):
        eps, contest, graph_type, grid_size, scheduler, policy, target_k = key
        rs    = groups[key]
        n     = grid_size ** 2
        mov_m, _ = _mean_std([r["moves"] for r in rs])

        if is_2p:
            ci_m, ci_s = _mean_std([r.get("coexistence_index", float("nan")) for r in rs])
            print(
                f"{eps:>6.2f} {graph_type:<8} {n:>5} {scheduler:<14} "
                f"{str(policy):<5} {str(target_k):>4} "
                f"{ci_m:>6.3f}±{ci_s:.3f} "
                f"{mov_m:>7.0f}"
            )
        else:
            surv_m, surv_s = _mean_std([r["survivors_count"]     for r in rs])
            jain_m, jain_s = _mean_std([r["jain_fairness_final"] for r in rs])
            print(
                f"{eps:>6.2f} {contest:<10} {graph_type:<8} {n:>5} {scheduler:<14} "
                f"{str(policy):<5} {str(target_k):>4} "
                f"{surv_m:>6.2f}±{surv_s:.2f} "
                f"{jain_m:>5.3f}±{jain_s:.3f} "
                f"{mov_m:>7.0f}"
            )


# ─────────────────────────────────────────────
# Sanity-check helper
# ─────────────────────────────────────────────

def spot_check_eps015(results: list[dict], is_2p: bool = False, n_cells: int = 10) -> None:
    """Print a few ε=0.15 cells for visual comparison against production.

    Because this sweep uses different condition indices than the production sub-sweeps,
    graph seeds differ — record-level matching is not possible.  Use this to verify
    that ε=0.15 distributional statistics look plausible (ballpark of production values)
    before trusting the cross-ε comparisons.
    """
    eps015 = [r for r in results if r.get("_eps") == 0.15 and not r.get("connectivity_failed")]
    if not eps015:
        print("No ε=0.15 records found.")
        return

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in eps015:
        key = (r["_graph_type"], r["_grid_size"], r.get("_target_k"), r.get("_policy"))
        groups[key].append(r)

    print(f"Spot-check: ε=0.15 rows (first {n_cells} cells, compare to production):")
    if is_2p:
        print(f"  {'graph':<8} {'n':>5} {'k':>4} {'policy':<5} {'ci_raw':>9} {'n_reps':>7}")
    else:
        print(f"  {'graph':<8} {'n':>5} {'k':>4} {'policy':<5} {'survivors':>10} {'jain':>8} {'n_reps':>7}")

    for key in sorted(groups)[:n_cells]:
        graph_type, grid_size, target_k, policy = key
        rs = groups[key]
        n  = grid_size ** 2
        if is_2p:
            ci_m, _ = _mean_std([r.get("coexistence_index", float("nan")) for r in rs])
            print(f"  {graph_type:<8} {n:>5} {str(target_k):>4} {str(policy):<5} {ci_m:>9.3f} {len(rs):>7}")
        else:
            surv_m, _ = _mean_std([r["survivors_count"]     for r in rs])
            jain_m, _ = _mean_std([r["jain_fairness_final"] for r in rs])
            print(f"  {graph_type:<8} {n:>5} {str(target_k):>4} {str(policy):<5} {surv_m:>10.2f} {jain_m:>8.3f} {len(rs):>7}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    mode  = sys.argv[1] if len(sys.argv) > 1 else "2p"
    label = sys.argv[2] if len(sys.argv) > 2 else "v1"

    SEED_BASE = 42

    if mode == "2p":
        print("=== EPS SWEEP — Two-player  (~6 656 runs) ===\n")
        results = run_eps_sweep(EPS_2P_PARAMS, seed_base=SEED_BASE,
                                worker_fn=_run_single_2p_eps)
        eps_summary_table(results, is_2p=True)
        spot_check_eps015(results, is_2p=True)
        save_results(results, f"eps_sweep_2p_{label}")

    elif mode == "rps":
        print("=== EPS SWEEP — Three-player RPS  (~6 656 runs) ===\n")
        results = run_eps_sweep(EPS_RPS_PARAMS, seed_base=SEED_BASE,
                                worker_fn=_run_single_eps)
        eps_summary_table(results, is_2p=False)
        spot_check_eps015(results, is_2p=False)
        save_results(results, f"eps_sweep_rps_{label}")

    elif mode == "ava":
        print("=== EPS SWEEP — Three-player AVA  (~6 656 runs) ===\n")
        results = run_eps_sweep(EPS_AVA_PARAMS, seed_base=SEED_BASE,
                                worker_fn=_run_single_eps)
        eps_summary_table(results, is_2p=False)
        spot_check_eps015(results, is_2p=False)
        save_results(results, f"eps_sweep_ava_{label}")

    else:
        print(f"Unknown mode: {mode!r}.  Use: 2p | rps | ava", file=sys.stderr)
        sys.exit(1)
