"""
run_gate_3p_diagnostic.py — Gate on/off comparison, three-player games.

Mirrors run_gate_diagnostic.py (two-player) as closely as possible.

Sweep:
  contests   : rps, all_vs_all
  graphs     : er, sbm, dc_sbm
  schedulers : sync, gillespie_occ
  k values   : 10, 14
  grid sizes : 15 (n=225), 25 (n=625)
  policies   : BBB only
  gate       : True, False
  reps       : 26

Total: 2×3×2×2×2×1×2×26 = 1,248 runs.

Output: results/gate_3p_diag_v1.pkl

Metrics reported per (contest, graph, scheduler, n, k, gate):
  coexist_rate  — genuine coexistence: survivors=3 AND Jain≥0.85
  mean_jain     — mean Jain fairness index
  extinct_rate  — extinction: survivors<3
  bite_rate     — mean gate_bite_rate (gate=ON only)
"""

from __future__ import annotations

import itertools
import os
import time
import concurrent.futures
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_one_game import run_one_game
from rps_sweep import (
    COLORS,
    PREY_OF,
    GRAPH_MAKERS,
    POLICY_CONFIGS,
    _BLOCK_GRAPH_TYPES,
    save_results,
)

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

GATE_3P_PARAMS = dict(
    contests     = ["rps", "all_vs_all"],
    graph_types  = ["er", "sbm", "dc_sbm"],
    schedulers   = ["sync", "gillespie_occ"],
    k_values     = [10, 14],
    grid_sizes   = [15, 25],       # n = 225, 625
    policies     = ["BBB"],
    gate_values  = [True, False],
    n_reps       = 26,
    n_moves      = 5_000,
)

JAIN_COEXIST = 0.85   # genuine coexistence threshold


# ─────────────────────────────────────────────────────────────────────────────
# Single-run worker
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_gate_3p(args):
    """Picklable wrapper: one three-player replicate with configurable gate."""
    (contest, graph_type, grid_size, scheduler,
     target_k, n_moves, rep, seed_base, policy_name, use_gate) = args

    prey_of   = PREY_OF[contest]
    policy_kw = POLICY_CONFIGS[policy_name]
    G = GRAPH_MAKERS[graph_type](grid_size, target_k, seed=seed_base + rep * 1000)

    if G is None:
        return {
            "_contest":            contest,
            "_graph_type":         graph_type,
            "_grid_size":          grid_size,
            "_scheduler":          scheduler,
            "_target_k":           target_k,
            "_policy":             policy_name,
            "_use_gate":           use_gate,
            "_rep":                rep,
            "_n_nodes":            grid_size ** 2,
            "connectivity_failed": True,
        }

    r = run_one_game(
        G,
        seed          = seed_base + rep * 1000 + 1,
        n_moves       = n_moves,
        colors        = COLORS,
        prey_of       = prey_of,
        scheduler     = scheduler,
        policy_kwargs = policy_kw,
        use_gate      = use_gate,
        store_replay        = False,
        store_metrics       = False,
        store_timeseries    = False,
        store_occ_ts        = False,
        store_eff_ts        = False,
        store_final_libs    = False,
        store_interblock    = (graph_type in _BLOCK_GRAPH_TYPES),
        store_R_ts          = False,
        store_sat_state     = False,
    )
    r["_contest"]    = contest
    r["_graph_type"] = graph_type
    r["_grid_size"]  = grid_size
    r["_scheduler"]  = scheduler
    r["_target_k"]   = target_k
    r["_policy"]     = policy_name
    r["_use_gate"]   = use_gate
    r["_rep"]        = rep
    r["_n_nodes"]    = grid_size ** 2
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Sweep runner
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep_gate_3p(params: dict, seed_base: int = 42) -> list[dict]:
    """Parallel sweep over (contest × graph × scheduler × k × n × policy × gate)."""
    conditions = list(itertools.product(
        params["contests"],
        params["graph_types"],
        params["grid_sizes"],
        params["schedulers"],
        params["policies"],
        params["k_values"],
        params["gate_values"],
    ))
    n_reps  = params["n_reps"]
    n_total = len(conditions) * n_reps
    n_workers = min(n_total, os.cpu_count() or 1)

    print(
        f"Sweep: {len(conditions)} conditions × {n_reps} reps "
        f"= {n_total} runs  "
        f"(n_moves={params['n_moves']}, k_values={params['k_values']}, "
        f"workers={n_workers})"
    )

    tasks = []
    for i, (contest, graph_type, grid_size, scheduler,
            policy_name, target_k, use_gate) in enumerate(conditions):
        cond_seed = seed_base + (i + 1) * 10_000
        for rep in range(n_reps):
            tasks.append((
                contest, graph_type, grid_size, scheduler,
                target_k, params["n_moves"], rep, cond_seed, policy_name, use_gate,
            ))

    all_results: list[dict] = []
    t0   = time.perf_counter()
    done = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_single_gate_3p, t): t for t in tasks}
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


# ─────────────────────────────────────────────────────────────────────────────
# Summary tables
# ─────────────────────────────────────────────────────────────────────────────

def print_gate_3p_summary(results: list[dict]) -> None:
    """Per-(contest, graph, sched, n, k, gate): coexist%, Jain, extinct%, bite%."""
    valid  = [r for r in results if not r.get("connectivity_failed")]
    n_fail = len(results) - len(valid)
    print(f"{len(valid)} valid runs ({n_fail} connectivity failures)\n")

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in valid:
        key = (r["_contest"], r["_graph_type"], r["_scheduler"],
               r["_grid_size"] ** 2, r["_target_k"], r["_use_gate"])
        groups[key].append(r)

    hdr = (
        f"{'contest':<11} {'graph':<6} {'sched':<14} {'n':>5} {'k':>4} {'gate':<5}  "
        f"{'total':>6}  {'coexist%':>9}  {'mean_jain':>10}  {'extinct%':>9}  {'bite%':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    # sort: contest, graph, sched, n, k, gate=ON before gate=OFF
    sort_key = lambda k: (k[0], k[1], k[2], k[3], k[4], not k[5])
    prev_cond = None

    for key in sorted(groups, key=sort_key):
        contest, graph, sched, n, k, gate = key
        rs = groups[key]

        cond = (contest, graph, sched, n, k)
        if prev_cond is not None and cond != prev_cond:
            print()
        prev_cond = cond

        gate_str   = "ON " if gate else "OFF"
        survivors  = np.array([r["survivors_count"]     for r in rs])
        jain       = np.array([r["jain_fairness_final"] for r in rs])

        coexist_mask = (survivors == 3) & (jain >= JAIN_COEXIST)
        extinct_mask = survivors < 3

        coexist_rate = coexist_mask.mean()
        mean_jain    = jain.mean()
        extinct_rate = extinct_mask.mean()

        if gate:
            bite_vals = [r.get("gate_bite_rate", float("nan")) for r in rs]
            bite = np.nanmean(bite_vals)
            bite_s = f"{bite:7.1%}"
        else:
            bite_s = f"{'—':>7}"

        print(
            f"{contest:<11} {graph:<6} {sched:<14} {n:>5} {k:>4} {gate_str:<5}  "
            f"{len(rs):>6}  {coexist_rate:>9.1%}  {mean_jain:>10.4f}  "
            f"{extinct_rate:>9.1%}  {bite_s}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = GATE_3P_PARAMS
    total = (len(p["contests"]) * len(p["graph_types"]) * len(p["schedulers"]) *
             len(p["k_values"]) * len(p["grid_sizes"]) * len(p["policies"]) *
             len(p["gate_values"]) * p["n_reps"])

    print("Running three-player gate on/off diagnostic sweep …")
    print(
        f"  {len(p['contests'])} contests × {len(p['graph_types'])} graphs × "
        f"{len(p['schedulers'])} schedulers × {len(p['k_values'])} k × "
        f"{len(p['grid_sizes'])} grid sizes × {len(p['policies'])} policy × "
        f"{len(p['gate_values'])} gate × {p['n_reps']} reps = {total} runs"
    )

    results = run_sweep_gate_3p(GATE_3P_PARAMS, seed_base=42)
    save_results(results, "gate_3p_diag_v1")

    print_gate_3p_summary(results)
