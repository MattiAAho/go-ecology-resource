"""
run_gate_diagnostic.py — Gate on/off comparison, two-player games.

Sweep:
  graphs     : er, sbm, dc_sbm
  schedulers : sync, gillespie_occ
  k values   : 10, 14
  grid sizes : 15 (n=225), 25 (n=625)
  policies   : BB (symmetric invasive baseline)
  gate       : True, False
  reps       : 26

Total: 3×2×2×2×1×2×26 = 1248 runs.

Output: results/gate_diag_v1.pkl

Primary prediction:
  segregation runs (ci > 0.85) should have higher capture_rate_per_move with
  gate=ON than with gate=OFF; dominance runs (ci < 0.10) should be unaffected.
  If indistinguishable across both regimes, the prediction fails.

Metrics reported per (graph, scheduler, n, k, gate), split by regime:
  capture_rate_per_move — primary prediction
  moves                 — game length (stones placed)
  gate_bite_rate        — fraction of attempted captures blocked (gate=ON only)
  coexistence_index     — stones-normalised CI (new formula, direct from result)
  regime counts         — dom / seg / other proportions
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
    GRAPH_MAKERS,
    POLICY_CONFIGS_2P,
    _BLOCK_GRAPH_TYPES,
    COLORS_2P,
    save_results,
)

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

GATE_DIAG_PARAMS = dict(
    graph_types  = ["er", "sbm", "dc_sbm"],
    schedulers   = ["sync", "gillespie_occ"],
    k_values     = [10, 14],
    grid_sizes   = [15, 25],       # n = 225, 625
    policies     = ["BB"],
    gate_values  = [True, False],
    n_reps       = 26,
    n_moves      = 5_000,
)

CI_SEG = 0.85   # segregation threshold
CI_DOM = 0.10   # dominance threshold

# ─────────────────────────────────────────────────────────────────────────────
# Single-run worker
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_gate_diag(args):
    """Picklable wrapper: one two-player replicate with configurable gate."""
    (graph_type, grid_size, scheduler,
     target_k, n_moves, rep, seed_base, policy_name, use_gate) = args

    policy_kw = POLICY_CONFIGS_2P[policy_name]
    G = GRAPH_MAKERS[graph_type](grid_size, target_k, seed=seed_base + rep * 1000)

    if G is None:
        return {
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
        seed        = seed_base + rep * 1000 + 1,
        n_moves     = n_moves,
        colors      = COLORS_2P,
        prey_of     = None,           # symmetric: each captures the other
        scheduler   = scheduler,
        policy_kwargs = policy_kw,
        use_gate    = use_gate,
        store_replay      = False,
        store_metrics     = False,
        store_timeseries  = False,    # gate_bite_rate is always tracked regardless
        store_occ_ts      = False,
        store_eff_ts      = False,
        store_final_libs  = False,
        store_interblock  = (graph_type in _BLOCK_GRAPH_TYPES),
        store_R_ts        = False,
        store_sat_state   = False,
    )
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

def run_sweep_gate(params: dict, seed_base: int = 42) -> list[dict]:
    """Parallel sweep over (graph × scheduler × k × n × policy × gate) conditions."""
    conditions = list(itertools.product(
        params["graph_types"],
        params["grid_sizes"],
        params["schedulers"],
        params["policies"],
        params["k_values"],
        params["gate_values"],
    ))
    n_reps   = params["n_reps"]
    n_total  = len(conditions) * n_reps
    n_workers = min(n_total, os.cpu_count() or 1)

    print(
        f"Sweep: {len(conditions)} conditions × {n_reps} reps "
        f"= {n_total} runs  "
        f"(n_moves={params['n_moves']}, k_values={params['k_values']}, "
        f"workers={n_workers})"
    )

    tasks = []
    for i, (graph_type, grid_size, scheduler, policy_name, target_k, use_gate) in enumerate(conditions):
        cond_seed = seed_base + (i + 1) * 10_000
        for rep in range(n_reps):
            tasks.append((
                graph_type, grid_size, scheduler,
                target_k, params["n_moves"], rep, cond_seed, policy_name, use_gate,
            ))

    all_results: list[dict] = []
    t0 = time.perf_counter()
    done = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_single_gate_diag, t): t for t in tasks}
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
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

REGIMES = [
    ("dom ", lambda ci: ci < CI_DOM),
    ("seg ", lambda ci: ci > CI_SEG),
    ("mid ", lambda ci: (ci >= CI_DOM) & (ci <= CI_SEG)),
]


def print_gate_comparison(results: list[dict]) -> None:
    """Per-(graph, sched, n, k, gate) × regime: capture rate, moves, bite rate, ci mean."""
    valid = [r for r in results if not r.get("connectivity_failed")]
    n_fail = len(results) - len(valid)
    print(f"{len(valid)} valid runs ({n_fail} connectivity failures)\n")

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in valid:
        key = (r["_graph_type"], r["_scheduler"], r["_grid_size"] ** 2,
               r["_target_k"], r["_use_gate"])
        groups[key].append(r)

    hdr = (
        f"{'graph':<6} {'sched':<14} {'n':>5} {'k':>4} {'gate':<5}  "
        f"{'regime':<5}  {'n':>5}  "
        f"{'cap_rate':>9}  {'moves':>7}  {'bite%':>7}  {'ci':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    # sort: graph, sched, n, k, then gate=ON before gate=OFF
    sort_key = lambda k: (k[0], k[1], k[2], k[3], not k[4])
    prev_cond = None

    for key in sorted(groups, key=sort_key):
        graph, sched, n, k, gate = key
        rs = groups[key]
        ci_arr = np.array([r["coexistence_index"] for r in rs])

        cond = (graph, sched, n, k)
        if prev_cond is not None and cond != prev_cond:
            print()          # blank line between conditions
        prev_cond = cond

        gate_str = "ON " if gate else "OFF"

        for label, mask_fn in REGIMES:
            mask = mask_fn(ci_arr)
            sub  = [r for r, m in zip(rs, mask) if m]
            if not sub:
                continue
            cap  = np.mean([r["capture_rate_per_move"] for r in sub])
            mov  = np.mean([r["moves"]                 for r in sub])
            bite = np.mean([r["gate_bite_rate"]        for r in sub]) if gate else float("nan")
            ci_m = np.mean([r["coexistence_index"]     for r in sub])
            bite_s = f"{bite:7.1%}" if gate else f"{'—':>7}"
            print(
                f"{graph:<6} {sched:<14} {n:>5} {k:>4} {gate_str:<5}  "
                f"{label:<5}  {len(sub):>5}  "
                f"{cap:>9.4f}  {mov:>7.0f}  {bite_s}  {ci_m:>6.3f}"
            )


def print_regime_proportions(results: list[dict]) -> None:
    """Print proportion of dom/seg/mid runs per (graph, sched, n, k, gate)."""
    valid = [r for r in results if not r.get("connectivity_failed")]

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in valid:
        key = (r["_graph_type"], r["_scheduler"], r["_grid_size"] ** 2,
               r["_target_k"], r["_use_gate"])
        groups[key].append(r)

    print(f"\n{'graph':<6} {'sched':<14} {'n':>5} {'k':>4} {'gate':<5}  "
          f"{'total':>6}  {'dom%':>6}  {'seg%':>6}  {'mid%':>6}")
    print("-" * 70)

    sort_key = lambda k: (k[0], k[1], k[2], k[3], not k[4])
    for key in sorted(groups, key=sort_key):
        graph, sched, n, k, gate = key
        rs  = groups[key]
        ci  = np.array([r["coexistence_index"] for r in rs])
        n_d = np.sum(ci < CI_DOM)
        n_s = np.sum(ci > CI_SEG)
        n_m = len(rs) - n_d - n_s
        gate_str = "ON " if gate else "OFF"
        print(
            f"{graph:<6} {sched:<14} {n:>5} {k:>4} {gate_str:<5}  "
            f"{len(rs):>6}  {n_d/len(rs):>6.1%}  {n_s/len(rs):>6.1%}  {n_m/len(rs):>6.1%}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = GATE_DIAG_PARAMS
    n_g  = len(p["graph_types"])
    n_sc = len(p["schedulers"])
    n_k  = len(p["k_values"])
    n_n  = len(p["grid_sizes"])
    n_po = len(p["policies"])
    n_ga = len(p["gate_values"])
    n_r  = p["n_reps"]
    total = n_g * n_sc * n_k * n_n * n_po * n_ga * n_r

    print("Running gate on/off diagnostic sweep …")
    print(
        f"  {n_g} graphs × {n_sc} schedulers × {n_k} k × {n_n} grid sizes × "
        f"{n_po} policy × {n_ga} gate × {n_r} reps = {total} runs"
    )

    results = run_sweep_gate(GATE_DIAG_PARAMS, seed_base=42)
    save_results(results, "gate_diag_v1")

    print_gate_comparison(results)
    print_regime_proportions(results)
