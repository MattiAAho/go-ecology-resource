"""REPLAY of m1's three-player gate diagnostic, SYNCHRONOUS arm, n 625, with the occupancy series stored (MA 11-24:
"rerunning would be best (to avoid any ambiguity)"; plan: dev 2026-11-24_gate_sync_series_plan.md).

Same conditions, seeds and enumeration as run_gate_3p_diagnostic.py (cond_seed = 42 + (i+1)*10,000 over its full
condition list; graph seed + rep*1000, game seed +1); only the synchronous, n 625 tasks; store_occ_ts=True (observation
only: run_one_game.py appends stones / n every 20 turns). RPS tasks first, then all-vs-all. Results saved after every
batch to results/gate_3p_sync_series_v1.pkl (a rerun skips the tasks already on file).

  python run_gate_3p_sync_series.py [--workers 30] [--contest rps|all_vs_all|both]
"""
import argparse
import concurrent.futures
import itertools
import os
import pickle
import time

import run_gate_3p_diagnostic as GD
from run_one_game import run_one_game
from rps_sweep import COLORS, PREY_OF, GRAPH_MAKERS, POLICY_CONFIGS, _BLOCK_GRAPH_TYPES

OUT = "results/gate_3p_sync_series_v1.pkl"
KEY = ("_contest", "_graph_type", "_target_k", "_use_gate", "_rep")


def worker(args):
    (contest, graph_type, grid_size, scheduler, target_k, n_moves, rep, seed_base, policy_name, use_gate) = args
    G = GRAPH_MAKERS[graph_type](grid_size, target_k, seed=seed_base + rep * 1000)
    meta = dict(_contest=contest, _graph_type=graph_type, _grid_size=grid_size, _scheduler=scheduler,
                _target_k=target_k, _policy=policy_name, _use_gate=use_gate, _rep=rep, _n_nodes=grid_size ** 2,
                _cond_seed=seed_base)
    if G is None:
        return dict(connectivity_failed=True, **meta)
    r = run_one_game(G, seed=seed_base + rep * 1000 + 1, n_moves=n_moves, colors=COLORS, prey_of=PREY_OF[contest],
                     scheduler=scheduler, policy_kwargs=POLICY_CONFIGS[policy_name], use_gate=use_gate,
                     store_replay=False, store_metrics=False, store_timeseries=False, store_occ_ts=True,
                     store_eff_ts=False, store_final_libs=False, store_interblock=(graph_type in _BLOCK_GRAPH_TYPES),
                     store_R_ts=False, store_sat_state=False)
    r.update(meta)
    return r


def tasks(contests):
    p = GD.GATE_3P_PARAMS
    conds = list(itertools.product(p["contests"], p["graph_types"], p["grid_sizes"], p["schedulers"], p["policies"],
                                   p["k_values"], p["gate_values"]))
    out = []
    for i, (c, gt, g, s, pol, k, gate) in enumerate(conds):
        if s == "sync" and g == 25 and c in contests:
            out += [(c, gt, g, s, k, p["n_moves"], rep, 42 + (i + 1) * 10_000, pol, gate) for rep in range(p["n_reps"])]
    return sorted(out, key=lambda t: (t[0] != "rps",))


def main(workers, contests):
    have = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else []
    done = {tuple(r[k] for k in KEY) for r in have}
    todo = [t for t in tasks(contests) if (t[0], t[1], t[4], t[9], t[6]) not in done]
    print(f"{len(todo)} games to run ({len(have)} on file), {workers} workers", flush=True)
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(workers) as ex:
        for j, r in enumerate(ex.map(worker, todo, chunksize=1)):
            have.append(r)
            if (j + 1) % workers == 0 or j + 1 == len(todo):
                pickle.dump(have, open(OUT + ".tmp", "wb"))
                os.replace(OUT + ".tmp", OUT)
                el = time.perf_counter() - t0
                print(f"  {j + 1}/{len(todo)} ({el:.0f}s, ~{el / (j + 1) * (len(todo) - j - 1):.0f}s left)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--contest", default="both")
    a = ap.parse_args()
    main(a.workers, ("rps", "all_vs_all") if a.contest == "both" else (a.contest,))
