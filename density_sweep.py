"""
density_sweep.py — ER and SBM density sweep, two-species go-ecology model.
Grid sizes 9, 15, 19, 25; target mean degrees 8/10/12/14/19; 26 replicates each.
"""

import time, gzip, pickle, math
import networkx as nx
import numpy as np
from scipy.optimize import brentq

from run_one_game import run_one_game, compute_coords
from parallel_benchmark_original_async_new import run_benchmark_parallel_original

# ── density configurations ────────────────────────────────────────────────

TARGET_DEGREES = {
    "deg19": 19,
    "deg14": 14,
    "deg12": 12,
    "deg10": 10,
    "deg08":  8,
}

def er_p_for_degree(grid_size, target_k):
    n = grid_size ** 2
    return target_k / (n - 1)

def sbm_P_for_degree(grid_size, target_k, modularity_ratio=8.0):
    n = grid_size ** 2
    unit = n / 6
    s = [3*unit, 2*unit, unit]
    r = 1.0 / modularity_ratio
    def mean_deg(pw):
        pb = pw * r
        P = [[pw,       pb,       pb*0.5 ],
             [pb,       pw*0.75,  pb     ],
             [pb*0.5,   pb,       pw*0.875]]
        d = sum(s[i]*s[j]*P[i][j] for i in range(3) for j in range(3)) / n
        return d
    pw = brentq(lambda pw: mean_deg(pw) - target_k, 1e-6, 1.0)
    pb = pw * r
    return [[pw,       pb,       pb*0.5 ],
            [pb,       pw*0.75,  pb     ],
            [pb*0.5,   pb,       pw*0.875]]

GRID_SIZES = [9, 15, 19, 25]

# ── graph builders ────────────────────────────────────────────────────────

def build_er(grid_size, er_p, sbm_seed=1, max_tries=10):
    n = grid_size ** 2
    for attempt in range(max_tries):
        G = nx.erdos_renyi_graph(n, er_p, seed=sbm_seed + attempt)
        if nx.is_connected(G):
            if attempt > 0:
                print(f"  ER connected after {attempt+1} attempts (n={n}, p={er_p})")
            return G
    raise RuntimeError(
        f"Could not generate connected ER graph after {max_tries} tries "
        f"(n={n}, p={er_p:.4f}). Increase p or max_tries."
    )

def build_sbm(grid_size, P, sbm_seed=1, max_tries=10):
    n = grid_size ** 2
    unit = n / 6
    size_1 = int(round(3 * unit))
    size_2 = int(round(2 * unit))
    size_3 = n - size_1 - size_2
    sizes = [size_1, size_2, size_3]
    assert all(s > 0 for s in sizes), f"Block sizes invalid: {sizes}"
    for attempt in range(max_tries):
        G = nx.stochastic_block_model(sizes, P, seed=sbm_seed + attempt)
        if nx.is_connected(G):
            if attempt > 0:
                print(f"  SBM connected after {attempt+1} attempts (n={n})")
            return G
    G_last = nx.stochastic_block_model(sizes, P, seed=sbm_seed)
    mean_deg = 2 * G_last.number_of_edges() / n
    raise RuntimeError(
        f"Could not generate connected SBM after {max_tries} tries "
        f"(n={n}, mean_deg≈{mean_deg:.2f}). P matrix may be too sparse."
    )

def mean_degree(G):
    return 2 * G.number_of_edges() / G.number_of_nodes()

# ── shared run config ─────────────────────────────────────────────────────

policy_balanced = dict(
    avoid_eye_filling=True,
    w_own_adj=3.0, w_libs=2.0, w_size=0.5,
    w_captures=10.0, w_opp_pressure=6.0,
    self_atari_penalty=20.0, w_degree=2.0, w_center=0.0,
)
policy_invasive = dict(
    w_overlap=+1.5, w_opp_pressure=8.0, w_captures=12.0,
    self_atari_penalty=15.0, w_territory=0.05, territory_depth=2,
    w_center=0.0, normalize_degree=True,
)
policy_coexist = dict(
    w_overlap=-1.5,
    w_territory=0.2,
    territory_depth=2,
    w_center=0.0,
    normalize_degree=True,
)
policyA = dict(policy_balanced); policyA.update(policy_coexist)
policyB = dict(policy_balanced); policyB.update(policy_invasive)

policy_kwargs = {"black": policyB, "white": policyA}

RUN_KWARGS = dict(
    run_one_game_fn=run_one_game,
    n_runs=26,
    n_moves=2500,
    eps=0.02,
    random_every=0,
    ko=True,
    policy_kwargs=policy_kwargs,
    keep_replay_k=13,
    keep_metrics_k=1,
    keep_timeseries_k=5,
    store_cap_ts=True,
    store_occ_ts=True,
    occ_every=25,
    use_gate=True,
    gate_delta=3.0,
    gate_hops=2,
    resource_alpha=0.01,
    resource_beta=0.005,
    resource_init=1.0,
    store_R_ts=True,
    R_every=25,
    scheduler="gillespie_occ",
    n_jobs=26,
    chunksize=1,
    start_method="fork",
    checkpoint_every=10,
    store_sat_state=True,
)

# ── sweep ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    all_results = {}

    for grid_size in GRID_SIZES:

        print(f"\n{'═'*60}")
        print(f"  Grid size: {grid_size}×{grid_size}  ({grid_size**2} nodes)")
        print(f"{'═'*60}")

        # ── ER sweep ──────────────────────────────────────────────────────
        for tag, target_k in TARGET_DEGREES.items():
            er_p = er_p_for_degree(grid_size, target_k)
            tag_full = f"er_{tag}"
            print(f"\n── {tag_full}  target_k={target_k}  p={er_p:.4f}  grid={grid_size} ──")
            try:
                G_er = build_er(grid_size, er_p)
            except RuntimeError as e:
                print(f"  SKIP: {e}")
                continue

            exp_key = (grid_size, tag_full)
            ckpt    = f"ckpt_BA_g{grid_size}_{tag_full}.pkl.gz"
            out     = f"results_BA_g{grid_size}_{tag_full}.pkl.gz"
            print(f"  mean degree = {mean_degree(G_er):.2f}")

            graphs = {"er": G_er}
            coords = {"er": compute_coords(G_er, seed=1, layout="spring")}

            t0 = time.time()
            res = run_benchmark_parallel_original(
                graphs=graphs,
                coords=coords,
                checkpoint_path=ckpt,
                **RUN_KWARGS,
            )
            dt = time.time() - t0
            print(f"  wall time: {dt/60:.1f} min")

            all_results[exp_key] = res
            with gzip.open(out, "wb") as f:
                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  saved → {out}")

        # ── SBM sweep ─────────────────────────────────────────────────────
        for tag, target_k in TARGET_DEGREES.items():
            P = sbm_P_for_degree(grid_size, target_k)
            tag_full = f"sbm_{tag}"
            exp_key = (grid_size, tag_full)
            ckpt    = f"ckpt_BA_g{grid_size}_{tag_full}.pkl.gz"
            out     = f"results_BA_g{grid_size}_{tag_full}.pkl.gz"
            print(f"\n── {tag_full}  target_k={target_k}  grid={grid_size} ──")
            try:
                G_sbm = build_sbm(grid_size, P)
            except RuntimeError as e:
                print(f"  SKIP: {e}")
                continue

            print(f"  mean degree = {mean_degree(G_sbm):.2f}")

            graphs = {"sbm": G_sbm}
            coords = {"sbm": compute_coords(G_sbm, seed=1, layout="spring")}

            t0 = time.time()
            res = run_benchmark_parallel_original(
                graphs=graphs,
                coords=coords,
                checkpoint_path=ckpt,
                **RUN_KWARGS,
            )
            dt = time.time() - t0
            print(f"  wall time: {dt/60:.1f} min")

            all_results[exp_key] = res
            with gzip.open(out, "wb") as f:
                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  saved → {out}")

    print(f"\n{'═'*60}")
    print("Sweep complete.")
    print(f"Keys in all_results: {sorted(all_results.keys())}")

    # ── quick coexistence summary ─────────────────────────────────────────
    print(f"\n{'Exp key':<30} {'topo':<6} {'CI mean':>8} {'CI sd':>7}")
    print("─" * 55)
    for (gs, tag), res in sorted(all_results.items()):
        for topo, runs in res.items():
            cis = [r["coexistence_index"] for r in runs if "coexistence_index" in r]
            if cis:
                print(f"g{gs}_{tag:<25} {topo:<6} "
                      f"{np.mean(cis):>8.3f} {np.std(cis):>7.3f}")
