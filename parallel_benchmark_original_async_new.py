# Matti Aho, 17.3.2026
# parallel_benchmark_original_async_new.py
import os
import math
import multiprocessing as mp
from typing import Dict, Any, List, Tuple, Optional
from tqdm.auto import tqdm
import gzip
import pickle
import time
import zlib
# import numpy as np

_WORK_GRAPHS = None
_WORK_COORDS = None
_WORK_RUN_ONE_GAME = None
_WORK_KW = None
_WORK_SCHEDULER = None

def _init_worker(graphs, coords, run_one_game_fn, shared_kwargs, scheduler):
    global _WORK_GRAPHS, _WORK_COORDS, _WORK_RUN_ONE_GAME, _WORK_KW, _WORK_SCHEDULER
    _WORK_GRAPHS = graphs
    _WORK_COORDS = coords
    _WORK_RUN_ONE_GAME = run_one_game_fn
    _WORK_KW = shared_kwargs
    _WORK_SCHEDULER = scheduler

def _worker_task(task: Tuple[str, int, int, bool, bool, bool]) -> Dict[str, Any]:
    gname, run_idx, seed, keep_replay, keep_metrics, keep_timeseries = task
    G = _WORK_GRAPHS[gname]
    coords = _WORK_COORDS.get(gname, None) if _WORK_COORDS else None

    # force timeseries collection if any TS-derived logging is requested
    # if _WORK_KW.get("store_cap_ts", False) or _WORK_KW.get("store_occ_ts", False):
    #     keep_timeseries = True

    if (_WORK_KW.get("store_cap_ts", False)
        or _WORK_KW.get("store_occ_ts", False)
        or _WORK_KW.get("store_R_ts", False)):
        keep_timeseries = True

    r = _WORK_RUN_ONE_GAME(
        G,
        coords=coords,
        seed=seed,
        store_replay=keep_replay,
        store_metrics=keep_metrics,
        store_timeseries=keep_timeseries,
        store_eff_ts=keep_timeseries,
        scheduler=_WORK_SCHEDULER,
        **_WORK_KW
    )

    # Strip heavy payload unless requested.
    # final_state and coords are kept unconditionally — they are cheap and
    # needed for downstream regime/eye analysis regardless of replay storage.
    if not keep_replay:
        r.pop("history", None)
        r.pop("final_R", None)

    if not keep_metrics:
        r.pop("metrics_history", None)

    if not keep_timeseries:
        for k in (
            "atari_counts",
            "capture_sizes",
            "pass_ts",
            "pass_with_legal_ts",
            "pass_no_legal_ts",
            "passes_failed_gate_ts",
            "illegal_ts",
            "color_ts",
            "captures_per_turn",
            "occ_b_ts", "occ_w_ts", "occ_t_ts",
            "eff_b_ts", "eff_w_ts",
            "R_mean_ts", "R_occ_mean_ts", "R_empty_mean_ts","R_occ_mass_ts",
            "R_every",
        ):
            r.pop(k, None)

    r["graph"] = gname
    r["run"] = run_idx
    r["seed"] = seed
    return r


def run_benchmark_parallel_original(
    *,
    graphs: Dict[str, Any],
    coords: Dict[str, Any],
    run_one_game_fn,
    n_runs: int,
    # passed to run_one_game_fn:
    n_moves: int = 400,
    eps: float = 0.15,
    random_every: int = 0,
    ko: bool = False,
    policy_kwargs: Optional[dict] = None,
    pass_threshold=None,
    scheduler="sync",
    
    # resouce gate
    use_gate: bool = False,
    gate_delta: float = 1.0,
    gate_hops: int = 2,
    
    # resource dynamics knobs (if your run_one_game supports them)
    resource_alpha: float = 0.0,
    resource_beta: float = 0.0,
    resource_init: float = 1.0,
    
    # NEW: long-run logging knobs
    store_cap_ts: bool = False,
    store_occ_ts: bool = False,
    occ_every: int = 10,

    # R control knobs
    store_R_ts: bool = False,
    R_every: int = 25,

    # output control
    store_sat_state: bool = True,
    keep_replay_k: int = 1,
    keep_metrics_k: int = 0,
    keep_timeseries_k: int = 0,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 10,
    # parallelism
    n_jobs: Optional[int] = None,
    chunksize: Optional[int] = None,
    start_method: str = "fork",
) -> Dict[str, List[Dict[str, Any]]]:

    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() or 1)

    gnames = list(graphs.keys())

    if chunksize is None:
        chunksize = max(1, int(math.ceil((len(gnames) * n_runs) / (10 * n_jobs))))
  
    shared_kwargs = dict(
        n_moves=n_moves,
        eps=eps,
        random_every=random_every,
        ko=ko,
        policy_kwargs=policy_kwargs,
        pass_threshold=pass_threshold,
    
        # long-run logging
        store_cap_ts=store_cap_ts,
        store_occ_ts=store_occ_ts,
        occ_every=occ_every,

        # resource field logging
        store_R_ts=store_R_ts,
        R_every=R_every,

        # gate + resource dynamics
        use_gate=use_gate,
        gate_delta=gate_delta,
        gate_hops=gate_hops,
        resource_alpha=resource_alpha,
        resource_beta=resource_beta,
        resource_init=resource_init,

        # for eye analysis
        store_sat_state=store_sat_state, 
    )

    # build tasks with deterministic seeds
    tasks: List[Tuple[str, int, int, bool, bool, bool]] = []
    for i in range(n_runs):
        for gname in gnames:
            base = zlib.crc32(gname.encode()) & 0xFFFF_FFFF
            seed = (base + 1_000_003 * i) & 0xFFFF_FFFF
            tasks.append((gname, i, int(seed),
                          i < keep_replay_k,
                          i < keep_metrics_k,
                          i < keep_timeseries_k))
    
    # tasks: List[Tuple[str, int, int, bool, bool, bool]] = []
    # for gname in gnames:
    #     base = zlib.crc32(gname.encode()) & 0xFFFF_FFFF # base = (hash(gname) & 0xFFFF_FFFF)
    #     for i in range(n_runs):
    #         seed = (base + 1_000_003 * i) & 0xFFFF_FFFF
    #         tasks.append((
    #             gname,
    #             i,
    #             int(seed),
    #             i < keep_replay_k,
    #             i < keep_metrics_k,
    #             i < keep_timeseries_k,
    #         ))
    
    # rng = np.random.default_rng(0)
    # rng.shuffle(tasks)
    
    ctx = mp.get_context(start_method)
    with ctx.Pool(
        processes=n_jobs,
        initializer=_init_worker,
        maxtasksperchild=20,
        initargs=(graphs, coords, run_one_game_fn, shared_kwargs, scheduler),
    ) as pool:
        out = []
        it = pool.imap_unordered(_worker_task, tasks, chunksize=chunksize)
        
        last_ckpt_t = time.time()   # <-- MUST be before loop, same indentation as loop
        for k, r in enumerate(tqdm(it, total=len(tasks), desc="Simulations", smoothing=0.05), start=1):
            out.append(r)

            # checkpoint every N results (and also not too frequently)
            if checkpoint_path and (k % checkpoint_every == 0) and (time.time() - last_ckpt_t >= 2.0):
                tmp_path = checkpoint_path + ".tmp"

                payload = {
                    "meta": {
                        "completed": k,
                        "total": len(tasks),
                        "n_runs": n_runs,
                        "gnames": gnames,
                        "n_moves": n_moves,
                        "eps": eps,
                        "random_every": random_every,
                        "ko": ko,
                        "keep_replay_k": keep_replay_k,
                        "keep_metrics_k": keep_metrics_k,
                        "keep_timeseries_k": keep_timeseries_k,
                        "scheduler": scheduler,
                        "n_jobs": n_jobs,
                        "chunksize": chunksize,
                        "start_method": start_method,
                        "timestamp": time.time(),
                    },
                    "partial_out": out,
                }

                with gzip.open(tmp_path, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

                # atomic replace so you never end up with a corrupted checkpoint
                os.replace(tmp_path, checkpoint_path)
                last_ckpt_t = time.time()   

    # regroup
    results = {g: [] for g in gnames}
    for r in out:
        results[r["graph"]].append(r)
    for g in gnames:
        results[g].sort(key=lambda d: d["run"])
    return results
