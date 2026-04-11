"""
run_sync_3p_diagnostic.py — Sync three-player sanity check.

Sweep:
  contests   : rps, all_vs_all
  scheduler  : sync
  graphs     : er, sbm
  k values   : 10, 14
  grid sizes : 15 (n=225), 25 (n=625)
  policies   : BBB, AAA
  reps       : 13

Total: 2×2×2×2×2×13 = 416 runs. Expected ~10-15 min.

Output: results/sync_3p_diag_v1.pkl

What to look for:
  - Survivors counts vs Gillespie at matched conditions
  - Does sync produce more extinction at smaller n (as in two-player)?
  - Does SBM >= ER topology ordering hold under sync?

Run:
    python run_sync_3p_diagnostic.py
"""

import pickle
from pathlib import Path

from rps_sweep import run_sweep, save_results, _run_single

DIAG_PARAMS = dict(
    contests    = ["rps", "all_vs_all"],
    graph_types = ["er", "sbm"],
    grid_sizes  = [15, 25],
    schedulers  = ["sync"],
    policies    = ["BBB", "AAA"],
    k_values    = [10, 14],
    n_moves     = 5_000,
    n_reps      = 13,
)

if __name__ == "__main__":
    print("Running sync three-player diagnostic sweep …")
    print("  2 contests × 2 graphs × 2 grid sizes × 1 scheduler "
          "× 2 policies × 2 k × 13 reps = 416 runs")

    results = run_sweep(DIAG_PARAMS, seed_base=42, worker_fn=_run_single)
    save_results(results, "sync_3p_diag_v1")

    # ── quick summary ──────────────────────────────────────────────────────
    import numpy as np
    from collections import defaultdict

    valid = [r for r in results if not r.get("connectivity_failed")]
    print(f"\n{len(valid)} valid runs ({len(results)-len(valid)} connectivity failures)\n")

    # Group by (contest, graph_type, grid_size, policy, target_k)
    groups = defaultdict(list)
    for r in valid:
        key = (r["_contest"], r["_graph_type"], r["_grid_size"],
               r["_policy"], r.get("_target_k","?"))
        groups[key].append(r)

    header = f"{'contest':<12} {'graph':<5} {'n':>5} {'policy':<6} {'k':>4}  " \
             f"{'surv':>10}  {'jain':>8}  {'ext%':>6}"
    print(header)
    print("-" * len(header))

    for key in sorted(groups):
        contest, gtype, gs, policy, k = key
        rs = groups[key]
        surv = np.array([r["survivors_count"]     for r in rs])
        jain = np.array([r["jain_fairness_final"]  for r in rs])
        ext  = np.mean(surv < 3)
        print(f"{contest:<12} {gtype:<5} {gs**2:>5} {policy:<6} {str(k):>4}  "
              f"{surv.mean():>5.2f}±{surv.std():.2f}  "
              f"{jain.mean():>5.3f}±{jain.std():.3f}  "
              f"{ext:>5.0%}")
