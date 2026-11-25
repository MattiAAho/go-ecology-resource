"""
run_sync_3p_regular.py — m1's synchronous three-player diagnostic, extended to the regular graphs (MA 2026-11-23:
"we run the test now"). Plan: dev repo 2026-11-23_m1_sync3p_regular_plan.md (committed before any game).

The same sweep as run_sync_3p_diagnostic.py (sync_3p_diag_v1.pkl: ER, SBM, k 10 and 14), on lattice and torus:
  contests   : rps, all_vs_all
  scheduler  : sync
  graphs     : lattice, torus          (k nominal 4, ignored by the makers, as FIXED_TOPO_PARAMS)
  grid sizes : 15 (n=225), 25 (n=625)
  policies   : BBB, AAA
  reps       : 13
  n_moves    : 5,000; gate on (rps_sweep._run_single); seed_base 42 (m1's formula, as the diagnostic)

Total: 2 x 2 x 2 x 2 x 13 = 208 runs.  Output: results/sync_3p_regular_v1.pkl (refuses to overwrite).

Run:
    python run_sync_3p_regular.py
"""
from pathlib import Path

from rps_sweep import run_sweep, save_results, _run_single

PARAMS = dict(
    contests    = ["rps", "all_vs_all"],
    graph_types = ["lattice", "torus"],
    grid_sizes  = [15, 25],
    schedulers  = ["sync"],
    policies    = ["BBB", "AAA"],
    k_values    = [4],
    n_moves     = 5_000,
    n_reps      = 13,
)

if __name__ == "__main__":
    out = Path("results/sync_3p_regular_v1.pkl")
    assert not out.exists(), f"{out} exists; refusing to overwrite"
    results = run_sweep(PARAMS, seed_base=42, worker_fn=_run_single)
    save_results(results, "sync_3p_regular_v1")
    valid = [r for r in results if not r.get("connectivity_failed")]
    print(f"{len(valid)} valid runs of {len(results)}")
