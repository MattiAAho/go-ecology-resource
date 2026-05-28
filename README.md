# Go-Ecology: Topology-Driven Contest Outcomes in a Resource-Competition Model

Replication code for the paper  
**"Irregular topology drives contest outcomes — coexistence, dominance, and bimodality in graph-based territorial competition"**

---

## Overview

This repository contains all code needed to replicate the simulation experiments,
statistical analyses, and figures reported in the paper.  No pre-computed results are
distributed; everything is generated from scratch by running the scripts in the order
described below.

The model is a spatial resource-competition game played on a Go board whose
neighbourhood structure is replaced by a random graph.  Two-player and three-player
variants are studied across five graph topologies (ER, SBM, DC-SBM, lattice, torus),
five mean-degree values (k = 8, 10, 12, 14, 19), four board sizes (n = 81, 225,
361, 625), and a systematic sweep of stochastic exploration rates — approximately
50 000 simulation runs in total.

---

## Repository structure

```
go_engine.py                    Core Go board simulator (Numba-accelerated)
run_one_game.py                 Single-game runner and coordinate helpers
parallel_benchmark_original_async_new.py   Multiprocessing harness
analysis_helpers.py             Coexistence-index computation (ci_2p, etc.)
rps_sweep.py                    Main sweep runner: two-player, RPS, AVA
run_gate_diagnostic.py          Two-player gate on/off diagnostic sweep
run_gate_3p_diagnostic.py       Three-player gate on/off diagnostic sweep
run_sync_3p_diagnostic.py       Synchronous-scheduler three-player diagnostic
run_eps_sweep.py                Exploration-rate (ε) sensitivity sweep (modes: 2p, rps, ava)
eye_analysis_3p.py              Eye/group structure analysis (three-player)
coexistence_territory_plots.py  Territory plots for genuine coexistence runs
statistical_analysis.py         Bayesian analyses 1-5b (PyMC 5)
plot_*.py                       One script per figure
generate_final_figures.py       Regenerate all seven main figures → figures/final/
generate_supplementary_figures.py   Regenerate S1–S7 → figures/final/
plot_gate_3p_diagnostic.py          Regenerate S8 → figures/fig_gate_3p_diagnostic.pdf
plot_occ_trajectories.py            Regenerate S9 → figures/fig_occ_trajectories_dcsbm_eps.pdf
plot_eps_policy_sensitivity.py      Regenerate S10 → figures/fig_eps_sbm_ci_distribution.pdf
plot_eps_dcsbm_ava_coexistence.py   Regenerate fig7 → figures/fig_eps_dcsbm_ava_coexistence.pdf
test_smoke.py                   Quick sanity-check (< 1 min)
results/                        Created automatically; receives all .pkl outputs
figures/                        Created automatically; receives all .pdf/.png outputs
traces/                         Created automatically; receives MCMC trace .nc files
```

---

## Requirements

Python 3.11 or 3.12 is recommended.  Install dependencies with:

```bash
pip install -r requirements.txt
```

**requirements.txt** (see file in repository):

| Package     | Version used |
|-------------|-------------|
| numpy       | 2.2         |
| scipy       | 1.16        |
| networkx    | 3.5         |
| numba       | 0.61        |
| matplotlib  | 3.10        |
| tqdm        | 4.67        |
| pymc        | 5.25        |
| arviz       | 0.22        |

> **Note on Numba:** `go_engine.py` uses `@njit` for the inner simulation loop.
> The first run will be slower while Numba compiles the kernels; subsequent runs
> use the cached bytecode.

---

## Replication pipeline

Run the steps below in order.  All scripts must be executed from the repository root.
A `results/` directory is created automatically on first run.

### Step 1 — Verify the environment

```bash
python test_smoke.py
```

Runs ~50 two-player games and ~50 three-player games on a small board (n = 81).
Should complete in under one minute and print `All smoke tests passed.`

### Step 2 — Two-player sweep

Generates `results/two_p_all_v1.pkl`  
(13 312 runs: ER + SBM + DC-SBM + lattice + torus, all k and grid sizes, two schedulers)

```bash
python rps_sweep.py full_2p v1
```

Expected wall time: **~9 hours** on the reference machine (32 workers).  
Scale proportionally with available cores.

### Step 3 — Three-player RPS sweep

Generates `results/full_all_v2.pkl`  
(6 656 runs: same topology/k/size grid, RPS contest, Gillespie-like scheduler)

```bash
python rps_sweep.py full_all v2
```

Expected wall time: **3–6 hours** (32 workers).

### Step 4 — Three-player AVA sweep

Generates `results/ava_all_v1.pkl`  
(6 656 runs: same grid, all-vs-all contest)

```bash
python rps_sweep.py full_ava v1
```

Expected wall time: **3–6 hours** (32 workers).

### Step 5 — Exploration-rate (ε) sweep

Generates three files covering two-player, RPS, and AVA competition across
ε ∈ {0.02, 0.04, 0.07, 0.15} (ER/SBM/DC-SBM) and ε ∈ {0.02, 0.15} (lattice/torus),
k = 10 and 14, n = 225 and 625, 26 replicates (6 656 runs each).

```bash
python run_eps_sweep.py 2p v1    # → results/eps_sweep_2p_v1.pkl   (~6 h, 32 workers)
python run_eps_sweep.py rps v1   # → results/eps_sweep_rps_v1.pkl   (~2 h, 32 workers)
python run_eps_sweep.py ava v1   # → results/eps_sweep_ava_v1.pkl   (~4 h, 32 workers)
```

Each mode can be run independently; all three are required for the full figure set.

### Step 6 — Gate diagnostics

Two-player gate diagnostic:

```bash
python run_gate_diagnostic.py
```

Generates `results/gate_diag_v1.pkl`  
(1 248 runs: gate on/off comparison, ER + SBM + DC-SBM, k = 10 and 14)  
Expected wall time: **30–60 minutes**.

Three-player gate diagnostic (required for S8):

```bash
python run_gate_3p_diagnostic.py
```

Generates `results/gate_3p_diag_v1.pkl`  
(2 496 runs: same design, RPS + AVA)  
Expected wall time: **1–2 hours**.

Synchronous-scheduler diagnostic:

```bash
python run_sync_3p_diagnostic.py
```

Generates `results/sync_3p_diag_v1.pkl`  
(416 runs: two-player + RPS + AVA under synchronous scheduling, ER + SBM, k = 10 and 14)  
Expected wall time: **15–30 minutes**.

### Step 7 — Eye / group structure analysis

Generates `results/eye_records_3p.pkl` and diagnostic plots in `results/`.  
Requires Steps 3 and 4 to be complete.

```bash
python eye_analysis_3p.py
```

Expected wall time: **5–15 minutes**.

### Step 8 — Statistical analysis

Runs five Bayesian models (PyMC 5, 4 chains × 2 000 draws each).  
Requires Steps 2–4.  
Saves MCMC traces to `traces/` and writes `statistical_results.md`.

```bash
python statistical_analysis.py
```

Expected wall time: **30–90 minutes** depending on CPU/GPU availability.  
If a trace file already exists it is loaded from cache rather than re-sampled.

### Step 9 — Generate all figures

Requires all previous steps to be complete.

```bash
python generate_final_figures.py           # fig1–fig6 → figures/final/
python generate_supplementary_figures.py   # S1–S7    → figures/final/
python plot_gate_3p_diagnostic.py          # S8       → figures/fig_gate_3p_diagnostic.pdf
python plot_occ_trajectories.py            # S9       → figures/fig_occ_trajectories_dcsbm_eps.pdf
python plot_eps_policy_sensitivity.py      # S10      → figures/fig_eps_sbm_ci_distribution.pdf
python plot_eps_dcsbm_ava_coexistence.py   # fig7     → figures/fig_eps_dcsbm_ava_coexistence.pdf
```

Individual figures can also be regenerated independently, e.g.:

```bash
python plot_monopoly_heatmap.py            # fig1
python plot_bimodality.py                  # fig2
python plot_bimodality_by_k.py             # fig3
python plot_coexistence_vs_k_combined.py   # fig5
```

---

## Approximate total compute

Wall times measured on a 32-worker machine (Python `ProcessPoolExecutor`).
Scale roughly linearly with available cores.

| Step | Runs | Wall time (32 workers) |
|------|------|------------------------|
| Two-player sweep              | 13 312 | ~9 h    |
| RPS sweep                     |  6 656 | ~3–6 h  |
| AVA sweep                     |  6 656 | ~3–6 h  |
| ε sweep — two-player          |  6 656 | ~6 h    |
| ε sweep — RPS                 |  6 656 | ~2 h    |
| ε sweep — AVA                 |  6 656 | ~4 h    |
| Sync 3P diagnostic            |    416 | ~0.25 h |
| Gate on/off diagnostic (2P)   |  1 248 | ~0.5–1 h |
| Gate on/off diagnostic (3P)   |  2 496 | ~1–2 h  |
| Eye analysis                  |   —    | ~0.25 h |
| Statistical analysis          |   —    | ~0.5–1.5 h |
| **Total**                     | **~50 700** | **~30–40 h** |

---

## Reproducing a single game (minimal example)

```python
from run_one_game import run_one_game, compute_coords
import networkx as nx

n = 9        # grid side → 81 nodes
G = nx.grid_2d_graph(n, n)
G = nx.convert_node_labels_to_integers(G)
coords = compute_coords(n)

result = run_one_game(
    G, coords,
    scheduler="gillespie_occ",
    seed=42,
)
print(result["black_occupancy"], result["coexistence_index"])
```

---

## Citation

*(to be added on publication)*

---

## License

This code is released under the [GNU General Public License v3.0](LICENSE).
