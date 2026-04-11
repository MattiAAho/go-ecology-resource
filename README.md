# Go-Ecology: Topology-Driven Contest Outcomes in a Resource-Competition Model

Replication code for the paper  
**"Irregular topology drives contest outcomes in a Go-based ecological resource model"**

---

## Overview

This repository contains all code needed to replicate the simulation experiments,
statistical analyses, and figures reported in the paper.  No pre-computed results are
distributed; everything is generated from scratch by running the scripts in the order
described below.

The model is a spatial resource-competition game played on a Go board whose
neighbourhood structure is replaced by a random graph.  Two-player and three-player
variants are studied across five graph topologies (ER, SBM, DC-SBM, lattice, torus),
five mean-degree values (k = 8, 10, 12, 14, 19), and four board sizes (n = 81, 225,
361, 625).

---

## Repository structure

```
go_engine.py                    Core Go board simulator (Numba-accelerated)
run_one_game.py                 Single-game runner and coordinate helpers
parallel_benchmark_original_async_new.py   Multiprocessing harness
analysis_helpers.py             Coexistence-index computation (ci_2p, etc.)
rps_sweep.py                    Main sweep runner: two-player, RPS, AVA
run_gate_diagnostic.py          Gate on/off diagnostic sweep
eye_analysis_3p.py              Eye/group structure analysis (three-player)
coexistence_territory_plots.py  Territory plots for genuine coexistence runs
statistical_analysis.py         Bayesian analyses 1-5b (PyMC 5)
plot_*.py                       One script per figure
generate_final_figures.py       Regenerate all six main figures → figures/final/
generate_supplementary_figures.py   Regenerate S1–S7 → figures/final/
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

Expected wall time: **4–8 hours** on a modern 8-core machine.

### Step 3 — Three-player RPS sweep

Generates `results/full_all_v2.pkl`  
(6 656 runs: same topology/k/size grid, RPS contest, Gillespie + sync)

```bash
python rps_sweep.py full_all v2
```

Expected wall time: **6–12 hours**.

### Step 4 — Three-player AVA sweep

Generates `results/ava_all_v1.pkl`  
(6 656 runs: same grid, all-vs-all contest)

```bash
python rps_sweep.py full_ava v1
```

Expected wall time: **6–12 hours**.

### Step 5 — Gate diagnostic

Generates `results/gate_diag_v1.pkl`  
(1 248 runs: gate on/off comparison, ER + SBM + DC-SBM, k = 10 and 14)

```bash
python run_gate_diagnostic.py
```

Expected wall time: **30–60 minutes**.

### Step 6 — Eye / group structure analysis

Generates `results/eye_records_3p.pkl` and diagnostic plots in `results/`.  
Requires Steps 3 and 4 to be complete.

```bash
python eye_analysis_3p.py
```

Expected wall time: **5–15 minutes**.

### Step 7 — Statistical analysis

Runs five Bayesian models (PyMC 5, 4 chains × 2 000 draws each).  
Requires Steps 2–4.  
Saves MCMC traces to `traces/` and writes `statistical_results.md`.

```bash
python statistical_analysis.py
```

Expected wall time: **30–90 minutes** depending on CPU/GPU availability.  
If a trace file already exists it is loaded from cache rather than re-sampled.

### Step 8 — Generate all figures

Requires all previous steps to be complete.

```bash
python generate_final_figures.py       # fig1–fig6 → figures/final/
python generate_supplementary_figures.py   # S1–S7  → figures/final/
```

Individual figures can also be regenerated independently, e.g.:

```bash
python plot_monopoly_heatmap.py        # fig1
python plot_bimodality.py              # fig2
python plot_bimodality_by_k.py         # fig3
python plot_coexistence_vs_k_combined.py   # fig5
```

---

## Approximate total compute

| Step | Runs | Wall time (8-core) |
|------|------|--------------------|
| Two-player sweep     | 13 312 | 4–8 h  |
| RPS sweep            |  6 656 | 6–12 h |
| AVA sweep            |  6 656 | 6–12 h |
| Gate diagnostic      |  1 248 | 0.5–1 h |
| Eye analysis         |  —     | 0.25 h  |
| Statistical analysis |  —     | 0.5–1.5 h |
| **Total**            | **27 872** | **~20–35 h** |

All sweeps are parallelised with Python `multiprocessing` and scale roughly linearly
with the number of available cores.

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

*(to be confirmed by the authors)*
