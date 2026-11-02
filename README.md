# Go-Ecology: a Go-inspired territorial competition model on graphs

Replication code for the paper  
**"Inspired by Go: playing ecology with complex topologies – two- and three-species territorial competition on regular and irregular graphs"**

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
plot_*.py                       One script per figure (see "Figures" below)
generate_final_figures.py       Regenerate manuscript Figures 2–7 → figures/final/fig1–fig6.pdf
generate_supplementary_figures.py   Regenerate S1, S2, S4–S7 → figures/final/
plot_model_schematic.py             Regenerate Figure 1 → figures/final/fig_schematic.pdf
plot_eps_dcsbm_ava_coexistence.py   Regenerate Figure 8 → figures/fig_eps_dcsbm_ava_coexistence.pdf
plot_gate_extended.py               Regenerate S3 → figures/final/S3.pdf
plot_gate_3p_diagnostic.py          Regenerate S8 → figures/fig_gate_3p_diagnostic.pdf
plot_occ_trajectories.py            Regenerate S9 → figures/fig_occ_trajectories_dcsbm_eps.pdf
plot_eps_policy_sensitivity.py      Regenerate S10 → figures/fig_eps_sbm_ci_distribution.pdf
plot_convergence_si.py              Regenerate S11 → figures/final/S11.pdf
analysis_time_to_limbo.py           Settling times of the outcome statistic (Methods; printed)
describe_m1_text_numbers.py         Numbers quoted in the text and tables that no figure shows (printed)
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
python generate_final_figures.py           # Figures 2–7 → figures/final/fig1–fig6.pdf
python generate_supplementary_figures.py   # S1, S2, S4–S7 → figures/final/
python plot_model_schematic.py             # Figure 1 → figures/final/fig_schematic.pdf
python plot_eps_dcsbm_ava_coexistence.py   # Figure 8 → figures/fig_eps_dcsbm_ava_coexistence.pdf
python plot_gate_extended.py               # S3       → figures/final/S3.pdf
python plot_gate_3p_diagnostic.py          # S8       → figures/fig_gate_3p_diagnostic.pdf
python plot_occ_trajectories.py            # S9       → figures/fig_occ_trajectories_dcsbm_eps.pdf
python plot_eps_policy_sensitivity.py      # S10      → figures/fig_eps_sbm_ci_distribution.pdf
python plot_convergence_si.py              # S11      → figures/final/S11.pdf
```

S11 reads the per-family files that Steps 2–4 write beside the combined ones (`full_er_sbm_v2.pkl`,
`full_dcsbm_v2.pkl`, `full_fixed_v2.pkl`, and the same for `ava_*_v1` and `two_p_*_v1`).

Individual figures can also be regenerated independently, e.g.:

```bash
python plot_monopoly_heatmap.py            # Figure 2 (fig1)
python plot_bimodality.py                  # Figure 3 (fig2)
python plot_bimodality_by_k.py             # Figure 4 (fig3)
python plot_coexistence_vs_k_combined.py   # Figure 6 (fig5)
```

### Step 10 — Numbers quoted in the text

Requires Steps 2–4 and the synchronous diagnostic of Step 6.

```bash
python describe_m1_text_numbers.py         # → results/m1_text_numbers.txt
python analysis_time_to_limbo.py           # settling times (Methods), printed
```

`describe_m1_text_numbers.py` prints the numbers in the text and tables that no figure shows: the weakest species'
share of the final stones on regular and irregular graphs (Table 4, Claim 3), the matched cells under both
schedulers, the two-player dominance rates per cell (Table 4, Claim 1; Figure 2 caption), the two-player dominance
and segregation shares (Figure 4 caption), and the regular–irregular contrast within each policy mix.

---

## Figures

The file names in `figures/final/` predate the manuscript's numbering: `fig1`–`fig6` are Figures 2–7.
Every file listed below is the version in the manuscript.

| Manuscript | Script | File in `figures/final/` |
|---|---|---|
| Figure 1 | `plot_model_schematic.py` | `fig_schematic.pdf` |
| Figure 2 | `plot_monopoly_heatmap.py` | `fig1.pdf` |
| Figure 3 | `plot_bimodality.py` | `fig2.pdf` |
| Figure 4 | `plot_bimodality_by_k.py` | `fig3.pdf` |
| Figure 5 | `generate_final_figures.py` (drawn there) | `fig4.pdf` |
| Figure 6 | `plot_coexistence_vs_k_combined.py` | `fig5.pdf` |
| Figure 7 | `plot_gate_diagnostic.py` | `fig6.pdf` |
| Figure 8 | `plot_eps_dcsbm_ava_coexistence.py` | `fig7.pdf` (copy of `figures/fig_eps_dcsbm_ava_coexistence.pdf`) |
| S1 | `plot_policy_sensitivity.py` | `S1.pdf` |
| S2 | `plot_capture_summary.py` | `S2.pdf` |
| S3 | `plot_gate_extended.py` | `S3.pdf` |
| S4, S5 | `plot_eye_diagnostic.py` | `S4.pdf`, `S5.pdf` |
| S6, S7 | `coexistence_territory_plots.py` | `S6.pdf`, `S7.pdf` |
| S8 | `plot_gate_3p_diagnostic.py` | `S8.pdf` (copy of `figures/fig_gate_3p_diagnostic.pdf`) |
| S9 | `plot_occ_trajectories.py` | `S9.pdf` (copy of `figures/fig_occ_trajectories_dcsbm_eps.pdf`) |
| S10 | `plot_eps_policy_sensitivity.py` | `S10.pdf` (copy of `figures/fig_eps_sbm_ci_distribution.pdf`) |
| S11 | `plot_convergence_si.py` | `S11.pdf` |

Figures 2–7 and S1, S2, S4–S7 are written to `figures/final/` by the two generators; the others are written by their
own scripts.

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

## Seeds

Every sweep in `rps_sweep.py` derives its seeds from the condition and replicate indices (`seed_base = 42` in all
production runs):

- condition seed = `seed_base + 10_000 * (i + 1)`, where `i` is the condition's index in the sweep's parameter product
  (contest × graph type × grid size × scheduler × policy × mean degree, mean degree innermost);
- graph seed = condition seed + `1_000 * rep`; game seed = graph seed + 1.

ER, SBM and DC-SBM graphs are drawn from the graph seed and redrawn (up to 20 tries) until connected; otherwise the
replicate is skipped. Lattice and torus graphs are fixed and ignore the seed, so their replicates differ only in the
game seed.

**Seed values recur between neighbouring conditions.** With 26 replicates the replicate stride (1,000) overruns the
condition stride (10,000): replicate `r` of condition `i` has the same seeds as replicate `r − 10` of condition `i + 1`
and replicate `r − 20` of condition `i + 2` (e.g. graph seed 20,042 is condition 0, replicate 10, and condition 1,
replicate 0). Neighbouring conditions usually differ only in mean degree, so such pairs of ER / SBM / DC-SBM graphs are
drawn from the same random numbers at different edge probabilities, and their games start from the same random
stream. Within a condition every seed is distinct, and every reported quantity is computed per condition. To identify
a graph, use (graph type, grid size, target k, graph seed), never the seed alone.

**Seeds restart in every sweep.** `i` counts from 0 in each call of `run_sweep`, and every production sweep uses
`seed_base = 42`, so two sweeps that list the same graph type, grid size and mean degree at the same index play on the
same graphs. Re-running a sweep with the same `seed_base` and settings reproduces its games exactly: it does not add
independent replicates, and pooling the two runs counts each game twice. For new replicates, choose a `seed_base`
whose seeds do not overlap any already used.

---

## Citation

*(to be added on publication)*

---

## License

This code is released under the [GNU General Public License v3.0](LICENSE).
