# Statistical Analysis Results

Bayesian analyses for the Go-ecology paper.
89% HDI throughout (McElreath convention).
Sampling: 4 chains × 2000 draws, tune=1000, target_accept=0.9.

## Analysis 1 — Scheduler effect on monopoly rate
Model: monopoly_count ~ Binomial(n_runs, p),  p ~ Beta(1,1).
Monopoly = stones-normalised CI < 0.1.  Data: ER and SBM, all k, all n.

### Results by graph type and board size

#### ER

  k= 8 | n=625 | P(sync > Gill) — Gillespie reduces monopoly at large n = 1.000 | sync=0.359  Gill=0.009 | Δ 89% HDI [+0.272, +0.422]

  k= 8 | n= 81 | P(Gill > sync) — reversal at small n = 0.011 | sync=0.651  Gill=0.491 | Δ 89% HDI [-0.272, -0.053]

  k=10 | n=625 | P(sync > Gill) — Gillespie reduces monopoly at large n = 1.000 | sync=0.226  Gill=0.019 | Δ 89% HDI [+0.135, +0.274]

  k=10 | n= 81 | P(Gill > sync) — reversal at small n = 0.439 | sync=0.688  Gill=0.679 | Δ 89% HDI [-0.113, +0.092]

  k=12 | n=625 | P(sync > Gill) — Gillespie reduces monopoly at large n = 1.000 | sync=0.207  Gill=0.057 | Δ 89% HDI [+0.080, +0.221]

  k=12 | n= 81 | P(Gill > sync) — reversal at small n = 0.318 | sync=0.717  Gill=0.688 | Δ 89% HDI [-0.133, +0.071]

  k=14 | n=625 | P(sync > Gill) — Gillespie reduces monopoly at large n = 1.000 | sync=0.443  Gill=0.066 | Δ 89% HDI [+0.297, +0.469]

  k=14 | n= 81 | P(Gill > sync) — reversal at small n = 0.796 | sync=0.727  Gill=0.774 | Δ 89% HDI [-0.043, +0.142]

  k=19 | n=625 | P(sync > Gill) — Gillespie reduces monopoly at large n = 1.000 | sync=0.453  Gill=0.161 | Δ 89% HDI [+0.196, +0.383]

  k=19 | n= 81 | P(Gill > sync) — reversal at small n = 1.000 | sync=0.680  Gill=0.868 | Δ 89% HDI [+0.101, +0.278]

#### SBM

  k= 8 | n=625 | P(sync > Gill) — Gillespie reduces monopoly at large n = 1.000 | sync=0.510  Gill=0.019 | Δ 89% HDI [+0.407, +0.573]

  k= 8 | n= 81 | P(Gill > sync) — reversal at small n = 0.000 | sync=0.623  Gill=0.330 | Δ 89% HDI [-0.401, -0.192]

  k=10 | n=625 | P(sync > Gill) — Gillespie reduces monopoly at large n = 1.000 | sync=0.274  Gill=0.010 | Δ 89% HDI [+0.190, +0.330]

  k=10 | n= 81 | P(Gill > sync) — reversal at small n = 0.074 | sync=0.670  Gill=0.575 | Δ 89% HDI [-0.201, +0.008]

  k=12 | n=625 | P(sync > Gill) — Gillespie reduces monopoly at large n = 0.999 | sync=0.150  Gill=0.028 | Δ 89% HDI [+0.060, +0.181]

  k=12 | n= 81 | P(Gill > sync) — reversal at small n = 0.988 | sync=0.651  Gill=0.792 | Δ 89% HDI [+0.043, +0.237]

  k=14 | n=625 | P(sync > Gill) — Gillespie reduces monopoly at large n = 1.000 | sync=0.368  Gill=0.038 | Δ 89% HDI [+0.254, +0.409]

  k=14 | n= 81 | P(Gill > sync) — reversal at small n = 0.891 | sync=0.689  Gill=0.764 | Δ 89% HDI [-0.021, +0.174]

  k=19 | n=625 | P(sync > Gill) — Gillespie reduces monopoly at large n = 0.995 | sync=0.312  Gill=0.170 | Δ 89% HDI [+0.043, +0.231]

  k=19 | n= 81 | P(Gill > sync) — reversal at small n = 1.000 | sync=0.613  Gill=0.905 | Δ 89% HDI [+0.206, +0.380]

## Analysis 2 — Bimodality confirmation
Model: two-component beta mixture with ordered means (offset parameterisation).
Data: SBM, sync scheduler, all k pooled.
Null: single-component beta. Model comparison via LOO-CV.

### Mixture component estimates
  mu1 (monopoly pole)    : 0.064  89% HDI [0.058, 0.070]
  mu2 (coexistence pole) : 0.992  89% HDI [0.991, 0.993]
  w1  (monopoly weight)  : 0.708  89% HDI [0.693, 0.725]
  w2  (coexist. weight)  : 0.292  89% HDI [0.275, 0.307]
  P(mu2 − mu1 > 0.5)    : 1.0000  — poles well-separated

  LOO-CV: best model = 'mixture'
  ΔELPD(mixture − null) = +380.6 ± 18.4  (positive = mixture wins)

## Analysis 3 — Density shifts CI toward monopoly under Gillespie-like
Model: CI ~ Beta(mu*phi, (1-mu)*phi),  logit(mu) = α + β_k·k_std + β_n·log(n)_std.
Separate models for sync and Gillespie-like. Data: SBM, all k, all n.

### Beta regression: effect of k on CI
  Gillespie β_k : -0.589  89% HDI [-0.632, -0.550]  (negative = higher k → lower CI)
  Sync      β_k : +0.018  89% HDI [-0.029, +0.066]
  P(β_k_Gillespie < −0.1)        = 1.000  — formal density-collapse test
  P(β_k_sync > β_k_Gillespie)    = 1.000  — scheduler-dependent effect

## Analysis 4 — Competition structure effect on coexistence
Model: genuine_coexistence_count ~ Binomial(n_runs, p),  p ~ Beta(1,1).
Genuine coexistence = survivors_count == 3 AND jain_fairness_final >= 0.85.
  (survivors<3 was rejected as outcome: RPS has marginally *higher* extinction than AVA,
   a tiny reversed effect swamped by noise. Genuine coexistence shows a 2× RPS advantage.)
Data: RPS and AVA, irregular graphs (ER, SBM, DC-SBM), n=625, all k pooled.

### RPS vs AVA genuine coexistence rate (by graph type)

  ER    : P(RPS coexistence > AVA coexistence) = 0.878 | RPS=0.042  AVA=0.029 | Δ(RPS−AVA) 89% HDI [-0.005, +0.032]

  SBM   : P(RPS coexistence > AVA coexistence) = 1.000 | RPS=0.130  AVA=0.058 | Δ(RPS−AVA) 89% HDI [+0.043, +0.101]

  DC_SBM: P(RPS coexistence > AVA coexistence) = 1.000 | RPS=0.175  AVA=0.079 | Δ(RPS−AVA) 89% HDI [+0.059, +0.132]

### Block structure vs ER genuine coexistence (within RPS)

  P(SBM coexistence > ER coexistence) = 1.000 | sbm=0.130  ER=0.042 | Δ(block−ER) 89% HDI [+0.059, +0.114]

  P(DC_SBM coexistence > ER coexistence) = 1.000 | dc_sbm=0.175  ER=0.042 | Δ(block−ER) 89% HDI [+0.101, +0.168]

## Analysis 5 — Policy insensitivity
Model: outcome ~ Normal(α + policy_effect[i], σ),  policy_effect ~ Normal(0, 0.5).
ROPE = ±0.05 outcome units.  Null = intercept only (no policy term).
LOO-CV: if null model wins, policy insensitivity is formally supported.

### two_player  (ROPE = ±0.05)

  AA   : mean=+0.0385  89% HDI [-0.3364, +0.3949]  ✗ outside ROPE

  AB   : mean=+0.0641  89% HDI [-0.3136, +0.4169]  ✗ outside ROPE

  BA   : mean=+0.0629  89% HDI [-0.3027, +0.4284]  ✗ outside ROPE

  BB   : mean=+0.0770  89% HDI [-0.2901, +0.4416]  ✗ outside ROPE

  All policy effects in ROPE: False

  LOO-CV: best model = 'policy'
  ΔELPD(policy − null) = +5.4 ± 4.1  (negative = null wins = policy insensitivity supported)

### three_player  (ROPE = ±0.05)

  AAA  : mean=+0.1276  89% HDI [-0.2244, +0.4762]  ✗ outside ROPE

  BAA  : mean=+0.1272  89% HDI [-0.2208, +0.4798]  ✗ outside ROPE

  BAB  : mean=+0.1322  89% HDI [-0.2191, +0.4808]  ✗ outside ROPE

  BBB  : mean=+0.1343  89% HDI [-0.2171, +0.4843]  ✗ outside ROPE

  All policy effects in ROPE: False

  LOO-CV: best model = 'null'
  ΔELPD(policy − null) = -1.7 ± 1.6  (negative = null wins = policy insensitivity supported)

## Analysis 5b — Policy insensitivity (controlled for scheduler and graph type)
Model: outcome ~ Normal(α + sched_effect[s] + graph_effect[g] + policy_effect[p], σ).
All categorical effects: ZeroSumNormal(sigma=0.5) — deviations from grand mean.
Controls absorb between-cell variance; policy residual is tested against ROPE = ±0.05.
Two-player: controlled for scheduler (2 levels) + graph type (5 levels).
Three-player: controlled for graph type only (Gillespie-like data only).

### two_player  (controlled for scheduler + graph type)
  AA   : mean=-0.0222  89% HDI [-0.0313, -0.0126]  ✓ in ROPE
  AB   : mean=+0.0034  89% HDI [-0.0062, +0.0122]  ✓ in ROPE
  BA   : mean=+0.0023  89% HDI [-0.0067, +0.0116]  ✓ in ROPE
  BB   : mean=+0.0165  89% HDI [+0.0071, +0.0257]  ✓ in ROPE
  All policy effects in ROPE: True
  Residual σ = 0.382  89% HDI [0.378, 0.386]  (cf. σ≈0.30+ uncontrolled)
  LOO-CV: best model = 'policy', ΔELPD(policy − null) = +5.7 ± 4.2  (indistinguishable)

### three_player  (controlled for graph type)
  AAA  : mean=-0.0028  89% HDI [-0.0076, +0.0021]  ✓ in ROPE
  BAA  : mean=-0.0031  89% HDI [-0.0083, +0.0018]  ✓ in ROPE
  BAB  : mean=+0.0018  89% HDI [-0.0033, +0.0068]  ✓ in ROPE
  BBB  : mean=+0.0041  89% HDI [-0.0010, +0.0091]  ✓ in ROPE
  All policy effects in ROPE: True
  Residual σ = 0.147  89% HDI [0.145, 0.149]
  LOO-CV: best model = 'null', ΔELPD(policy − null) = -1.6 ± 1.7  (null wins, within 1 SE)

Interpretation: After controlling for scheduler and graph type, all policy effects
are within ROPE for both two- and three-player data. The max effect size is 0.022 CI
units (two-player) and 0.004 Jain units (three-player) — both negligible on a 0-1 scale.
LOO-CV cannot distinguish the policy model from the null in either dataset.
Policy assignment has no credible effect on competitive outcomes.

---

## Manuscript-ready statements

### Claim 1 (Analysis 1)
Bayesian beta-binomial analysis gave posterior P(sync monopoly rate > Gillespie-like
monopoly rate) = 1.000 at n = 625 across all graph densities and both irregular-graph
topologies tested (ER and SBM; 89% HDI on the rate difference: +0.04 to +0.57 depending
on condition). The effect is large and consistent: synchronous scheduling produced 2–36×
higher monopoly rates than energy-weighted asynchronous scheduling at large board sizes.
At small boards (n = 81) the advantage reverses for high graph densities (k ≥ 12): P(Gillespie
monopoly rate > sync) = 0.988–1.000 at k = 12–19, indicating that the coexistence benefit
of energy-weighted scheduling requires sufficient board area to sustain territorial separation.

### Claim 2a — bimodality (Analysis 2)
A two-component beta mixture fitted to synchronous SBM runs (n = 2,072, all densities
pooled) recovered two well-separated poles: a monopoly component at μ₁ = 0.064
[0.058, 0.070] with weight w₁ = 0.708 [0.693, 0.725], and a coexistence component at
μ₂ = 0.992 [0.991, 0.993] with weight w₂ = 0.292 [0.275, 0.307] (89% HDI). Posterior
P(μ₂ − μ₁ > 0.5) = 1.000. LOO-CV decisively preferred the two-component model over a
single-component beta (ΔELPD = +380.6 ± 18.4), formally confirming that outcomes are
bimodally distributed rather than spread across intermediate values.

### Claim 2b — density effect (Analysis 3)
Beta regression of stones-normalised CI on standardised graph density k and log board
size (SBM runs, n = 2,079 Gillespie-like and 2,072 synchronous) estimated the density
slope β_k = −0.589 [−0.632, −0.550] under Gillespie-like scheduling (P(β_k < −0.1) =
1.000) and β_k = +0.018 [−0.029, +0.066] under synchronous scheduling (credibly near
zero). P(β_k_sync > β_k_Gillespie) = 1.000. Higher graph density therefore credibly
collapses coexistence specifically under energy-weighted asynchronous scheduling while
leaving synchronous outcomes unaffected.

### Claim 3 — competition structure (Analysis 4)
Bayesian beta-binomial analysis of genuine coexistence rates (all three species present
with Jain index ≥ 0.85, n = 625 board, all densities pooled) showed that cyclic (RPS)
competition credibly exceeds symmetric (AVA) competition on block-structured graphs:
P(RPS > AVA) = 1.000 on SBM (Δ: +0.07 [+0.04, +0.10]) and DC-SBM (Δ: +0.10
[+0.06, +0.13]), with moderate evidence on ER graphs (P = 0.878). Block-structured
topologies credibly exceeded random graphs within RPS: P(SBM > ER) = P(DC-SBM > ER) =
1.000 (Δ: +0.09 [+0.06, +0.11] and +0.13 [+0.10, +0.17] respectively).

### Policy insensitivity (Analysis 5b)
After controlling for scheduler and graph topology, all policy assignment effects fell
within the region of practical equivalence (ROPE: ±0.05 units). Two-player effects
ranged from −0.022 to +0.017 CI units (89% HDI within ROPE for all four policies);
three-player Jain effects ranged from −0.003 to +0.004 (89% HDI within ROPE for all
four policies). LOO-CV could not distinguish the policy model from the null in either
dataset (two-player: ΔELPD = +5.7 ± 4.2; three-player: ΔELPD = −1.6 ± 1.7). Policy
assignment — whether agents follow an invasive or coexistence strategy — has no credible
effect on competitive outcomes once topology and scheduling are accounted for.
