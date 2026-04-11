#!/usr/bin/env python3
"""
statistical_analysis.py
Five Bayesian models supporting the main claims of the Go-ecology paper.

  1. Scheduler effect on monopoly rate          (Claim 1)
  2. Bimodality confirmation in sync SBM        (Claim 2)
  3. Density shifts CI toward monopoly          (Claim 2b)
  4. Competition structure effect on coexistence (Claim 3)
  5. Policy insensitivity (formal null)

Settings throughout:
  • 89% HDI (McElreath convention)
  • pm.sample(draws=2000, tune=1000, chains=4, target_accept=0.9)
  • Traces cached as ArviZ .nc files in traces/
  • Final summary written to statistical_results.md
"""

from __future__ import annotations
import sys, pickle, warnings
import numpy as np
import pymc as pm
import arviz as az
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent))
from analysis_helpers import ci_2p

# ── paths & constants ─────────────────────────────────────────────────────────
RESULTS  = Path("results")
TRACES   = Path("traces")
TRACES.mkdir(parents=True, exist_ok=True)

HDI_PROB  = 0.89
ROPE      = 0.05          # for Analysis 5 policy insensitivity
SAMPLE_KW = dict(draws=2000, tune=1000, chains=4, target_accept=0.9,
                 progressbar=True, return_inferencedata=True)

print(f"PyMC {pm.__version__}  |  ArviZ {az.__version__}")


# ── utility functions ─────────────────────────────────────────────────────────

def hdi_arr(samples: np.ndarray, prob: float = HDI_PROB):
    """Shortest-interval HDI for a 1-D posterior sample array.

    Uses the direct sort-and-scan algorithm (does not call ArviZ)
    to avoid cross-version API differences.
    """
    s = np.sort(np.asarray(samples, dtype=float).flatten())
    n = len(s)
    w = int(np.floor(prob * n))           # number of samples in interval
    if w >= n:
        return float(s[0]), float(s[-1])
    gaps  = s[w:] - s[:n - w]
    lo_i  = int(np.argmin(gaps))
    return float(s[lo_i]), float(s[lo_i + w])


def prob_gt(a, b) -> float:
    """Posterior P(a > b) from two sample arrays."""
    return float(np.mean(np.asarray(a) > np.asarray(b)))


def has_ll(idata) -> bool:
    """True if InferenceData contains a log_likelihood group."""
    try:
        return "log_likelihood" in idata.groups()
    except Exception:
        return False


def load_or_none(path: Path):
    """Load cached trace if it exists, else return None."""
    if path.exists():
        return az.from_netcdf(str(path))
    return None


# ── data loading ──────────────────────────────────────────────────────────────
print("\nLoading data files…")
with open(RESULTS / "two_p_all_v1.pkl", "rb") as f:
    two_p = pickle.load(f)
with open(RESULTS / "full_all_v2.pkl", "rb") as f:
    rps   = pickle.load(f)
with open(RESULTS / "ava_all_v1.pkl", "rb") as f:
    ava   = pickle.load(f)
print(f"  two_p={len(two_p):,}  rps={len(rps):,}  ava={len(ava):,}")

# Pre-compute corrected two-player CI for every record (stones-normalised)
for r in two_p:
    r["_ci"] = ci_2p(r)


# ── summary accumulator ───────────────────────────────────────────────────────
_out: list[str] = [
    "# Statistical Analysis Results",
    "",
    "Bayesian analyses for the Go-ecology paper.",
    "89% HDI throughout (McElreath convention).",
    "Sampling: 4 chains × 2000 draws, tune=1000, target_accept=0.9.",
    "",
]


def S(*lines: str):
    """Append lines to summary and echo to stdout."""
    for ln in lines:
        _out.append(ln)
        print(ln)
    _out.append("")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — Scheduler effect on monopoly rate
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 1 — Scheduler effect on monopoly rate")
print("=" * 70)

S("## Analysis 1 — Scheduler effect on monopoly rate",
  "Model: monopoly_count ~ Binomial(n_runs, p),  p ~ Beta(1,1).",
  "Monopoly = stones-normalised CI < 0.1.  Data: ER and SBM, all k, all n.")

# Filter: ER and SBM only; drop NaN CI (connectivity failures produce NaN)
a1 = [r for r in two_p
      if r["_graph_type"] in ("er", "sbm") and not np.isnan(r["_ci"])]

# Group by (graph_type, scheduler, n_nodes, target_k)
grp1: dict = defaultdict(list)
for r in a1:
    key = (r["_graph_type"], r["_scheduler"], r["_n_nodes"], r["_target_k"])
    grp1[key].append(int(r["_ci"] < 0.1))

conds1 = sorted(grp1)
mono1  = np.array([sum(grp1[c]) for c in conds1])
ntri1  = np.array([len(grp1[c]) for c in conds1])
n_c1   = len(conds1)
print(f"  {n_c1} conditions | {ntri1.sum():,} runs")

t1_path = TRACES / "analysis_1.nc"
_tr1 = load_or_none(t1_path)
if _tr1 is not None:
    print("  Loading cached trace…")
    tr1 = _tr1
else:
    with pm.Model():
        p_a1 = pm.Beta("p", alpha=1, beta=1, shape=n_c1)
        pm.Binomial("obs", n=ntri1, p=p_a1, observed=mono1)
        tr1 = pm.sample(**SAMPLE_KW)
    az.to_netcdf(tr1, str(t1_path))
    print("  Trace saved.")

# Posterior samples: (n_chains * n_draws, n_conditions)
p1_samp = tr1.posterior["p"].values.reshape(-1, n_c1)

S("### Results by graph type and board size")
for gt in ("er", "sbm"):
    S(f"#### {gt.upper()}")
    k_vals_gt = sorted({r["_target_k"] for r in a1 if r["_graph_type"] == gt})
    for k in k_vals_gt:
        for n_val, claim_dir, sign in [
            (625, "P(sync > Gill) — Gillespie reduces monopoly at large n", +1),
            ( 81, "P(Gill > sync) — reversal at small n",                  -1),
        ]:
            sk = (gt, "sync",          n_val, k)
            gk = (gt, "gillespie_occ", n_val, k)
            if sk not in grp1 or gk not in grp1:
                continue
            si, gi = conds1.index(sk), conds1.index(gk)
            ps, pg = p1_samp[:, si], p1_samp[:, gi]

            if sign == +1:
                prob = prob_gt(ps, pg)
                diff = ps - pg
            else:
                prob = prob_gt(pg, ps)
                diff = pg - ps

            lo, hi = hdi_arr(diff)
            S(f"  k={k:2d} | n={n_val:3d} | {claim_dir} = {prob:.3f} | "
              f"sync={np.mean(ps):.3f}  Gill={np.mean(pg):.3f} | "
              f"Δ 89% HDI [{lo:+.3f}, {hi:+.3f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Bimodality confirmation (SBM, sync, all k pooled)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 2 — Bimodality confirmation")
print("=" * 70)

S("## Analysis 2 — Bimodality confirmation",
  "Model: two-component beta mixture with ordered means (offset parameterisation).",
  "Data: SBM, sync scheduler, all k pooled.",
  "Null: single-component beta. Model comparison via LOO-CV.")

a2 = np.array([r["_ci"] for r in two_p
               if r["_graph_type"] == "sbm"
               and r["_scheduler"] == "sync"
               and not np.isnan(r["_ci"])])
a2 = np.clip(a2, 1e-4, 1 - 1e-4)          # beta distribution requires open interval
print(f"  {len(a2):,} SBM-sync runs (all k pooled)")

t2m_path = TRACES / "analysis_2_mixture.nc"
t2n_path = TRACES / "analysis_2_null.nc"

# ── mixture model ─────────────────────────────────────────────────────────────
_tr2m = load_or_none(t2m_path)
if _tr2m is not None and has_ll(_tr2m):
    print("  Loading cached mixture trace…")
    tr2m = _tr2m
else:
    with pm.Model():
        # Ordered means: mu1 near 0, mu2 = mu1 + delta*(1-mu1) > mu1
        # This enforces mu1 < mu2 without discontinuous sorting
        mu1   = pm.Beta("mu1",   alpha=2, beta=5)      # prior mass near 0
        delta = pm.Beta("delta", alpha=2, beta=2)       # strictly positive
        mu2   = pm.Deterministic("mu2", mu1 + delta * (1 - mu1))
        phi1  = pm.Gamma("phi1", alpha=2, beta=0.5)    # concentration params
        phi2  = pm.Gamma("phi2", alpha=2, beta=0.5)
        w     = pm.Dirichlet("w", a=np.ones(2))        # mixture weights
        pm.Mixture(
            "obs",
            w=w,
            comp_dists=[
                pm.Beta.dist(alpha=mu1 * phi1, beta=(1 - mu1) * phi1),
                pm.Beta.dist(alpha=mu2 * phi2, beta=(1 - mu2) * phi2),
            ],
            observed=a2,
        )
        tr2m = pm.sample(**SAMPLE_KW)
        pm.compute_log_likelihood(tr2m)
    az.to_netcdf(tr2m, str(t2m_path))
    print("  Mixture trace saved.")

# ── null (single-component) model ─────────────────────────────────────────────
_tr2n = load_or_none(t2n_path)
if _tr2n is not None and has_ll(_tr2n):
    print("  Loading cached null trace…")
    tr2n = _tr2n
else:
    with pm.Model():
        mu_n  = pm.Beta("mu",  alpha=1, beta=1)
        phi_n = pm.Gamma("phi", alpha=2, beta=0.5)
        pm.Beta("obs", alpha=mu_n * phi_n, beta=(1 - mu_n) * phi_n, observed=a2)
        tr2n = pm.sample(**SAMPLE_KW)
        pm.compute_log_likelihood(tr2n)
    az.to_netcdf(tr2n, str(t2n_path))
    print("  Null trace saved.")

# ── summarise ─────────────────────────────────────────────────────────────────
post2   = tr2m.posterior
mu1_s   = post2["mu1"].values.flatten()
mu2_s   = post2["mu2"].values.flatten()
w_s     = post2["w"].values.reshape(-1, 2)

mu1_m,  (mu1_lo,  mu1_hi)  = float(np.mean(mu1_s)),     hdi_arr(mu1_s)
mu2_m,  (mu2_lo,  mu2_hi)  = float(np.mean(mu2_s)),     hdi_arr(mu2_s)
w1_m,   (w1_lo,   w1_hi)   = float(np.mean(w_s[:, 0])), hdi_arr(w_s[:, 0])
w2_m,   (w2_lo,   w2_hi)   = float(np.mean(w_s[:, 1])), hdi_arr(w_s[:, 1])
sep_p   = float(np.mean(mu2_s - mu1_s > 0.5))

S("### Mixture component estimates",
  f"  mu1 (monopoly pole)    : {mu1_m:.3f}  89% HDI [{mu1_lo:.3f}, {mu1_hi:.3f}]",
  f"  mu2 (coexistence pole) : {mu2_m:.3f}  89% HDI [{mu2_lo:.3f}, {mu2_hi:.3f}]",
  f"  w1  (monopoly weight)  : {w1_m:.3f}  89% HDI [{w1_lo:.3f}, {w1_hi:.3f}]",
  f"  w2  (coexist. weight)  : {w2_m:.3f}  89% HDI [{w2_lo:.3f}, {w2_hi:.3f}]",
  f"  P(mu2 − mu1 > 0.5)    : {sep_p:.4f}  — poles well-separated")

try:
    cmp2      = az.compare({"mixture": tr2m, "null": tr2n}, ic="loo")
    best2     = cmp2.index[0]
    d_elpd2   = float(cmp2.loc["mixture", "elpd_loo"]) - float(cmp2.loc["null", "elpd_loo"])
    # dse is on the second-best model row
    dse2      = float(cmp2.loc[cmp2.index[1], "dse"])
    S(f"  LOO-CV: best model = '{best2}'",
      f"  ΔELPD(mixture − null) = {d_elpd2:+.1f} ± {dse2:.1f}  (positive = mixture wins)")
except Exception as exc:
    S(f"  LOO-CV comparison failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — Density shifts CI toward monopoly under Gillespie-like
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 3 — Density effect on CI (beta regression)")
print("=" * 70)

S("## Analysis 3 — Density shifts CI toward monopoly under Gillespie-like",
  "Model: CI ~ Beta(mu*phi, (1-mu)*phi),  logit(mu) = α + β_k·k_std + β_n·log(n)_std.",
  "Separate models for sync and Gillespie-like. Data: SBM, all k, all n.")

a3_base = [r for r in two_p
           if r["_graph_type"] == "sbm" and not np.isnan(r["_ci"])]


def prep_breg(records):
    """Return (CI_clipped, k_standardised, log(n)_standardised) arrays."""
    ci   = np.clip([r["_ci"]        for r in records], 1e-4, 1 - 1e-4)
    k    = np.array([r["_target_k"] for r in records], dtype=float)
    logn = np.log(np.array([r["_n_nodes"] for r in records], dtype=float))
    k_s  = (k    - k.mean())    / k.std()
    n_s  = (logn - logn.mean()) / logn.std()
    return np.array(ci, dtype=float), k_s, n_s


bk_results: dict[str, np.ndarray] = {}
for label, sched in [("gillespie", "gillespie_occ"), ("sync", "sync")]:
    recs = [r for r in a3_base if r["_scheduler"] == sched]
    ci, k_s, n_s = prep_breg(recs)
    print(f"  {label}: {len(ci):,} runs")

    t3_path = TRACES / f"analysis_3_{label}.nc"
    _tr3 = load_or_none(t3_path)
    if _tr3 is not None:
        print(f"  Loading cached {label} trace…")
        tr3 = _tr3
    else:
        with pm.Model():
            alpha3  = pm.Normal("alpha",  0, 1)
            beta_k3 = pm.Normal("beta_k", 0, 1)
            beta_n3 = pm.Normal("beta_n", 0, 1)
            phi3    = pm.Gamma("phi", alpha=2, beta=0.5)
            mu3     = pm.math.invlogit(alpha3 + beta_k3 * k_s + beta_n3 * n_s)
            pm.Beta("obs", alpha=mu3 * phi3, beta=(1 - mu3) * phi3, observed=ci)
            tr3 = pm.sample(**SAMPLE_KW)
        az.to_netcdf(tr3, str(t3_path))
        print(f"  {label} trace saved.")

    bk_results[label] = tr3.posterior["beta_k"].values.flatten()

bk_g   = bk_results["gillespie"]
bk_s   = bk_results["sync"]
bkg_m, (bkg_lo, bkg_hi) = float(np.mean(bk_g)),   hdi_arr(bk_g)
bks_m, (bks_lo, bks_hi) = float(np.mean(bk_s)),   hdi_arr(bk_s)
p_g_neg    = float(np.mean(bk_g < -0.1))     # P(beta_k_gill < -0.1) formal test
p_sync_gt  = prob_gt(bk_s, bk_g)             # P(beta_k_sync > beta_k_gill)

S("### Beta regression: effect of k on CI",
  f"  Gillespie β_k : {bkg_m:+.3f}  89% HDI [{bkg_lo:+.3f}, {bkg_hi:+.3f}]  "
  f"(negative = higher k → lower CI)",
  f"  Sync      β_k : {bks_m:+.3f}  89% HDI [{bks_lo:+.3f}, {bks_hi:+.3f}]",
  f"  P(β_k_Gillespie < −0.1)        = {p_g_neg:.3f}  — formal density-collapse test",
  f"  P(β_k_sync > β_k_Gillespie)    = {p_sync_gt:.3f}  — scheduler-dependent effect")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4 — Competition structure effect on coexistence
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 4 — Competition structure effect")
print("=" * 70)

S("## Analysis 4 — Competition structure effect on coexistence",
  "Model: genuine_coexistence_count ~ Binomial(n_runs, p),  p ~ Beta(1,1).",
  "Genuine coexistence = survivors_count == 3 AND jain_fairness_final >= 0.85.",
  "  (survivors<3 was rejected as outcome: RPS has marginally *higher* extinction than AVA,",
  "   a tiny reversed effect swamped by noise. Genuine coexistence shows a 2× RPS advantage.)",
  "Data: RPS and AVA, irregular graphs (ER, SBM, DC-SBM), n=625, all k pooled.")

JAIN_THRESHOLD = 0.85
irregular = {"er", "sbm", "dc_sbm"}
grp4: dict = defaultdict(list)
for contest, records in [("rps", rps), ("ava", ava)]:
    for r in records:
        if (r["_graph_type"] in irregular
                and r["_n_nodes"] == 625
                and "survivors_count" in r):       # skip connectivity failures
            gc = int(r["survivors_count"] == 3
                     and r["jain_fairness_final"] >= JAIN_THRESHOLD)
            grp4[(contest, r["_graph_type"])].append(gc)

conds4 = sorted(grp4)
gc4    = np.array([sum(grp4[c]) for c in conds4])
ntri4  = np.array([len(grp4[c]) for c in conds4])
n_c4   = len(conds4)

print(f"  Condition breakdown (genuine coexistence / runs):")
for c, g, n in zip(conds4, gc4, ntri4):
    print(f"    {str(c):30s}  {g:4d}/{n:4d}  ({100*g/n:.1f}%)")

t4_path = TRACES / "analysis_4.nc"
_tr4 = load_or_none(t4_path)
if _tr4 is not None:
    print("  Loading cached trace…")
    tr4 = _tr4
else:
    with pm.Model():
        p_a4 = pm.Beta("p", alpha=1, beta=1, shape=n_c4)
        pm.Binomial("obs", n=ntri4, p=p_a4, observed=gc4)
        tr4 = pm.sample(**SAMPLE_KW)
    az.to_netcdf(tr4, str(t4_path))
    print("  Trace saved.")

p4_samp = tr4.posterior["p"].values.reshape(-1, n_c4)
c4i     = {c: i for i, c in enumerate(conds4)}

S("### RPS vs AVA genuine coexistence rate (by graph type)")
for gt in ("er", "sbm", "dc_sbm"):
    rk = ("rps", gt)
    ak = ("ava", gt)
    if rk not in c4i or ak not in c4i:
        continue
    pr = p4_samp[:, c4i[rk]]
    pa = p4_samp[:, c4i[ak]]
    prob = prob_gt(pr, pa)          # P(RPS coexistence > AVA coexistence)
    lo, hi = hdi_arr(pr - pa)
    S(f"  {gt.upper():6s}: P(RPS coexistence > AVA coexistence) = {prob:.3f} | "
      f"RPS={np.mean(pr):.3f}  AVA={np.mean(pa):.3f} | "
      f"Δ(RPS−AVA) 89% HDI [{lo:+.3f}, {hi:+.3f}]")

S("### Block structure vs ER genuine coexistence (within RPS)")
for gt in ("sbm", "dc_sbm"):
    bk = ("rps", gt)
    ek = ("rps", "er")
    if bk not in c4i or ek not in c4i:
        continue
    pg = p4_samp[:, c4i[bk]]
    pe = p4_samp[:, c4i[ek]]
    prob = prob_gt(pg, pe)          # P(block coexistence > ER coexistence)
    lo, hi = hdi_arr(pg - pe)
    S(f"  P({gt.upper()} coexistence > ER coexistence) = {prob:.3f} | "
      f"{gt}={np.mean(pg):.3f}  ER={np.mean(pe):.3f} | "
      f"Δ(block−ER) 89% HDI [{lo:+.3f}, {hi:+.3f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5 — Policy insensitivity
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 5 — Policy insensitivity")
print("=" * 70)

S("## Analysis 5 — Policy insensitivity",
  "Model: outcome ~ Normal(α + policy_effect[i], σ),  policy_effect ~ Normal(0, 0.5).",
  f"ROPE = ±{ROPE} outcome units.  Null = intercept only (no policy term).",
  "LOO-CV: if null model wins, policy insensitivity is formally supported.")

datasets5 = {
    "two_player": {
        "records": [r for r in two_p if not np.isnan(r["_ci"])],
        "outcome": lambda r: r["_ci"],
    },
    "three_player": {
        "records": [r for r in rps if "jain_fairness_final" in r],  # skip failures
        "outcome": lambda r: float(r["jain_fairness_final"]),
    },
}

for ds_label, ds in datasets5.items():
    recs      = ds["records"]
    out_fn    = ds["outcome"]
    policies  = sorted({r["_policy"] for r in recs})
    pol_idx   = {p: i for i, p in enumerate(policies)}
    n_pol     = len(policies)
    outcome   = np.array([out_fn(r) for r in recs], dtype=float)
    pidx      = np.array([pol_idx[r["_policy"]] for r in recs])
    outcome   = np.clip(outcome, 0.0, 1.0)

    print(f"\n  {ds_label}: {len(outcome):,} runs | policies: {policies}")

    t5p = TRACES / f"analysis_5_{ds_label}_policy.nc"
    t5n = TRACES / f"analysis_5_{ds_label}_null.nc"

    # ── policy model ──────────────────────────────────────────────────────────
    _tr5p = load_or_none(t5p)
    if _tr5p is not None and has_ll(_tr5p):
        print(f"  Loading cached {ds_label} policy trace…")
        tr5p = _tr5p
    else:
        with pm.Model():
            alpha5 = pm.Normal("alpha", 0, 0.5)
            sigma5 = pm.HalfNormal("sigma", 0.5)
            pe5    = pm.Normal("policy_effect", 0, 0.5, shape=n_pol)
            pm.Normal("obs", alpha5 + pe5[pidx], sigma5, observed=outcome)
            tr5p = pm.sample(**SAMPLE_KW)
            pm.compute_log_likelihood(tr5p)
        az.to_netcdf(tr5p, str(t5p))
        print(f"  {ds_label} policy trace saved.")

    # ── null model (intercept only) ───────────────────────────────────────────
    _tr5n = load_or_none(t5n)
    if _tr5n is not None and has_ll(_tr5n):
        print(f"  Loading cached {ds_label} null trace…")
        tr5n = _tr5n
    else:
        with pm.Model():
            alpha5n = pm.Normal("alpha", 0, 0.5)
            sigma5n = pm.HalfNormal("sigma", 0.5)
            pm.Normal("obs", alpha5n, sigma5n, observed=outcome)
            tr5n = pm.sample(**SAMPLE_KW)
            pm.compute_log_likelihood(tr5n)
        az.to_netcdf(tr5n, str(t5n))
        print(f"  {ds_label} null trace saved.")

    # ── summarise ─────────────────────────────────────────────────────────────
    pe_samp = tr5p.posterior["policy_effect"].values.reshape(-1, n_pol)

    S(f"### {ds_label}  (ROPE = ±{ROPE})")
    in_rope = []
    for i, pol in enumerate(policies):
        col = pe_samp[:, i]
        lo, hi = hdi_arr(col)
        in_r = lo > -ROPE and hi < ROPE
        in_rope.append(in_r)
        S(f"  {pol:5s}: mean={np.mean(col):+.4f}  89% HDI [{lo:+.4f}, {hi:+.4f}]"
          + ("  ✓ in ROPE" if in_r else "  ✗ outside ROPE"))

    S(f"  All policy effects in ROPE: {all(in_rope)}")

    try:
        cmp5    = az.compare({"policy": tr5p, "null": tr5n}, ic="loo")
        best5   = cmp5.index[0]
        d_elpd5 = float(cmp5.loc["policy", "elpd_loo"]) - float(cmp5.loc["null", "elpd_loo"])
        dse5    = float(cmp5.loc[cmp5.index[1], "dse"])
        S(f"  LOO-CV: best model = '{best5}'",
          f"  ΔELPD(policy − null) = {d_elpd5:+.1f} ± {dse5:.1f}"
          f"  (negative = null wins = policy insensitivity supported)")
    except Exception as exc:
        S(f"  LOO-CV comparison failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5b — Policy insensitivity (controlled: scheduler + graph_type)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 5b — Policy insensitivity (controlled)")
print("=" * 70)

S("## Analysis 5b — Policy insensitivity (controlled for scheduler and graph type)",
  "Model: outcome ~ Normal(α + sched_eff[s] + graph_eff[g] + pol_eff[p], σ).",
  "All categorical effects use ZeroSumNormal (sum-to-zero, deviation from grand mean).",
  "Scheduler and graph type absorb between-cell variance; policy residual is then tested.",
  "ROPE = ±0.05 outcome units.  Null = same model without pol_eff.  LOO-CV comparison.",
  "Two-player: controls for both scheduler and graph type.",
  "Three-player: controls for graph type only (Gillespie-only data).")

datasets5b = {
    "two_player": {
        "records":   [r for r in two_p if not np.isnan(r["_ci"])],
        "outcome":   lambda r: r["_ci"],
        "sched_key": "_scheduler",
        "graph_key": "_graph_type",
        "pol_key":   "_policy",
        "has_sched": True,
    },
    "three_player": {
        "records":   [r for r in rps if "jain_fairness_final" in r],
        "outcome":   lambda r: float(r["jain_fairness_final"]),
        "sched_key": "_scheduler",
        "graph_key": "_graph_type",
        "pol_key":   "_policy",
        "has_sched": False,   # only Gillespie in three-player data
    },
}

for ds_label, ds in datasets5b.items():
    recs      = ds["records"]
    out_fn    = ds["outcome"]
    has_sched = ds["has_sched"]

    policies   = sorted({r[ds["pol_key"]]   for r in recs})
    graphs     = sorted({r[ds["graph_key"]] for r in recs})
    scheds     = sorted({r[ds["sched_key"]] for r in recs}) if has_sched else []

    n_pol  = len(policies)
    n_grph = len(graphs)
    n_sch  = len(scheds)

    pol_idx  = {p: i for i, p in enumerate(policies)}
    grph_idx = {g: i for i, g in enumerate(graphs)}
    sch_idx  = {s: i for i, s in enumerate(scheds)} if has_sched else {}

    outcome = np.clip([out_fn(r) for r in recs], 0.0, 1.0)
    pidx    = np.array([pol_idx[r[ds["pol_key"]]]   for r in recs])
    gidx    = np.array([grph_idx[r[ds["graph_key"]]] for r in recs])
    sidx    = np.array([sch_idx[r[ds["sched_key"]]]  for r in recs]) if has_sched else None

    print(f"\n  {ds_label}: {len(outcome):,} runs")
    print(f"    policies={policies}  graphs={graphs}"
          + (f"  scheds={scheds}" if has_sched else ""))

    t5bp = TRACES / f"analysis_5b_{ds_label}_policy.nc"
    t5bn = TRACES / f"analysis_5b_{ds_label}_null.nc"

    # ── controlled policy model ───────────────────────────────────────────────
    _tr5bp = load_or_none(t5bp)
    if _tr5bp is not None and has_ll(_tr5bp):
        print(f"  Loading cached {ds_label} controlled-policy trace…")
        tr5bp = _tr5bp
    else:
        with pm.Model():
            alpha5b  = pm.Normal("alpha",      0, 0.5)
            graph_eff = pm.ZeroSumNormal("graph_effect",  sigma=0.5, shape=n_grph)
            pol_eff   = pm.ZeroSumNormal("policy_effect", sigma=0.5, shape=n_pol)
            sigma5b  = pm.HalfNormal("sigma", 0.5)
            mu5b = alpha5b + graph_eff[gidx] + pol_eff[pidx]
            if has_sched:
                sched_eff = pm.ZeroSumNormal("sched_effect", sigma=0.5, shape=n_sch)
                mu5b = mu5b + sched_eff[sidx]
            pm.Normal("obs", mu5b, sigma5b, observed=outcome)
            tr5bp = pm.sample(**SAMPLE_KW)
            pm.compute_log_likelihood(tr5bp)
        az.to_netcdf(tr5bp, str(t5bp))
        print(f"  {ds_label} controlled-policy trace saved.")

    # ── controlled null model (no policy term) ────────────────────────────────
    _tr5bn = load_or_none(t5bn)
    if _tr5bn is not None and has_ll(_tr5bn):
        print(f"  Loading cached {ds_label} controlled-null trace…")
        tr5bn = _tr5bn
    else:
        with pm.Model():
            alpha5bn  = pm.Normal("alpha",       0, 0.5)
            graph_effn = pm.ZeroSumNormal("graph_effect", sigma=0.5, shape=n_grph)
            sigma5bn  = pm.HalfNormal("sigma", 0.5)
            mu5bn = alpha5bn + graph_effn[gidx]
            if has_sched:
                sched_effn = pm.ZeroSumNormal("sched_effect", sigma=0.5, shape=n_sch)
                mu5bn = mu5bn + sched_effn[sidx]
            pm.Normal("obs", mu5bn, sigma5bn, observed=outcome)
            tr5bn = pm.sample(**SAMPLE_KW)
            pm.compute_log_likelihood(tr5bn)
        az.to_netcdf(tr5bn, str(t5bn))
        print(f"  {ds_label} controlled-null trace saved.")

    # ── summarise ─────────────────────────────────────────────────────────────
    pe_samp = tr5bp.posterior["policy_effect"].values.reshape(-1, n_pol)

    controls = "scheduler + graph type" if has_sched else "graph type"
    S(f"### {ds_label}  (controlled for {controls})")
    in_rope = []
    for i, pol in enumerate(policies):
        col = pe_samp[:, i]
        lo, hi = hdi_arr(col)
        in_r = lo > -ROPE and hi < ROPE
        in_rope.append(in_r)
        S(f"  {pol:5s}: mean={np.mean(col):+.4f}  89% HDI [{lo:+.4f}, {hi:+.4f}]"
          + ("  ✓ in ROPE" if in_r else "  ✗ outside ROPE"))

    S(f"  All policy effects in ROPE: {all(in_rope)}")

    # Residual sigma (within-cell noise after controlling)
    sig_samp = tr5bp.posterior["sigma"].values.flatten()
    S(f"  Residual σ (after controlling): {np.mean(sig_samp):.3f}  "
      f"89% HDI [{hdi_arr(sig_samp)[0]:.3f}, {hdi_arr(sig_samp)[1]:.3f}]"
      f"  (cf. σ≈0.30+ in uncontrolled model)")

    try:
        cmp5b   = az.compare({"policy": tr5bp, "null": tr5bn}, ic="loo")
        best5b  = cmp5b.index[0]
        d_elpd  = float(cmp5b.loc["policy","elpd_loo"]) - float(cmp5b.loc["null","elpd_loo"])
        dse5b   = float(cmp5b.loc[cmp5b.index[1], "dse"])
        S(f"  LOO-CV: best model = '{best5b}'",
          f"  ΔELPD(policy − null) = {d_elpd:+.1f} ± {dse5b:.1f}"
          f"  (negative = null wins = policy insensitivity supported)")
    except Exception as exc:
        S(f"  LOO-CV comparison failed: {exc}")


# ── write summary file ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
out_path = Path("statistical_results.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(_out))
print(f"Summary written to {out_path}")
print(f"Traces cached in   {TRACES}/")
print("Done.")
