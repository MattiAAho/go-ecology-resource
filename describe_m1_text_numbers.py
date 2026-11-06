"""
describe_m1_text_numbers.py — Numbers quoted in the manuscript text and tables that no figure shows.

Reads the result files (no games are run) and prints the numbers behind:
  1. Table 4 (Claim 3 rows) and the three-player Results: the weakest species' share of the final stones,
     regular vs irregular graphs, per competition structure and board size.
  2. The scheduler-robustness paragraph: a species absent at the end, synchronous vs Gillespie-like, on the cells
     where all three competition structures were run under both schedulers (ER and SBM, k = 10 and 14).
  3. Figure 2's caption and Table 4 (Claim 1 rows): two-player dominance rates (CI < 0.10) per cell.
  4. The two-player bimodality paragraph and Figure 4's caption: the dominance (CI < 0.10) and segregation (CI > 0.85)
     shares per graph family and per mean degree.
  5. The behavioural-strategy paragraph: whether "regular graphs end more even than irregular graphs" holds within each
     of the four policy mixes, per board size and competition structure.
  6. The bimodality paragraph and the Discussion: capture events per move, and stones captured per move in the
     segregation and the dominance runs.
  7. The scheduler paragraph: the ratio of synchronous to Gillespie-like dominance rates at n = 625.
  8. The three-player paragraph and S6's caption: genuine coexistence per graph family, and its ratios.
  9. The scheduler-robustness paragraph: the synchronous three-player diagnostic, per cell.
 10. The two-player gate paragraph and Figure 7's caption: the gate diagnostic.
 11. The three-player gate paragraph and S8's caption: the three-player gate diagnostic.
 12. The exploration-rate paragraph and the behavioural-strategy paragraph: the exploration-rate sweep.
 13. S1's caption and the behavioural-strategy paragraph: the mean outcome per policy combination.
 14. S4's caption: stone groups in two-player segregation runs; the exploration-rate paragraph and S9's caption: the
     all-three-present gap between the two exploration rates over time, and how fast the board fills.
The settling times in Methods and in S11's caption are printed by analysis_time_to_limbo.py; the posterior
probabilities and HDIs of the Bayesian analyses are in statistical_results.md (statistical_analysis.py).

Definitions:
  s_min  = the smallest species' share of the stones at the final turn (a species with no stones counts as 0);
           two-player CI = 2 s_min (= ci_2p), so CI < 0.10 is s_min < 0.05 and CI > 0.85 is s_min > 0.425.
  Rows flagged connectivity_failed are dropped. REGULAR = lattice, torus; IRREGULAR = ER, SBM, DC-SBM (all mean degrees).
  Sections 1, 4 (beside) and 5 use the energy-weighted asynchronous, Gillespie-like scheduler (`gillespie_occ`) only.
  Genuine coexistence = all three species present and Jain index >= 0.85; complete monopoly = one species left.
  Shapes: games per s_min bin
    three-player  0 | (0, .05] | (.05, .10] | (.10, .20] | (.20, 1/3]
    two-player    0 | (0, .05] | (.05, .20] | (.20, .425] | (.425, .5]

Requires results/two_p_all_v1.pkl, full_all_v2.pkl, ava_all_v1.pkl (Steps 2-4), eps_sweep_2p_v1.pkl,
eps_sweep_rps_v1.pkl and eps_sweep_ava_v1.pkl (Step 5), and gate_diag_v1.pkl, gate_3p_diag_v1.pkl, sync_3p_diag_v1.pkl (Step 6).

Usage:
    python describe_m1_text_numbers.py        # prints, and writes results/m1_text_numbers.txt
"""
import gc
import pickle
from collections import Counter, defaultdict

import numpy as np

from analysis_helpers import ci_2p

OUT = "results/m1_text_numbers.txt"
CONTESTS = (("two-player", "two_p_all_v1.pkl", (1, -1)), ("RPS", "full_all_v2.pkl", (1, -1, 2)),
            ("AVA", "ava_all_v1.pkl", (1, -1, 2)))
CUTS = {2: (.05, .20, .425), 3: (.05, .10, .20)}
REGULAR = ("lattice", "torus")
IRREGULAR = ("er", "sbm", "dc_sbm")
GRIDS = (9, 15, 19, 25)
KS = (8, 10, 12, 14, 19)
MIXES = {"two-player": ("BB", "BA", "AB", "AA"), "RPS": ("AAA", "BAA", "BAB", "BBB"),
         "AVA": ("AAA", "BAA", "BAB", "BBB")}
BOOT = 2000
RNG = np.random.default_rng(2026)

lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(str(s))


def s_min(r, cols):
    v = [r["n_stones"].get(c, r["n_stones"].get(str(c), 0)) for c in cols]
    return min(v) / sum(v) if sum(v) else 0.0


def shape(xs, ns):
    h = Counter(0 if x == 0 else 1 + sum(x > c for c in CUTS[ns]) for x in xs)
    return " / ".join(f"{h[i]:4d}" for i in range(5))


def pct(x):
    return f"{100 * x:.1f}%"


def load(fname, cols):
    """Compact rows: (scheduler, graph type, grid, target k, policy, s_min, ci_2p or None, captures_n, moves,
    capture_rate_per_move, survivors_count, jain_fairness_final), connectivity failures out."""
    with open(f"results/{fname}", "rb") as fh:
        R = pickle.load(fh)
    out, skipped = [], Counter()
    for r in R:
        if r.get("connectivity_failed"):
            skipped[r["_scheduler"]] += 1
            continue
        ci = ci_2p(r) if len(cols) == 2 and r.get("black_occupancy") is not None else None
        out.append((r["_scheduler"], r["_graph_type"], r["_grid_size"], r["_target_k"], r.get("_policy"),
                    s_min(r, cols), ci, r["captures_n"], r["moves"], r["capture_rate_per_move"],
                    r.get("survivors_count"), r.get("jain_fairness_final")))
    del R
    gc.collect()
    return out, dict(skipped)


# ── 1. Weakest species' share, regular vs irregular ──────────────────────────
def section1(rows):
    say("1. WEAKEST SPECIES' SHARE OF THE FINAL STONES, REGULAR VS IRREGULAR (Gillespie-like; Table 4, Claim 3 rows)")
    say("   per board size: shape [games]; % a species absent (s_min = 0) / % weakest at most 5% / % weakest above 20%")
    for name, _, cols in CONTESTS:
        ns = len(cols)
        R = [x for x in rows[name] if x[0] == "gillespie_occ"]
        say(f"=== {name} ({len(R)} rows)")
        for g in GRIDS:
            for kind, boards in (("irregular", IRREGULAR), ("regular", REGULAR)):
                xs = np.array([x[5] for x in R if x[2] == g and x[1] in boards])
                say(f"  n {g * g:4d} {kind:9s}: {shape(xs, ns)}  [{len(xs)}]   absent {pct(np.mean(xs == 0))} / "
                    f"<= 5% {pct(np.mean(xs <= .05))} / > 20% {pct(np.mean(xs > .20))}")
    say()


# ── 2. Matched cells, both schedulers ────────────────────────────────────────
def section2(rows, sync3):
    say("2. MATCHED CELLS UNDER BOTH SCHEDULERS (ER and SBM, k = 10 and 14; scheduler-robustness paragraph)")
    say("   three-player synchronous rows: sync_3p_diag_v1.pkl; all other rows: the production files")
    for g in (15, 25):
        for sch in ("sync", "gillespie_occ"):
            for name, _, cols in CONTESTS:
                R = rows[name] + (sync3[name] if sch == "sync" and name != "two-player" else [])
                xs = np.array([x[5] for x in R if x[0] == sch and x[2] == g and x[1] in ("er", "sbm")
                               and x[3] in (10, 14)])
                if len(xs):
                    say(f"  n {g * g:3d} {sch:13s} {name:10s}: {shape(xs, len(cols))}  [{len(xs)}]  "
                        f"s_min <= .05: {np.mean(xs <= .05):.2f}; s_min = 0: {np.mean(xs == 0):.2f}")
    say()


# ── 3. Two-player dominance rates per cell ───────────────────────────────────
def section3(rows):
    say("3. TWO-PLAYER DOMINANCE RATE (CI < 0.10) PER CELL (Figure 2 caption; Table 4, Claim 1 rows)")
    C = defaultdict(list)
    for sch, gt, g, k, _, _, ci in (x[:7] for x in rows["two-player"]):
        if gt in IRREGULAR and ci is not None:
            C[(gt, sch, g, k)].append(ci < 0.1)
    rate = {key: float(np.mean(v)) for key, v in C.items()}
    get = lambda gt, s, g, k: rate.get((gt, s, g, k))
    say("   %, rows n = 625 / 361 / 225 / 81, columns k = 8 10 12 14 19")
    for s in ("sync", "gillespie_occ"):
        for gt in IRREGULAR:
            say(f"  {s:<13} {gt:<7} " + " | ".join(
                " ".join(f"{100 * get(gt, s, g, k):3.0f}" if get(gt, s, g, k) is not None else "  —" for k in KS)
                for g in GRIDS[::-1]))

    def rng_(cells):
        v = [x for x in cells if x is not None]
        return f"{100 * min(v):.0f}–{100 * max(v):.0f}% ({len(v)} cells)"

    cells = lambda s, gs, graphs, ks=KS: [get(gt, s, g, k) for g in gs for gt in graphs for k in ks]
    say(f"  n = 625, ER/SBM/DC-SBM, per k: Gillespie-like {rng_(cells('gillespie_occ', (25,), IRREGULAR))}, "
        f"sync {rng_(cells('sync', (25,), IRREGULAR))}")
    say(f"  n = 81, ER/SBM, k = 19: Gillespie-like {rng_(cells('gillespie_occ', (9,), IRREGULAR[:2], (19,)))}, "
        f"sync {rng_(cells('sync', (9,), IRREGULAR[:2], (19,)))}")
    say(f"  sync, n <= 361: {rng_(cells('sync', (9, 15, 19), IRREGULAR))}")
    up = [f"{gt} k {k}" for gt in IRREGULAR for k in KS
          if get(gt, "gillespie_occ", 9, k) is not None and get(gt, "gillespie_occ", 9, k) > get(gt, "sync", 9, k)]
    say(f"  n = 81 cells where the Gillespie-like rate exceeds the synchronous: {', '.join(up)}")
    say()


# ── 4. Two-player bimodality ─────────────────────────────────────────────────
def section4(rows):
    say("4. TWO-PLAYER DOMINANCE / SEGREGATION SHARES (bimodality paragraph; Figure 4 caption)")
    R = rows["two-player"]
    say("   per graph family, n >= 225 (grids 15-25): shape [games]; share CI < .10 / middle / CI > .85")
    for sch in ("sync", "gillespie_occ"):
        say(f"=== {sch}")
        for b in REGULAR + IRREGULAR:
            xs = np.array([2 * x[5] for x in R if x[0] == sch and x[1] == b and x[2] in (15, 19, 25)])
            lo, hi = np.mean(xs < .10), np.mean(xs > .85)
            say(f"  {b:<8} grid 15-25: {shape(xs / 2, 2)} [{len(xs)}]   {lo:.2f} / {1 - lo - hi:.2f} / {hi:.2f}")
    xs = np.array([x[5] for x in R if x[0] == "gillespie_occ" and x[1] in ("sbm", "dc_sbm") and x[2] in (15, 19, 25)])
    rest = xs[xs <= .425]
    say(f"  Gillespie-like, SBM and DC-SBM, n >= 225, runs outside the segregation mode (CI <= .85): {len(rest)}; "
        f"losing species at 5-20% of the stones {np.mean((rest > .05) & (rest <= .20)):.2f}, "
        f"dominance (CI < .10) {np.mean(rest < .05):.2f}")
    say("   SBM and DC-SBM, per mean degree: share dominance (CI < .10) / segregation (CI > .85) [games]")
    for sch in ("sync", "gillespie_occ"):
        say(f"=== {sch}")
        for k in KS:
            cell = []
            for label, gs in (("all sizes", GRIDS), ("n 81", (9,)), ("n >= 225", (15, 19, 25))):
                x = np.array([2 * r[5] for r in R if r[0] == sch and r[1] in ("sbm", "dc_sbm") and r[3] == k
                              and r[2] in gs])
                cell.append(f"{label} {np.mean(x < .10):.2f} / {np.mean(x > .85):.2f} [{len(x)}]")
            say(f"  k {k:>2}: " + "   ".join(cell))
    say()


# ── 5. The direction within each policy mix ──────────────────────────────────
def section5(rows):
    say("5. REGULAR MINUS IRREGULAR WITHIN EACH POLICY MIX (Gillespie-like; behavioural-strategy paragraph)")
    say("   per game r = S x s_min (S species); share high (r >= .7) and share low (r < .3), regular minus irregular;")
    say(f"   95% bootstrap ({BOOT}, games resampled within each graph type x mean degree cell). HOLDS = the high")
    say("   difference > 0 with its interval above 0, and the low difference < 0.")
    for name, _, cols in CONTESTS:
        S = len(cols)
        cells = defaultdict(list)
        for sch, gt, g, k, pol, sm, _ in (x[:7] for x in rows[name]):
            if sch == "gillespie_occ":
                cells[(pol, g, gt in REGULAR, gt, k)].append(S * sm)
        cells = {key: np.array(v) for key, v in cells.items()}
        say(f"=== {name}")
        holds = {}
        for g in GRIDS:
            for m in MIXES[name]:
                reg = [v for key, v in cells.items() if key[:3] == (m, g, True)]
                irr = [v for key, v in cells.items() if key[:3] == (m, g, False)]

                def diff(a, b):
                    a, b = np.concatenate(a), np.concatenate(b)
                    return np.mean(a >= .7) - np.mean(b >= .7), np.mean(a < .3) - np.mean(b < .3)

                dh, dl = diff(reg, irr)
                boot = [diff([RNG.choice(c, len(c)) for c in reg], [RNG.choice(c, len(c)) for c in irr])[0]
                        for _ in range(BOOT)]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                holds[(m, g)] = bool(dh > 0 and lo > 0 and dl < 0)
                say(f"  grid {g:>2} {m:3s}: high diff {dh:+.2f} [{lo:+.2f}, {hi:+.2f}] low diff {dl:+.2f} -> "
                    f"{'HOLDS' if holds[(m, g)] else 'does not hold'}")
        for g in GRIDS:
            say(f"  n {g * g:3d}: holds in {sum(holds[(m, g)] for m in MIXES[name])} of 4 mixes")
    say()


def genuine(surv, jain):
    return surv == 3 and (jain or 0) >= 0.85


def rng0(vals):
    lo, hi = f"{100 * min(vals):.0f}", f"{100 * max(vals):.0f}"
    return f"{lo}%" if lo == hi else f"{lo}–{hi}%"


def load_diag(fname):
    """Rows of a diagnostic or exploration-rate file (no connectivity failures occur in these files)."""
    with open(f"results/{fname}", "rb") as fh:
        R = pickle.load(fh)
    out = [dict(contest=r.get("_contest"), sch=r["_scheduler"], gt=r["_graph_type"], n=r["_n_nodes"],
                k=r["_target_k"], pol=r.get("_policy"), gate=r.get("_use_gate"), eps=r.get("_eps"),
                bite=r.get("gate_bite_rate"), s2=s_min(r, (1, -1)), surv=r.get("survivors_count"),
                jain=r.get("jain_fairness_final"))
           for r in R if not r.get("connectivity_failed")]
    del R
    gc.collect()
    return out


# ── 6. Captures per move ─────────────────────────────────────────────────────
def section6(rows):
    say("6. CAPTURES PER MOVE (bimodality paragraph; Discussion)")
    say("   capture events per move = all capture events / all moves, production rows")
    for name, _, _ in CONTESTS:
        for sch in ("sync", "gillespie_occ"):
            R = [x for x in rows[name] if x[0] == sch]
            if R:
                say(f"  {name:10s} {sch:13s}: {pct(sum(x[7] for x in R) / sum(x[8] for x in R))} [{len(R)} games]")
    say("   two-player, SBM and DC-SBM: mean stones captured per move (capture_rate_per_move) in segregation runs")
    say("   (CI > .85) and in dominance runs (CI < .10)")
    for sch in ("sync", "gillespie_occ"):
        R = [x for x in rows["two-player"] if x[0] == sch and x[1] in ("sbm", "dc_sbm")]
        seg = [x[9] for x in R if 2 * x[5] > .85]
        dom = [x[9] for x in R if 2 * x[5] < .10]
        say(f"  {sch:13s}: segregation {np.mean(seg):.3f} [{len(seg)}], dominance {np.mean(dom):.3f} [{len(dom)}]")
    say()


# ── 7. Dominance-rate ratio at n = 625 ───────────────────────────────────────
def section7(rows):
    say("7. DOMINANCE RATE, SYNCHRONOUS / GILLESPIE-LIKE, n = 625, ER AND SBM (scheduler paragraph)")
    say("   per cell: posterior mean rate (dominance runs + 1) / (games + 2), Beta(1, 1) prior; dominance = CI < 0.10")
    rat = []
    for gt in ("er", "sbm"):
        for k in KS:
            c = [x for x in rows["two-player"] if x[1] == gt and x[3] == k and x[2] == 25]
            pm = {}
            for sch in ("sync", "gillespie_occ"):
                d = [2 * x[5] < .1 for x in c if x[0] == sch]
                pm[sch] = (sum(d) + 1) / (len(d) + 2)
            rat.append(pm["sync"] / pm["gillespie_occ"])
            say(f"  {gt:3s} k {k:2d}: sync {pm['sync']:.3f}, Gillespie-like {pm['gillespie_occ']:.3f}, "
                f"ratio {rat[-1]:.1f}")
    say(f"  ratio: median {np.median(rat):.1f}, range {min(rat):.1f}–{max(rat):.1f} ({len(rat)} cells)")
    say()


# ── 8. Genuine coexistence per graph family ──────────────────────────────────
def section8(rows):
    say("8. GENUINE COEXISTENCE PER GRAPH FAMILY, ALL BOARD SIZES AND MEAN DEGREES (three-player paragraph; S6 caption)")
    gen = {}
    for name in ("RPS", "AVA"):
        cells = []
        for gt in IRREGULAR:
            R = [x for x in rows[name] if x[1] == gt]
            gen[(name, gt)] = np.mean([genuine(x[10], x[11]) for x in R])
            cells.append(f"{gt} {pct(gen[(name, gt)])} [{len(R)}]")
        say(f"  {name}: " + ", ".join(cells))
    for name in ("RPS", "AVA"):
        say(f"  {name}, block-structured / ER: " + ", ".join(f"{gt} {gen[(name, gt)] / gen[(name, 'er')]:.1f}x"
                                                          for gt in ("sbm", "dc_sbm")))
    say("  RPS / AVA: " + ", ".join(f"{gt} {gen[('RPS', gt)] / gen[('AVA', gt)]:.1f}x" for gt in IRREGULAR))
    say()


# ── 9. Synchronous three-player diagnostic per cell ──────────────────────────
def section9(rows, sync3):
    say("9. SYNCHRONOUS THREE-PLAYER DIAGNOSTIC PER CELL (scheduler-robustness paragraph)")
    for name in ("RPS", "AVA"):
        reps = Counter((x[1], x[2], x[3], x[4]) for x in sync3[name])
        say(f"=== {name}: synchronous rows {len(sync3[name])}, games per cell (graph, n, k, policy) "
            f"{sorted(set(reps.values()))}, policies {sorted({x[4] for x in sync3[name]})}")
        say("   complete monopoly (one species left) / all three present [games]; Gillespie-like = the production "
            "games with the same policies (AAA, BBB)")
        mono = {"sync": [], "gillespie_occ": []}
        for g in (15, 25):
            for gt in ("er", "sbm"):
                for k in (10, 14):
                    out = []
                    for sch, R in (("sync", sync3[name]), ("gillespie_occ", rows[name])):
                        c = [x for x in R if x[0] == sch and x[1] == gt and x[2] == g and x[3] == k
                             and x[4] in ("AAA", "BBB")]
                        m = np.mean([x[10] == 1 for x in c])
                        if g == 25:
                            mono[sch].append(m)
                        out.append(f"{sch} {pct(m)} / {pct(np.mean([x[10] == 3 for x in c]))} [{len(c)}]")
                    say(f"  n {g * g:3d} {gt:3s} k {k}: " + " | ".join(out))
        say(f"  n = 625, complete monopoly per cell: sync {rng0(mono['sync'])}, Gillespie-like "
            f"{rng0(mono['gillespie_occ'])}")
    say()


# ── 10. Two-player gate diagnostic ───────────────────────────────────────────
def section10():
    say("10. TWO-PLAYER GATE DIAGNOSTIC (gate_diag_v1.pkl; gate paragraph; Figure 7 caption)")
    R = load_diag("gate_diag_v1.pkl")
    say(f"   {len(R)} games, policies {sorted({x['pol'] for x in R})}; segregation = CI > .85, dominance = CI < .10;")
    say("   refused = mean share of attempted captures the gate refused (gate_bite_rate), in the gate-ON segregation runs")
    cell = lambda sch, n, gt, gate, k=None: [x for x in R if x["sch"] == sch and x["n"] == n and x["gt"] == gt
                                              and bool(x["gate"]) == gate and (k is None or x["k"] == k)]
    seg = lambda c: np.mean([2 * x["s2"] > .85 for x in c])
    dom = lambda c: np.mean([2 * x["s2"] < .10 for x in c])
    S = {}
    for sch in ("gillespie_occ", "sync"):
        for n in (225, 625):
            for gt in IRREGULAR:
                on, off = cell(sch, n, gt, True), cell(sch, n, gt, False)
                S[(sch, n, gt)] = (seg(on), dom(on), seg(off), dom(off))
                ref = [x["bite"] for x in on if 2 * x["s2"] > .85]
                say(f"  {sch:13s} n {n} {gt:6s} (k 10 and 14): ON segregation {pct(seg(on))} dominance {pct(dom(on))} "
                    f"[{len(on)}]" + (f" refused {pct(np.mean(ref))}" if ref else "")
                    + f" | OFF segregation {pct(seg(off))} dominance {pct(dom(off))} [{len(off)}]")
    say("   Gillespie-like, n = 625, per graph family and mean degree:")
    on_c, off_c, ref_c, none = [], [], [], []
    for gt in IRREGULAR:
        for k in (10, 14):
            on, off = cell("gillespie_occ", 625, gt, True, k), cell("gillespie_occ", 625, gt, False, k)
            ref = [x["bite"] for x in on if 2 * x["s2"] > .85]
            (on_c if seg(on) > 0 else none).append(seg(on) if seg(on) > 0 else f"{gt} k {k}")
            off_c.append(seg(off))
            ref_c += [np.mean(ref)] if ref else []
            say(f"  {gt:6s} k {k}: ON segregation {pct(seg(on))} [{len(on)}]"
                + (f" refused {pct(np.mean(ref))}" if ref else "") + f" | OFF segregation {pct(seg(off))} [{len(off)}]")
    g625 = [S[("gillespie_occ", 625, gt)] for gt in IRREGULAR]
    say(f"  n = 625, Gillespie-like, k pooled: segregation ON {rng0([v[0] for v in g625])}, OFF "
        f"{rng0([v[2] for v in g625])}; dominance ON {rng0([v[1] for v in g625])}, OFF {rng0([v[3] for v in g625])}")
    say(f"  n = 625, Gillespie-like, per cell: segregation ON {rng0(on_c)} where it occurred (none in: "
        f"{', '.join(none) or '—'}), OFF {rng0(off_c)}; refused {rng0(ref_c)}")
    e = S[("sync", 625, "er")]
    say(f"  n = 625, synchronous, ER: segregation ON {pct(e[0])}, OFF {pct(e[2])}")
    say()


# ── 11. Three-player gate diagnostic ─────────────────────────────────────────
def section11():
    say("11. THREE-PLAYER GATE DIAGNOSTIC (gate_3p_diag_v1.pkl; gate paragraph; S8 caption)")
    R = load_diag("gate_3p_diag_v1.pkl")
    say(f"   {len(R)} games, policies {sorted({x['pol'] for x in R})}; genuine coexistence at n = 625, k = 10 and 14 "
        "pooled [games]")
    for contest, label in (("rps", "RPS"), ("all_vs_all", "AVA")):
        for sch in ("gillespie_occ", "sync"):
            out = []
            for gt in IRREGULAR:
                v = []
                for gate in (True, False):
                    c = [x for x in R if x["contest"] == contest and x["sch"] == sch and x["gt"] == gt
                         and x["n"] == 625 and bool(x["gate"]) == gate]
                    v.append(f"{pct(np.mean([genuine(x['surv'], x['jain']) for x in c]))} [{len(c)}]")
                out.append(f"{gt} ON {v[0]} OFF {v[1]}")
            say(f"  {label} {sch:13s}: " + " | ".join(out))
    say()


# ── 12. Exploration-rate sweep ───────────────────────────────────────────────
def section12():
    say("12. EXPLORATION-RATE SWEEP (eps_sweep_2p_v1.pkl, eps_sweep_rps_v1.pkl, eps_sweep_ava_v1.pkl; exploration and "
        "strategy paragraphs)")
    EPS = (0.02, 0.04, 0.07, 0.15)
    D = load_diag("eps_sweep_2p_v1.pkl")
    say("   two-player dominance (CI < .10) per exploration rate [games]")
    for gt in IRREGULAR:
        say(f"  {gt:6s}: " + " | ".join(
            f"eps {e}: {pct(np.mean([2 * x['s2'] < .10 for x in D if x['gt'] == gt and x['eps'] == e]))} "
            f"[{sum(x['gt'] == gt and x['eps'] == e for x in D)}]" for e in EPS))
    del D
    R = load_diag("eps_sweep_rps_v1.pkl")
    say("   RPS, genuine coexistence per exploration rate, n = 225 and 625 pooled [games] (regular graphs: 0.02 and 0.15 only)")
    for gt in REGULAR + IRREGULAR:
        say(f"  RPS {gt:7s}: " + " | ".join(
            f"eps {e}: {pct(np.mean([x['surv'] == 3 and x['jain'] >= .85 for x in R if x['gt'] == gt and x['eps'] == e]))} "
            f"[{sum(x['gt'] == gt and x['eps'] == e for x in R)}]" for e in EPS if any(x['gt'] == gt and x['eps'] == e for x in R)))
    del R
    A = load_diag("eps_sweep_ava_v1.pkl")
    say("   AVA, all three species present at the end, per exploration rate")
    for gt in IRREGULAR:
        say(f"  {gt:6s}: " + " | ".join(
            f"eps {e}: {pct(np.mean([x['surv'] == 3 for x in A if x['gt'] == gt and x['eps'] == e]))}" for e in EPS))
    say("   AVA, all three present, BBB minus AAA, all exploration rates pooled [games per policy]")
    for gt in IRREGULAR:
        b = [x["surv"] == 3 for x in A if x["gt"] == gt and x["pol"] == "BBB"]
        a = [x["surv"] == 3 for x in A if x["gt"] == gt and x["pol"] == "AAA"]
        say(f"  {gt:6s}: BBB {pct(np.mean(b))}, AAA {pct(np.mean(a))}, difference {100 * (np.mean(b) - np.mean(a)):+.1f} "
            f"percentage points [{len(b)}, {len(a)}]")
    say()


# ── 13. Mean outcome per policy combination (S1) ─────────────────────────────
def section13(rows):
    say("13. MEAN OUTCOME PER POLICY COMBINATION (S1's data and filters; S1 caption, strategy paragraph)")
    say("   two-player: mean CI; three-player: mean Jain index; all board sizes and mean degrees pooled")
    P2, P3 = ("AA", "AB", "BA", "BB"), ("AAA", "BAA", "BAB", "BBB")
    G = IRREGULAR + REGULAR
    panels = [("2P sync", [(x[1], x[4], x[6]) for x in rows["two-player"] if x[0] == "sync" and x[6] is not None], P2),
              ("2P gillespie", [(x[1], x[4], x[6]) for x in rows["two-player"]
                                if x[0] == "gillespie_occ" and x[6] is not None], P2),
              ("RPS", [(x[1], x[4], x[11]) for x in rows["RPS"] if x[11]], P3),
              ("AVA", [(x[1], x[4], x[11]) for x in rows["AVA"] if x[11]], P3)]
    for name, D, P in panels:
        M = {(g, p): np.mean([v for gt, pol, v in D if gt == g and pol == p]) for g in G for p in P}
        say(f"== {name}")
        for g in G:
            v = [M[(g, p)] for p in P]
            say(f"  {g:7s} " + " ".join(f"{p} {x:.4f}" for p, x in zip(P, v))
                + f" | spread {max(v) - min(v):.3f} | {P[0]}->{P[-1]} {v[-1] - v[0]:+.3f}")
        for p in P:
            lo_reg = min(M[(g, p)] for g in REGULAR)
            hi_irr = max(M[(g, p)] for g in IRREGULAR)
            say(f"  {p}: lowest regular {lo_reg:.4f} vs highest irregular {hi_irr:.4f} -> "
                f"{'regular above' if lo_reg > hi_irr else 'NOT above'} (margin {lo_reg - hi_irr:+.4f})")
        if name == "2P sync":
            sync_aa = (M[("torus", "AA")], M[("er", "AA")])
    say(f"   2P sync, AA, the pair S1's caption calls level: torus {sync_aa[0]:.5f}, ER {sync_aa[1]:.5f}")
    say()


# ── 14. S4 group counts; S9 survival gap and fill pace (added 2026-11-03) ───
def section14():
    say("14. S4 AND S9 CAPTIONS, EXPLORATION-RATE PARAGRAPH (two_p_all_v1.pkl; eps_sweep_ava_v1.pkl)")
    with open("results/two_p_all_v1.pkl", "rb") as fh:
        R = pickle.load(fh)
    say("   S4: two-player segregation runs (CI > .85), ER / SBM / DC-SBM, all schedulers, k, n and policies pooled;")
    say("   winner = the colour with more stones (as plot_eye_diagnostic.py); stone groups at the final turn")
    for gt in IRREGULAR:
        w, l = [], []
        for r in R:
            if r.get("connectivity_failed") or r["_graph_type"] != gt or r.get("black_occupancy") is None:
                continue
            if ci_2p(r) <= .85:
                continue
            b_wins = r["black_occupancy"] >= r["white_occupancy"]
            w.append(r.get("black_groups" if b_wins else "white_groups", 0))
            l.append(r.get("white_groups" if b_wins else "black_groups", 0))
        for name, v in (("winner", np.array(w)), ("loser", np.array(l))):
            say(f"  {gt:6s} {name:6s}: median {np.median(v):.0f}, two or more groups {pct(np.mean(v >= 2))}, "
                f"max {v.max()} [{len(v)}]")
    del R
    gc.collect()
    with open("results/eps_sweep_ava_v1.pkl", "rb") as fh:
        A = [r for r in pickle.load(fh) if r["_graph_type"] == "dc_sbm" and r["_eps"] in (0.02, 0.15)]
    gc.collect()
    cols, every = (1, -1, 2), 20                         # occ_ts_by_color: one snapshot every 20 turns, 250 snapshots
    say("   S9 (left panel's rule): share of DC-SBM AVA runs with all three species above 0.1% of the board [runs]")
    alive = {}
    for e in (0.02, 0.15):
        R = [r for r in A if r["_eps"] == e]
        alive[e] = np.mean([[all(r["occ_ts_by_color"][c][i] > 0.001 for c in cols) for i in range(250)] for r in R],
                           axis=0)
        say(f"  eps {e}: " + ", ".join(f"turn {t}: {pct(alive[e][t // every])}" for t in (500, 2000, 4980))
            + f" [{len(R)}]")
    say("  gap (eps .15 minus .02), percentage points: " + ", ".join(
        f"turn {t}: {100 * (alive[0.15][t // every] - alive[0.02][t // every]):+.1f}" for t in (500, 2000, 4980)))
    say("   S9 right panel: mean occupancy per species (total / 3) at turn 200, by final presence and board size [runs]")
    for e in (0.02, 0.15):
        for n in (225, 625):
            out = []
            for name, keep in (("all present", lambda r: r.get("survivors_count", 3) == 3),
                               ("species lost", lambda r: r.get("survivors_count", 3) < 3)):
                R = [r for r in A if r["_eps"] == e and r["_n_nodes"] == n and keep(r)]
                m = np.mean([np.mean([r["occ_ts_by_color"][c][200 // every] for c in cols]) for r in R]) if R else np.nan
                out.append(f"{name} {m:.3f} [{len(R)}]")
            say(f"  eps {e} n {n}: " + " | ".join(out))
    del A
    gc.collect()
    say()


def main():
    say(__doc__.strip().split("\n\n")[0])
    say()
    rows, skips = {}, {}
    for name, fname, cols in CONTESTS:
        rows[name], skips[name] = load(fname, cols)
    with open("results/sync_3p_diag_v1.pkl", "rb") as fh:
        S3 = pickle.load(fh)
    sync3 = {"RPS": [], "AVA": []}
    for r in S3:
        if not r.get("connectivity_failed"):
            sync3["RPS" if r["_contest"] == "rps" else "AVA"].append(
                (r["_scheduler"], r["_graph_type"], r["_grid_size"], r["_target_k"], r.get("_policy"),
                 s_min(r, (1, -1, 2)), None, r.get("captures_n"), r.get("moves"), r.get("capture_rate_per_move"),
                 r.get("survivors_count"), r.get("jain_fairness_final")))
    del S3
    say("rows kept: " + ", ".join(f"{n} {len(rows[n])} (connectivity failures dropped: {skips[n]})" for n in rows)
        + f", synchronous three-player diagnostic {len(sync3['RPS']) + len(sync3['AVA'])}")
    say()
    section1(rows)
    section2(rows, sync3)
    section3(rows)
    section4(rows)
    section5(rows)
    section6(rows)
    section7(rows)
    section8(rows)
    section9(rows, sync3)
    section10()
    section11()
    section12()
    section13(rows)
    section14()
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
