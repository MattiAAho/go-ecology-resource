"""
describe_m1_text_numbers.py — Numbers quoted in the manuscript text and tables that no figure shows.

Reads the production rows (no games are run) and prints, in five sections, the numbers behind:
  1. Table 4 (Claim 3 rows) and the three-player Results: the weakest species' share of the final stones,
     regular vs irregular graphs, per competition structure and board size.
  2. The scheduler-robustness paragraph: a species absent at the end, synchronous vs Gillespie-like, on the cells
     where all three competition structures were run under both schedulers (ER and SBM, k = 10 and 14).
  3. Figure 2's caption and Table 4 (Claim 1 rows): two-player dominance rates (CI < 0.10) per cell.
  4. The two-player bimodality paragraph and Figure 4's caption: the dominance (CI < 0.10) and segregation (CI > 0.85)
     shares per graph family and per mean degree.
  5. The behavioural-strategy paragraph: whether "regular graphs end more even than irregular graphs" holds within each
     of the four policy mixes, per board size and competition structure.

Definitions:
  s_min  = the smallest species' share of the stones at the final turn (a species with no stones counts as 0);
           two-player CI = 2 s_min (= ci_2p), so CI < 0.10 is s_min < 0.05 and CI > 0.85 is s_min > 0.425.
  Rows flagged connectivity_failed are dropped. REGULAR = lattice, torus; IRREGULAR = ER, SBM, DC-SBM (all mean degrees).
  Sections 1, 4 (beside) and 5 use the energy-weighted asynchronous, Gillespie-like scheduler (`gillespie_occ`) only.
  Shapes: games per s_min bin
    three-player  0 | (0, .05] | (.05, .10] | (.10, .20] | (.20, 1/3]
    two-player    0 | (0, .05] | (.05, .20] | (.20, .425] | (.425, .5]

Requires results/two_p_all_v1.pkl, full_all_v2.pkl, ava_all_v1.pkl (Steps 2-4) and sync_3p_diag_v1.pkl (Step 6).

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
    """Compact rows: (scheduler, graph type, grid, target k, policy, s_min, ci_2p or None), connectivity failures out."""
    with open(f"results/{fname}", "rb") as fh:
        R = pickle.load(fh)
    out, skipped = [], Counter()
    for r in R:
        if r.get("connectivity_failed"):
            skipped[r["_scheduler"]] += 1
            continue
        ci = ci_2p(r) if len(cols) == 2 and r.get("black_occupancy") is not None else None
        out.append((r["_scheduler"], r["_graph_type"], r["_grid_size"], r["_target_k"], r.get("_policy"),
                    s_min(r, cols), ci))
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
    for sch, gt, g, k, _, _, ci in rows["two-player"]:
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
        for sch, gt, g, k, pol, sm, _ in rows[name]:
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
                 s_min(r, (1, -1, 2)), None))
    del S3
    say("rows kept: " + ", ".join(f"{n} {len(rows[n])} (connectivity failures dropped: {skips[n]})" for n in rows)
        + f", synchronous three-player diagnostic {len(sync3['RPS']) + len(sync3['AVA'])}")
    say()
    section1(rows)
    section2(rows, sync3)
    section3(rows)
    section4(rows)
    section5(rows)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
