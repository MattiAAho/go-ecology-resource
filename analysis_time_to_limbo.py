"""Time-to-limbo / convergence analysis for manuscript-1 Methods.

Answers the question "what locks the results, why 5000 steps?" from EXISTING
production occupancy trajectories (occ_ts_by_color, every 20 turns). No re-run.

Macro-observable = the outcome statistic actually used in the paper:
  3p (rps/ava): Jain coexistence index J(t)
  2p:           CI(t) = 1 - |b-w|/(b+w)   (stones-normalised, per CLAUDE.md)

Per run:
  final_val  = mean of statistic over last 10% of snapshots (asymptote/centre)
  settle_t   = first turn after which |stat - final_val| <= TOL for ALL later snaps
  osc_std    = std of statistic over last 30% of snapshots (oscillation amplitude)
  stat_mid   = statistic at t~=2500  (eps-stationarity test vs final_val)

Split by SCHEDULER (sync vs gillespie_occ) because sync 2p oscillates (round-robin
bimodality) while async converges.
"""

from __future__ import annotations

import collections
import gc
import pickle

import numpy as np

TOL = 0.05            # outcome-statistic tolerance for "settled"
OSC_CONV = 0.05       # late-window std below this => converged (not oscillating)
MID_T = 2500


def stat_traj(rec, mode):
    d = rec["occ_ts_by_color"]
    cols = sorted(d.keys())
    M = np.array([d[c] for c in cols], dtype=float).T  # snaps x colors
    t = np.asarray(rec["occ_t_ts"], dtype=int)
    if mode == "2p":
        b, w = M[:, 0], M[:, 1]
        tot = b + w
        s = np.where(tot > 0, 1.0 - np.abs(b - w) / np.where(tot > 0, tot, 1), np.nan)
    else:
        tot = M.sum(axis=1)
        ssq = (M ** 2).sum(axis=1)
        s = np.where(tot > 0, (tot ** 2) / (M.shape[1] * np.where(ssq > 0, ssq, 1)), np.nan)
    return s, t


def per_run(rec, mode):
    s, t = stat_traj(rec, mode)
    n = len(s)
    last10 = max(1, n // 10)
    last30 = max(1, (3 * n) // 10)
    final_val = np.nanmean(s[-last10:])
    osc_std = np.nanstd(s[-last30:])
    dev = np.abs(s - final_val)
    ok = dev <= TOL
    bad = np.where(~ok)[0]
    idx = 0 if bad.size == 0 else min(bad[-1] + 1, n - 1)
    settle_t = int(t[idx])
    mid_i = int(np.argmin(np.abs(t - MID_T)))
    return {
        "gtype": rec.get("_graph_type"),
        "sched": rec.get("_scheduler"),
        "n_nodes": rec.get("_n_nodes"),
        "settle_t": settle_t,
        "osc_std": float(osc_std),
        "final_val": float(final_val),
        "stat_mid": float(s[mid_i]),
        "sat_t": rec.get("sat_t"),
    }


def analyse_file(path, mode):
    recs = pickle.load(open(path, "rb"))
    out = [per_run(r, mode) for r in recs if r.get("occ_ts_by_color") is not None]
    del recs
    gc.collect()
    return out


def summarize(rows, mode):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["sched"], r["gtype"])].append(r)
    print(f"\n================ MODE = {mode}  (n={len(rows)}) ================")
    print(f"{'sched':<13} {'topo':<8} {'n':>5} | {'settle_t turns':^18} | conv% "
          f"| osc_std | |mid-final|")
    print(f"{'':<13} {'':<8} {'':>5} | {'med':>5} {'p90':>5} {'p95':>5} |       "
          f"|         |")
    print("-" * 86)
    for sched in sorted(set(s for s, _ in by)):
        for g in sorted(set(g for s, g in by if s == sched)):
            rs = by[(sched, g)]
            st = np.array([x["settle_t"] for x in rs])
            conv = np.array([x["osc_std"] < OSC_CONV for x in rs])
            osc = np.array([x["osc_std"] for x in rs])
            diff = np.abs(np.array([x["stat_mid"] - x["final_val"] for x in rs]))
            n = int(np.round(np.median([x["n_nodes"] for x in rs])))
            print(f"{sched:<13} {g:<8} {n:>5} | {np.median(st):>5.0f} "
                  f"{np.percentile(st,90):>5.0f} {np.percentile(st,95):>5.0f} | "
                  f"{100*conv.mean():>4.0f}% | {np.mean(osc):>7.4f} | {np.nanmean(diff):>7.4f}")
        # scheduler rollup
        rs = [x for x in rows if x["sched"] == sched]
        st = np.array([x["settle_t"] for x in rs])
        conv = np.array([x["osc_std"] < OSC_CONV for x in rs])
        diff = np.abs(np.array([x["stat_mid"] - x["final_val"] for x in rs]))
        print(f"{sched+' ALL':<13} {'':<8} {'':>5} | {np.median(st):>5.0f} "
              f"{np.percentile(st,90):>5.0f} {np.percentile(st,95):>5.0f} | "
              f"{100*conv.mean():>4.0f}% | {'':>7} | {np.nanmean(diff):>7.4f}")
        print("-" * 86)


FILES = {
    "rps": ["results/full_er_sbm_v2.pkl", "results/full_dcsbm_v2.pkl", "results/full_fixed_v2.pkl"],
    "ava": ["results/ava_er_sbm_v1.pkl", "results/ava_dcsbm_v1.pkl", "results/ava_fixed_v1.pkl"],
    "2p":  ["results/two_p_er_sbm_v1.pkl", "results/two_p_dcsbm_v1.pkl", "results/two_p_fixed_v1.pkl"],
}

if __name__ == "__main__":
    for mode, files in FILES.items():
        rows = []
        for f in files:
            print(f"  loading {f} ...", flush=True)
            rows += analyse_file(f, mode)
        summarize(rows, mode)
