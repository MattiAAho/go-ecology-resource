"""Supplementary Figure S11 — convergence of the outcome statistic over simulation time.

Median + 10-90 percentile band of the outcome statistic (Jain index for 3p,
stones-normalised CI for 2p) vs turn, from the production occupancy trajectories.
Async (Gillespie-like) panels settle to a quasi-stationary value; sync 2p oscillates.
"""
from __future__ import annotations

import gc
import pickle

import matplotlib.pyplot as plt
import numpy as np

MID_T = 2500


def stat_traj(rec, mode):
    d = rec["occ_ts_by_color"]
    M = np.array([d[c] for c in sorted(d.keys())], dtype=float).T  # snaps x colors
    if mode == "2p":
        b, w = M[:, 0], M[:, 1]
        tot = b + w
        return np.where(tot > 0, 1.0 - np.abs(b - w) / np.where(tot > 0, tot, 1), np.nan)
    tot = M.sum(axis=1)
    ssq = (M ** 2).sum(axis=1)
    return np.where(tot > 0, (tot ** 2) / (M.shape[1] * np.where(ssq > 0, ssq, 1)), np.nan)


def collect(files, mode, sched=None):
    trajs, tref = [], None
    for f in files:
        recs = pickle.load(open(f, "rb"))
        for r in recs:
            if r.get("occ_ts_by_color") is None:
                continue
            if sched is not None and r.get("_scheduler") != sched:
                continue
            s = stat_traj(r, mode)
            if len(s) == 250:
                trajs.append(s)
                if tref is None:
                    tref = np.asarray(r["occ_t_ts"], dtype=int)
        del recs
        gc.collect()
    return np.vstack(trajs), tref


FULL = ["results/full_er_sbm_v2.pkl", "results/full_dcsbm_v2.pkl", "results/full_fixed_v2.pkl"]
AVA = ["results/ava_er_sbm_v1.pkl", "results/ava_dcsbm_v1.pkl", "results/ava_fixed_v1.pkl"]
TWOP = ["results/two_p_er_sbm_v1.pkl", "results/two_p_dcsbm_v1.pkl", "results/two_p_fixed_v1.pkl"]

panels = [
    ("(a) Three-player RPS — Gillespie-like", FULL, "rps", "gillespie_occ", "Jain coexistence index"),
    ("(b) Three-player AVA — Gillespie-like", AVA, "ava", "gillespie_occ", "Jain coexistence index"),
    ("(c) Two-player — Gillespie-like", TWOP, "2p", "gillespie_occ", "Coexistence index"),
    ("(d) Two-player — synchronous", TWOP, "2p", "sync", "Coexistence index"),
]

fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
for col, (title, files, mode, sched, ylab) in enumerate(panels):
    A, t = collect(files, mode, sched)
    final = np.nanmean(A[:, -25:], axis=1, keepdims=True)   # per-run asymptote (last 10%)

    # row 1: statistic median + 10-90 band
    ax = axes[0, col]
    ax.fill_between(t, np.nanpercentile(A, 10, axis=0), np.nanpercentile(A, 90, axis=0),
                    alpha=0.25, color="#1f77b4", label="10–90th pct")
    ax.plot(t, np.nanmedian(A, axis=0), color="#1f77b4", lw=1.8, label="median")
    ax.axvline(MID_T, ls="--", color="0.4", lw=1, label="midpoint (2,500)")
    ax.set_title(f"{title}\n(n = {A.shape[0]:,})", fontsize=9)
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 1.02)
    if col == 0:
        ax.set_ylabel("Outcome statistic")
        ax.legend(fontsize=7, loc="lower right")

    # row 2: convergence — per-run |statistic - final value|
    dev = np.abs(A - final)
    ax2 = axes[1, col]
    ax2.fill_between(t, np.nanpercentile(dev, 50, axis=0), np.nanpercentile(dev, 90, axis=0),
                     alpha=0.25, color="#d62728")
    ax2.plot(t, np.nanmedian(dev, axis=0), color="#d62728", lw=1.8, label="median deviation")
    ax2.axhline(0.05, ls=":", color="0.4", lw=1, label="0.05 tolerance")
    ax2.axvline(MID_T, ls="--", color="0.4", lw=1)
    ax2.set_xlabel("turn")
    ax2.set_xlim(0, 5000)
    ax2.set_ylim(0, 0.55)
    if col == 0:
        ax2.set_ylabel("|statistic − final|")
        ax2.legend(fontsize=7, loc="upper right")
    print(f"{title}: n={A.shape[0]}  median-dev@2500={np.nanmedian(dev[:, np.argmin(np.abs(t-MID_T))]):.4f}")

fig.tight_layout()
fig.savefig("figures/final/S11.pdf", bbox_inches="tight")
fig.savefig("figures/final/S11.png", dpi=150, bbox_inches="tight")
print("saved figures/final/S11.pdf and .png")
plt.close(fig)
