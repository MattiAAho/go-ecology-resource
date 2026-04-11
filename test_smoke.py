"""
Smoke test — run after removing dead code from go_engine.py and run_one_game.py.

Checks:
  1. Imports from go_engine still work (fragmentation_index present, compute_network_metrics gone)
  2. run_one_game completes with store_metrics=True (exercises compute_network_metrics in run_one_game)
  3. fragmentation_index is called correctly inside run_one_game (via frag= at end of game)
  4. gillespie_occ scheduler completes a short run
"""

import networkx as nx
import numpy as np

# ── 1. go_engine imports ──────────────────────────────────────────────────────
from go_engine import fragmentation_index, GameState

# compute_network_metrics should NOT be importable from go_engine after cleanup
try:
    from go_engine import compute_network_metrics as _cnm_engine
    print("WARN: go_engine.compute_network_metrics still present (not yet removed)")
except ImportError:
    print("OK  : go_engine.compute_network_metrics removed")

# ── 2. run_one_game imports ───────────────────────────────────────────────────
from run_one_game import run_one_game, make_game_from_graph, compute_network_metrics

# pick_color_* and run_benchmark should NOT be importable after cleanup
for name in ("pick_color_from_occupancy", "pick_color_balanced", "pick_color_energy", "run_benchmark"):
    try:
        import run_one_game as _m
        getattr(_m, name)
        print(f"WARN: run_one_game.{name} still present (not yet removed)")
    except AttributeError:
        print(f"OK  : run_one_game.{name} removed")

# ── 3. short two-player sync game with store_metrics=True ────────────────────
G = nx.erdos_renyi_graph(25, 0.15, seed=42)
while not nx.is_connected(G):
    G = nx.erdos_renyi_graph(25, 0.15, seed=G.number_of_edges())

r = run_one_game(
    G,
    n_moves=200,
    store_metrics=True,
    store_replay=False,
    store_timeseries=False,
    scheduler="sync",
)
assert "black_fragmentation" in r, "black_fragmentation missing from result"
assert "capture_rate_per_move" in r
print(f"OK  : sync game completed, black_fragmentation={r['black_fragmentation']:.3f}")

# ── 4. short gillespie_occ game ───────────────────────────────────────────────
r2 = run_one_game(
    G,
    n_moves=200,
    store_metrics=False,
    store_replay=False,
    store_timeseries=False,
    scheduler="gillespie_occ",
)
assert "coexistence_index" in r2
print(f"OK  : gillespie_occ game completed, ci={r2['coexistence_index']:.3f}")

# ── 5. three-player game ──────────────────────────────────────────────────────
r3 = run_one_game(
    G,
    n_moves=200,
    colors=(+1, -1, +2),
    store_metrics=False,
    store_replay=False,
    store_timeseries=False,
    scheduler="sync",
)
assert "coexistence_index" in r3
print(f"OK  : three-player game completed, ci={r3['coexistence_index']:.3f}")

print("\nAll smoke tests passed.")
