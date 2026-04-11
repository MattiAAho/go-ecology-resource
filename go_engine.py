"""
go_engine.py — compact Go-like engine on arbitrary NetworkX graphs

Design goals
- Works on ANY undirected graph G where nodes are hashable.
- State is a dict: node -> {0 empty, +1 black, -1 white}.
- Implements: groups, liberties, capture, suicide (no ko by default).
- Provides: legal_moves(), play_move() with rollback safety, and a simple pluggable policy API.
- Grid-specific heuristics (eyes/border) are OPTIONAL and only used if you supply node coordinates.

Notes
- This is "Go-like" on graphs: capture/suicide rules apply; ko and superko are optional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, List, Optional, Set, Tuple, Callable, Literal
import copy as _copy
from collections import deque
from enum import IntEnum
from contextlib import contextmanager

import numpy as np
import networkx as nx
import math
from numba import njit

Node = Hashable
Color = int  # +1 black, -1 white
State = Dict[Node, int]

# ----------------------------
# CSR adjacency builder
# ----------------------------

def _build_csr(idx_to_node, node_to_idx, neigh):
    """Convert neigh dict-of-lists into CSR flat_adj / offsets arrays (int32)."""
    n = len(idx_to_node)
    nb_lists = [[node_to_idx[w] for w in neigh[v]] for v in idx_to_node]
    deg = np.array([len(nb) for nb in nb_lists], dtype=np.int32)
    offsets = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(deg, out=offsets[1:])
    flat_adj = np.empty(int(offsets[-1]), dtype=np.int32)
    for i, nb in enumerate(nb_lists):
        flat_adj[offsets[i]:offsets[i + 1]] = nb
    return flat_adj, offsets


# ----------------------------
# Numba JIT BFS kernel
# ----------------------------

@njit(cache=True)
def _bfs_group_libs(seed_idx, color, state_arr, flat_adj, offsets):
    """
    BFS from seed_idx over nodes of the same color.
    Returns (comp_indices, lib_indices) as int32 arrays.
    comp: nodes belonging to the group.
    libs: empty (state==0) neighbor nodes of the group.
    """
    n = len(state_arr)
    comp = np.empty(n, dtype=np.int32)
    libs = np.empty(n, dtype=np.int32)
    comp_n = np.int32(0)
    libs_n = np.int32(0)
    seen = np.zeros(n, dtype=np.bool_)
    queue = np.empty(n, dtype=np.int32)
    q_head = np.int32(0)
    q_tail = np.int32(0)

    seen[seed_idx] = True
    queue[q_tail] = seed_idx
    q_tail += np.int32(1)

    while q_head < q_tail:
        u = queue[q_head]
        q_head += np.int32(1)
        comp[comp_n] = u
        comp_n += np.int32(1)
        for k in range(offsets[u], offsets[u + 1]):
            w = flat_adj[k]
            wc = state_arr[w]
            if wc == 0:
                if not seen[w]:
                    seen[w] = True
                    libs[libs_n] = w
                    libs_n += np.int32(1)
            elif wc == color and not seen[w]:
                seen[w] = True
                queue[q_tail] = w
                q_tail += np.int32(1)

    return comp[:comp_n], libs[:libs_n]


@njit(cache=True)
def _bfs_count_only(seed_idx, color, state_arr, flat_adj, offsets):
    """
    Like _bfs_group_libs but returns only (comp_size, lib_count) as ints.
    No array allocation for results — used in the policy scoring hot path.
    """
    n = len(state_arr)
    seen = np.zeros(n, dtype=np.bool_)
    queue = np.empty(n, dtype=np.int32)
    q_head = np.int32(0)
    q_tail = np.int32(0)
    comp_n = np.int32(0)
    libs_n = np.int32(0)

    seen[seed_idx] = True
    queue[q_tail] = seed_idx
    q_tail += np.int32(1)

    while q_head < q_tail:
        u = queue[q_head]
        q_head += np.int32(1)
        comp_n += np.int32(1)
        for k in range(offsets[u], offsets[u + 1]):
            w = flat_adj[k]
            wc = state_arr[w]
            if wc == 0:
                if not seen[w]:
                    seen[w] = True
                    libs_n += np.int32(1)
            elif wc == color and not seen[w]:
                seen[w] = True
                queue[q_tail] = w
                q_tail += np.int32(1)

    return comp_n, libs_n


def group_size_and_lib_count(gs, v):
    """
    Returns (color, comp_size, lib_count) without building Python sets.
    Use this in scoring/policy paths where only the counts matter.
    Returns (0, 0, 0) if v is empty.
    """
    v_idx = gs.node_to_idx[v]
    color = int(gs.state_arr[v_idx])
    if color == 0:
        return 0, 0, 0
    comp_n, libs_n = _bfs_count_only(
        np.int32(v_idx), np.int8(color), gs.state_arr, gs.flat_adj, gs.offsets
    )
    return color, int(comp_n), int(libs_n)


class MoveStatus(IntEnum):
    PASS = 0
    ILLEGAL = 1
    EXECUTED = 2
    FAILED_GATE = 3

# ----------------------------
# Core engine
# ----------------------------

@dataclass
class GameState:
    G: nx.Graph
    state: State
    coords: Optional[Dict[Node, Tuple[float, float]]] = None
    ko: bool = False
    _prev_hash: Optional[int] = field(default=None, init=False)

    # N-player identity
    colors: Tuple[int, ...] = (+1, -1)
    # prey_of[c] = tuple of colors that c can capture.  None → all others.
    prey_of: Optional[Dict[int, Tuple[int, ...]]] = None

    # counters
    n_nodes: int = 0
    pass_streak: int = 0
    # n_stones built in __post_init__; not in constructor
    n_stones: Optional[Dict[int, int]] = field(default=None, init=False, repr=False, compare=False)

    # --- env knobs ---
    use_gate: bool = False
    gate_delta: float = 1.0
    gate_hops: int = 2
    R: Optional[Dict[Node, float]] = None

    # --- caches (NOT in constructor) ---
    neigh: Optional[Dict[Node, List[Node]]] = field(
        default=None,
        init=False,   # ✅ do not pass in constructor
        repr=False,
        compare=False
    )
    deg: Optional[Dict[Node, int]] = field(
        default=None,
        init=False,   # ✅ do not pass in constructor
        repr=False,
        compare=False
    )
    deg_mu: Optional[float] = field(default=None, init=False, repr=False, compare=False)
    deg_sd: Optional[float] = field(default=None, init=False, repr=False, compare=False)

    # --- fast representation (CSR adjacency + state array) ---
    node_to_idx: Optional[Dict] = field(default=None, init=False, repr=False, compare=False)
    idx_to_node: Optional[List] = field(default=None, init=False, repr=False, compare=False)
    state_arr: Optional[np.ndarray] = field(default=None, init=False, repr=False, compare=False)
    flat_adj: Optional[np.ndarray] = field(default=None, init=False, repr=False, compare=False)
    offsets: Optional[np.ndarray] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.neigh is None:
            self.neigh = {u: list(self.G.neighbors(u)) for u in self.G.nodes()}
        if self.deg is None:
            self.deg = {u: len(self.neigh[u]) for u in self.neigh}
        if self.deg_mu is None or self.deg_sd is None:
            degs = np.fromiter((self.deg[u] for u in self.G.nodes()), dtype=float)
            self.deg_mu = float(degs.mean())
            self.deg_sd = float(degs.std()) + 1e-9
        # build node ordering and CSR
        if self.node_to_idx is None:
            self.idx_to_node = list(self.G.nodes())
            self.node_to_idx = {v: i for i, v in enumerate(self.idx_to_node)}
        if self.state_arr is None:
            n = len(self.idx_to_node)
            self.state_arr = np.zeros(n, dtype=np.int8)
            for i, v in enumerate(self.idx_to_node):
                self.state_arr[i] = self.state.get(v, 0)
        if self.flat_adj is None:
            self.flat_adj, self.offsets = _build_csr(
                self.idx_to_node, self.node_to_idx, self.neigh
            )
        if self.n_stones is None:
            self.n_stones = {c: sum(1 for v in self.G.nodes() if self.state.get(v, 0) == c)
                             for c in self.colors}

        # --- prey_of validation ---
        if self.prey_of is not None:
            color_set = set(self.colors)
            bad_keys = set(self.prey_of.keys()) - color_set
            if bad_keys:
                raise ValueError(
                    f"prey_of contains keys not in colors: {bad_keys}"
                )
            for predator, preys in self.prey_of.items():
                bad_vals = set(preys) - color_set
                if bad_vals:
                    raise ValueError(
                        f"prey_of[{predator!r}] contains colors not in colors: {bad_vals}"
                    )
                if predator in preys:
                    raise ValueError(
                        f"prey_of[{predator!r}] contains the predator itself (self-predation not allowed)"
                    )

    def board_hash(self) -> int:
        nodes = sorted(self.G.nodes(), key=repr)
        return hash(tuple((v, self.state.get(v, 0)) for v in nodes))

    def copy(self) -> "GameState":
        # Use __new__ to skip __post_init__ (avoids rebuilding caches on every policy copy)
        gs = GameState.__new__(GameState)
        gs.G = self.G
        gs.state = dict(self.state)
        gs.coords = self.coords
        gs.ko = self.ko
        gs._prev_hash = self._prev_hash
        gs.colors = self.colors          # immutable tuple, safe to share
        gs.prey_of = self.prey_of        # None or immutable dict of tuples
        gs.n_nodes = self.n_nodes
        gs.n_stones = dict(self.n_stones)
        gs.pass_streak = self.pass_streak
        gs.use_gate = self.use_gate
        gs.gate_delta = self.gate_delta
        gs.gate_hops = self.gate_hops
        gs.R = None if self.R is None else dict(self.R)
        # shared immutable graph-derived caches
        gs.neigh = self.neigh
        gs.deg = self.deg
        gs.deg_mu = self.deg_mu
        gs.deg_sd = self.deg_sd
        # shared immutable CSR and node mappings
        gs.node_to_idx = self.node_to_idx
        gs.idx_to_node = self.idx_to_node
        gs.flat_adj = self.flat_adj
        gs.offsets = self.offsets
        # copy mutable state array
        gs.state_arr = self.state_arr.copy()
        return gs

def clamp01(x: float) -> float:
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    return float(x)

def update_R_local(
    gs: GameState,
    *,
    v_played,
    dead: Optional[Set[Node]],
    alpha: float,
    beta: float,
    neigh_wear: float = 0.25,
):
    """
    Cheap local resource update after an event.
    - Wear on occupied nodes in the contested neighborhood
    - Renewal on empty nodes in the same neighborhood
    """
    if gs.R is None:
        return

    R = gs.R
    touched = set()

    if v_played is not None:
        touched.add(v_played)
        touched.update(gs.neigh[v_played])

    if dead:
        for u in dead:
            touched.add(u)
            touched.update(gs.neigh[u])

    a = float(alpha)
    b = float(beta)
    nw = float(neigh_wear)

    for u in touched:
        occ = gs.state.get(u, 0) != 0
        if occ:
            # depletion (optionally weaker on neighbors)
            wear = a if u == v_played else a * nw
            R[u] = clamp01(R[u] * (1.0 - wear))
            # If you want stronger wear on the focal played node only, handle it separately.
            # R[u] = clamp01(R[u] * (1.0 - a * nw))
        else:
            # renewal toward 1
            R[u] = clamp01(R[u] + b * (1.0 - R[u]))

def update_R_global(
    gs: GameState,
    *,
    alpha: float,
    beta: float,
):
    """
    Occasional full-field update (amortized).
    Wear on occupied nodes, renewal on empty nodes.
    """
    if gs.R is None:
        return

    R = gs.R
    a = float(alpha)
    b = float(beta)

    for u in gs.G.nodes():
        if gs.state.get(u, 0) != 0:
            R[u] = clamp01(R[u] * (1.0 - a))
        else:
            R[u] = clamp01(R[u] + b * (1.0 - R[u]))

    
@dataclass
class Undo:
    v: Any
    prev_v: int
    dead_by_color: Dict[int, Set]   # prey_color -> set of captured nodes
    n_stones: Dict[int, int]        # snapshot before the move
    prev_hash: Optional[int]

def play_move_with_undo(gs, v, color, rng) -> Tuple[int, int, Optional["Undo"]]:
    """
    Execute move (with gate) and if EXECUTED return Undo for reverting.
    """
    prev_v = gs.state.get(v, 0)
    n_stones0 = dict(gs.n_stones)
    prev_hash0 = gs._prev_hash

    status, cap, dead_by_color = play_move(gs, v, color, rng, return_dead=True)

    if status != MoveStatus.EXECUTED:
        return status, cap, None

    return status, cap, Undo(
        v=v,
        prev_v=prev_v,
        dead_by_color=dead_by_color,
        n_stones=n_stones0,
        prev_hash=prev_hash0,
    )

def undo_move(gs, undo: "Undo") -> None:
    """
    Revert an EXECUTED move applied by play_move_with_undo.
    Restores gs.state, gs.state_arr, counts, and ko hash.
    """
    nti = gs.node_to_idx
    # restore captured stones (each color restored to its own value)
    for prey_c, dead_set in undo.dead_by_color.items():
        for u in dead_set:
            gs.state[u] = prey_c
            gs.state_arr[nti[u]] = prey_c

    # restore played vertex
    if undo.prev_v == 0:
        gs.state[undo.v] = 0
        gs.state_arr[nti[undo.v]] = 0
    else:
        gs.state[undo.v] = undo.prev_v
        gs.state_arr[nti[undo.v]] = undo.prev_v

    # restore counters and ko hash
    gs.n_stones = dict(undo.n_stones)
    gs._prev_hash = undo.prev_hash
    
    
def ensure_R(gs: GameState, init_value: float = 1.0) -> None:
    if gs.R is None:
        gs.R = {v: float(init_value) for v in gs.G.nodes()}
    
def empty_nodes(gs: GameState) -> List[Node]:
    return [v for v in gs.G.nodes() if gs.state.get(v, 0) == 0]


def induced_subgraph_nodes(gs: GameState, color: Color) -> List[Node]:
    return [v for v in gs.G.nodes() if gs.state.get(v, 0) == color]


def groups_and_liberties(gs: GameState, color: Color):
    visited = set()
    out = []
    color8 = np.int8(color)
    itn = gs.idx_to_node
    nti = gs.node_to_idx

    for v, c in gs.state.items():
        if c != color or v in visited:
            continue
        comp_arr, libs_arr = _bfs_group_libs(
            np.int32(nti[v]), color8, gs.state_arr, gs.flat_adj, gs.offsets
        )
        comp = {itn[int(i)] for i in comp_arr}
        libs = {itn[int(i)] for i in libs_arr}
        visited |= comp
        out.append((comp, libs))

    return out


def group_and_liberties_of_node(gs, v) -> Tuple[int, Optional[Set], Optional[Set]]:
    """If v is empty -> (0, None, None). Else -> (color, group_set, liberties_set)."""
    v_idx = gs.node_to_idx[v]
    color = int(gs.state_arr[v_idx])
    if color == 0:
        return 0, None, None
    comp_arr, libs_arr = _bfs_group_libs(
        np.int32(v_idx), np.int8(color), gs.state_arr, gs.flat_adj, gs.offsets
    )
    itn = gs.idx_to_node
    comp = {itn[int(i)] for i in comp_arr}
    libs = {itn[int(i)] for i in libs_arr}
    return color, comp, libs


def bfs_group_and_liberties(gs, seed, color) -> Tuple[Set, Set]:
    """BFS component of `color` from seed; returns (group_set, liberties_set)."""
    comp_arr, libs_arr = _bfs_group_libs(
        np.int32(gs.node_to_idx[seed]), np.int8(color), gs.state_arr, gs.flat_adj, gs.offsets
    )
    itn = gs.idx_to_node
    comp = {itn[int(i)] for i in comp_arr}
    libs = {itn[int(i)] for i in libs_arr}
    return comp, libs


def captured_components(gs: GameState, color: Color) -> List[Set[Node]]:
    """
    Return list of components (as sets) of `color` that have 0 liberties.
    """
    comps: List[Set[Node]] = []
    for comp, libs in groups_and_liberties(gs, color):
        if len(libs) == 0:
            comps.append(set(comp))
    return comps


def captured_stones(gs: GameState, color: Color) -> Set[Node]:
    dead: Set[Node] = set()
    for comp in captured_components(gs, color):
        dead |= comp
    return dead


def is_suicide_after_captures(gs: GameState, v: Node, color: Color) -> bool:
    """
    Assumes gs.state[v] == color and opponent captures already removed.
    Returns True if the group containing v has 0 liberties.
    """
    _, comp, libs = group_and_liberties_of_node(gs, v)
    if comp is None or libs is None:
        return True
    return len(libs) == 0


def resource_gate_pexec(gs, SA: float, SD: float) -> float:
    """
    Execution probability using resident resistance (delta):
        p_exec = SA / (SA + delta * SD)

    delta=1 => neutral
    delta>1 => defender advantage (harder to displace)
    """
    if not getattr(gs, "use_gate", False):
        return 1.0

    delta = float(getattr(gs, "gate_delta", 1.0))
    denom = SA + delta * SD
    if denom <= 0.0:
        return 0.0
    p = SA / denom
    # numerical safety
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return p

# def resource_gate_pexec(gs, SA: float, SD: float) -> float:
#     """
#     Return execution probability p_exec in [0,1] using the ratio gate.
#     p_exec = (SA/(SA+SD))**k
#     """
#     if not getattr(gs, "use_gate", False):
#         return 1.0  # gate disabled => always execute

#     k = float(getattr(gs, "gate_k", 1.0))
#     denom = SA + SD
#     if denom <= 0.0:
#         return 0.0
#     p = SA / denom
#     # numerical safety
#     if p <= 0.0:
#         return 0.0
#     if p >= 1.0:
#         return 1.0
#     return p ** k

def nodes_within_k_hops(gs, v, k: int):
    """Return set of nodes within k hops of v (including v). Uses gs.neigh adjacency."""
    if k <= 0:
        return {v}
    visited = {v}
    frontier = {v}
    for _ in range(k):
        nxt = set()
        for x in frontier:
            for nb in gs.neigh[x]:
                if nb not in visited:
                    visited.add(nb)
                    nxt.add(nb)
        frontier = nxt
        if not frontier:
            break
    return visited


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def group_resource_sum(gs, comp: Set) -> float:
    # gs.R can be dict-like or array-like; adjust if needed
    R = gs.R
    if hasattr(R, "get"):
        return sum(R.get(u, 1.0) for u in comp)
    else:
        return float(sum(R[u] for u in comp))


def group_resource_sum_restricted(gs, comp: set, allowed: set) -> float:
    """Sum R over nodes in comp ∩ allowed."""
    R = gs.R
    if hasattr(R, "get"):
        return sum(R.get(u, 1.0) for u in comp if u in allowed)
    else:
        return float(sum(R[u] for u in comp if u in allowed))


def play_move(gs, v, color, rng, *, return_dead: bool = False):
    if gs.state.get(v, 0) != 0:
        return (MoveStatus.ILLEGAL, 0, None) if return_dead else (MoveStatus.ILLEGAL, 0)

    old_prev_hash = gs._prev_hash

    # Prey colors: what this color can capture
    if gs.prey_of is not None:
        prey_colors = gs.prey_of.get(color, ())
    else:
        prey_colors = tuple(c for c in gs.colors if c != color)

    v_idx = gs.node_to_idx[v]
    gs.state[v] = color
    gs.state_arr[v_idx] = color
    gs.n_stones[color] += 1

    # Find neighbour groups of prey colors that have zero liberties after placement
    seen_prey: Dict[int, Set] = {}
    cap_comps = []  # list of (comp_set, prey_color)

    for nb in gs.neigh[v]:
        nb_c = gs.state.get(nb, 0)
        if nb_c not in prey_colors:
            continue
        if nb in seen_prey.get(nb_c, _EMPTY_SET):
            continue
        comp, libs = bfs_group_and_liberties(gs, nb, nb_c)
        seen_prey.setdefault(nb_c, set()).update(comp)
        if len(libs) == 0:
            cap_comps.append((comp, nb_c))

    if cap_comps and getattr(gs, "use_gate", False):
        hops = int(getattr(gs, "gate_hops", 2))
        zone = nodes_within_k_hops(gs, v, hops)
        _, att_comp, _ = group_and_liberties_of_node(gs, v)
        SA = group_resource_sum_restricted(gs, att_comp, zone)
        SD = sum(group_resource_sum_restricted(gs, comp, zone) for comp, _ in cap_comps)
        p_exec = resource_gate_pexec(gs, SA, SD)

        if rng.random() >= p_exec:
            gs.state[v] = 0
            gs.state_arr[v_idx] = 0
            gs.n_stones[color] -= 1
            gs._prev_hash = old_prev_hash
            return (MoveStatus.FAILED_GATE, 0, None) if return_dead else (MoveStatus.FAILED_GATE, 0)

    # Build dead_by_color and flat dead set
    dead_by_color: Dict[int, Set] = {}
    for comp, prey_c in cap_comps:
        dead_by_color.setdefault(prey_c, set()).update(comp)

    dead: Set = set()
    for s in dead_by_color.values():
        dead |= s

    nti = gs.node_to_idx
    for u in dead:
        gs.state[u] = 0
        gs.state_arr[nti[u]] = 0

    for prey_c, s in dead_by_color.items():
        gs.n_stones[prey_c] -= len(s)
    num_captured = len(dead)

    _, _, att_lib_count = group_size_and_lib_count(gs, v)
    if att_lib_count == 0:
        # rollback
        for prey_c, s in dead_by_color.items():
            for u in s:
                gs.state[u] = prey_c
                gs.state_arr[nti[u]] = prey_c
            gs.n_stones[prey_c] += len(s)
        gs.state[v] = 0
        gs.state_arr[v_idx] = 0
        gs.n_stones[color] -= 1
        gs._prev_hash = old_prev_hash
        return (MoveStatus.ILLEGAL, 0, None) if return_dead else (MoveStatus.ILLEGAL, 0)

    if gs.ko:
        new_hash = gs.board_hash()
        if old_prev_hash is not None and new_hash == old_prev_hash:
            # rollback
            for prey_c, s in dead_by_color.items():
                for u in s:
                    gs.state[u] = prey_c
                    gs.state_arr[nti[u]] = prey_c
                gs.n_stones[prey_c] += len(s)
            gs.state[v] = 0
            gs.state_arr[v_idx] = 0
            gs.n_stones[color] -= 1
            gs._prev_hash = old_prev_hash
            return (MoveStatus.ILLEGAL, 0, None) if return_dead else (MoveStatus.ILLEGAL, 0)
        gs._prev_hash = new_hash

    # return_dead=True: return dead_by_color dict; callers that need flat set should flatten
    return (MoveStatus.EXECUTED, num_captured, dead_by_color) if return_dead else (MoveStatus.EXECUTED, num_captured)

_EMPTY_SET: Set = frozenset()

@contextmanager
def gate_disabled(gs):
    old = getattr(gs, "use_gate", False)
    gs.use_gate = False
    try:
        yield
    finally:
        gs.use_gate = old

def is_legal(gs: GameState, v: Node, color: Color) -> bool:
    trial = gs.copy()
    rng = np.random.default_rng(0)
    with gate_disabled(trial):
        status, _ = play_move(trial, v, color, rng)
    return status == MoveStatus.EXECUTED

def legal_moves(gs: GameState, color: Color) -> List[Node]:
    moves: List[Node] = []
    rng = np.random.default_rng(0)
    for v in empty_nodes(gs):
        trial = gs.copy()
        with gate_disabled(trial):
            status, _ = play_move(trial, v, color, rng)
        if status == MoveStatus.EXECUTED:
            moves.append(v)
    return moves

def is_true_eye_like(gs: GameState, v: Node, color: Color) -> bool:
    """
    Graph-general "eye-like" test:
    - v is empty
    - all neighbors are your color
    - opponent cannot legally play at v (would be suicide without capture)
    This is a pragmatic substitute for grid-diagonal eye tests and generalizes to arbitrary graphs.
    """
    if gs.state.get(v, 0) != 0:
        return False
    nbrs = list(gs.neigh[v])
    if not nbrs:
        return False
    if not all(gs.state.get(u, 0) == color for u in nbrs):
        return False

    rng = np.random.default_rng(0)
    for opp_c in gs.colors:
        if opp_c == color:
            continue
        trial = gs.copy()
        with gate_disabled(trial):
            status, _ = play_move(trial, v, opp_c, rng)
        if status == MoveStatus.EXECUTED:
            return False  # at least one opponent can legally play here

    return True

def check_no_zero_liberty_groups(gs: GameState) -> bool:
    """
    Debug invariant: no group should have 0 liberties AND a predator adjacent.

    Under all-vs-all (prey_of=None) this is equivalent to "no 0-lib groups".
    Under cyclic/partial predation a group may survive with 0 liberties if it is
    surrounded only by non-predators (Option A semantics).
    """
    for c in gs.colors:
        if gs.prey_of is not None:
            # predators of c
            predators = {pc for pc, preys in gs.prey_of.items() if c in preys}
        else:
            predators = set(gs.colors) - {c}

        for comp, libs in groups_and_liberties(gs, c):
            if len(libs) > 0:
                continue
            # 0-lib group: invalid only if a predator is adjacent to it
            if any(gs.state.get(nb, 0) in predators for u in comp for nb in gs.neigh[u]):
                return False
    return True

# ----------------------------
# Policy interface
# ----------------------------

def niche_overlap(gs: GameState, v: Node, color: Color) -> float:
    """Fraction of neighbors occupied by any other species. Range [0, 1]."""
    deg = gs.deg[v]
    if deg == 0:
        return 0.0
    opp_n = sum(1 for u in gs.neigh[v] if gs.state.get(u, 0) not in (0, color))
    return opp_n / deg


def territory_access(trial: GameState, v: Node, color: Color, depth: int = 2) -> int:
    """
    Count reachable empty nodes within BFS depth, where BFS can traverse:
      - empty nodes (count them)
      - own stones (traverse through them)
    Does NOT traverse through opponent stones.
    """
    visited = {v}
    q = deque([(v, 0)])
    accessible = 0

    while q:
        x, d = q.popleft()
        if d == depth:
            continue
        for y in trial.G.neighbors(x):
            if y in visited:
                continue
            visited.add(y)
            s = trial.state.get(y, 0)
            if s == 0:
                accessible += 1
                q.append((y, d + 1))
            elif s == color:
                q.append((y, d + 1))
            # else opponent: stop
    return accessible


MovePolicy = Callable[[GameState, Color, np.random.Generator], Optional[Node]]

def policy_random(gs: GameState, color: Color, rng: np.random.Generator) -> Optional[Node]:
    moves = legal_moves(gs, color)
    if not moves:
        return None
    return moves[int(rng.integers(len(moves)))]

def policy_simple_heuristic(
    avoid_eye_filling: bool = True,
    w_own_adj: float = 5.0,
    w_libs: float = 1.5,
    w_size: float = 0.2,
    w_captures: float = 8.0,
    w_opp_pressure: float = 3.0,
    w_center: float = 0.2,
    self_atari_penalty: float = 50.0,
    w_degree: float = 2.0,

    # candidate pruning / pass
    sample_frac: float = 0.10,
    pass_threshold_frac: float = 0.002,

    # NEW ecological knobs
    w_overlap: float = 0.0,          # interference competition (seek/avoid)
    w_territory: float = 0.0,        # exploitation competition (resource access)
    territory_depth: int = 2,
    normalize_degree: bool = True,   # recommended for ER/SBM comparability
) -> MovePolicy:

    def _policy(gs: GameState, color: Color, rng: np.random.Generator) -> Optional[Node]:
        best_v: Optional[Node] = None
        best_score = -1e18
        # prey colors: what this color competes against for scoring purposes
        if gs.prey_of is not None:
            prey_colors = frozenset(gs.prey_of.get(color, ()))
        else:
            prey_colors = frozenset(c for c in gs.colors if c != color)

        PASS_THRESHOLD = pass_threshold_frac * len(gs.G)

        # Optional center (meaningful on grid/torus; layout-artifact on ER/SBM)
        has_coords = bool(gs.coords)
        cx = cy = 0.0
        if has_coords:
            xs = [c[0] for c in gs.coords.values()]
            ys = [c[1] for c in gs.coords.values()]
            if xs and ys:
                cx, cy = float(np.mean(xs)), float(np.mean(ys))
            else:
                has_coords = False

        # Degree normalization constants (once per move)
        if normalize_degree:
            deg_mu = gs.deg_mu
            deg_sd = gs.deg_sd
        else:
            deg_mu = 0.0
            deg_sd = 1.0

        # Candidate pruning: frontier empties + sample of other empties
        empties = empty_nodes(gs)
        stones = [u for u, s in gs.state.items() if s != 0]
        frontier_set = {v for u in stones for v in gs.neigh[u] if gs.state.get(v, 0) == 0} if stones else set()

        candidates = list(frontier_set) if frontier_set else empties

        if frontier_set:
            other_empties = [v for v in empties if v not in frontier_set]
            if other_empties:
                k = int(sample_frac * len(other_empties))
                if k > 0:
                    k = min(k, len(other_empties))
                    idx = rng.choice(len(other_empties), size=k, replace=False)
                    candidates += [other_empties[i] for i in idx]

        candidates = list(dict.fromkeys(candidates))

        # Evaluate candidates
        for v in candidates:
            if avoid_eye_filling and is_true_eye_like(gs, v, color):
                continue

            # Make gate deterministically OFF for scoring, regardless of env
            old_use_gate = gs.use_gate
            gs.use_gate = False
        
            undo = None
            terr_val = 0
            try:
                # libs_before computed on ORIGINAL position (prey neighbors only)
                libs_before_map = {}
                for u in gs.neigh[v]:
                    if gs.state.get(u, 0) in prey_colors:
                        _, _, cnt = group_size_and_lib_count(gs, u)
                        libs_before_map[u] = cnt

                status, cap_count, undo = play_move_with_undo(gs, v, color, rng)
                if status != MoveStatus.EXECUTED:
                    continue

                # Now gs == "trial position"
                _, size, libn = group_size_and_lib_count(gs, v)

                own_adj = sum(1 for u in gs.neigh[v] if gs.state.get(u, 0) == color)

                opp_pressure = 0.0
                for u in gs.neigh[v]:
                    if u not in libs_before_map:
                        continue
                    before = libs_before_map[u]
                    if gs.state.get(u, 0) not in prey_colors:
                        # captured: add full before-libs as pressure
                        opp_pressure += float(before)
                    else:
                        _, _, after = group_size_and_lib_count(gs, u)
                        opp_pressure += float(before - after)
        
                # territory should be computed on TRIAL position (current gs)
                if w_territory != 0.0:
                    terr_val = territory_access(gs, v, color, territory_depth)
        
            finally:
                if undo is not None:
                    undo_move(gs, undo)
                gs.use_gate = old_use_gate
        
            # now gs is back to ORIGINAL position; use stored terr_val etc.
            node_degree = float(gs.deg[v])
            deg_term = (node_degree - deg_mu) / deg_sd if normalize_degree else node_degree
        
            center_bonus = 0.0
            if has_coords and v in gs.coords:
                x, y = gs.coords[v]
                center_bonus = -float(np.hypot(x - cx, y - cy))
        
            overlap = niche_overlap(gs, v, color)  # original position is fine
        
            score = (
                w_degree * deg_term +
                w_own_adj * own_adj +
                w_libs * libn +
                w_size * size +
                w_captures * cap_count +
                w_opp_pressure * opp_pressure +
                w_center * center_bonus +
                w_overlap * overlap +
                w_territory * float(terr_val)
            )
    
            if libn == 1 and cap_count == 0:
                score -= self_atari_penalty

            score += 1e-6 * float(rng.standard_normal())

            if score > best_score:
                best_score = score
                best_v = v

        return best_v if best_score >= PASS_THRESHOLD else None

    return _policy


def step(
    gs: GameState,
    color: Color,
    rng: np.random.Generator,
    policy: MovePolicy
) -> Tuple[int, Optional[Node], int]:
    """
    One ply.
    Returns (status, v, captured_count).
    status ∈ {PASS, ILLEGAL, EXECUTED, FAILED_GATE}.
    """

    v = policy(gs, color, rng)

    if v is None:
        return MoveStatus.PASS, None, 0, None  # pass

    status, cap, dead_by_color = play_move(gs, v, color, rng, return_dead=True)

    if status == MoveStatus.ILLEGAL:
        # defensive fallback (should be rare)
        return MoveStatus.ILLEGAL, v, 0, None

    # flatten dead_by_color → flat set for downstream callers (resource update etc.)
    dead: Set = set()
    if dead_by_color:
        for s in dead_by_color.values():
            dead |= s

    return status, v, cap, dead


def step_mixed(
    gs: GameState,
    color: Color,
    t: int,
    rng: np.random.Generator,
    policy_main: MovePolicy,
    policy_rand: MovePolicy = policy_random,
    random_every: int = 50,
    eps: float = 0.0,
) -> Tuple[bool, Optional[Node], int]:
    """
    Periodic random or epsilon-random; otherwise main policy.
    """
    use_rand = (random_every and t % random_every == 0) or (eps > 0 and rng.random() < eps)
    pol = policy_rand if use_rand else policy_main
    return step(gs, color, rng, pol)

# ----------------------------
# Metrics (graph-general)
# ----------------------------

# def compute_network_metrics(gs: GameState) -> dict:
#     metrics = {
#         "black": {"components": [], "liberties": [], "efficiency": []},
#         "white": {"components": [], "liberties": [], "efficiency": []},
#     }
#     for color, label in [(+1, "black"), (-1, "white")]:
#         for comp, libs in groups_and_liberties(gs, color):
#             s = len(comp)
#             l = len(libs)
#             metrics[label]["components"].append(s)
#             metrics[label]["liberties"].append(l)
#             metrics[label]["efficiency"].append(l / s if s > 0 else 0.0)
#     return metrics

def fragmentation_index(gs: GameState) -> dict:
    frag = {"black": 1.0, "white": 1.0}
    for color, label in [(+1, "black"), (-1, "white")]:
        total = sum(1 for v in gs.G.nodes() if gs.state.get(v, 0) == color)
        if total == 0:
            continue
        sizes = [len(comp) for comp, _ in groups_and_liberties(gs, color)]
        if sizes:
            frag[label] = 1.0 - (max(sizes) / total)
    return frag