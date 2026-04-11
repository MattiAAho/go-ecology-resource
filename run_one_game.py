import numpy as np
import networkx as nx
from typing import Literal

from go_engine import (
    GameState,
    MoveStatus,
    ensure_R,
    empty_nodes,
    groups_and_liberties,
    fragmentation_index,
    check_no_zero_liberty_groups,
    is_legal,
    policy_simple_heuristic,
    step_mixed,
    update_R_local,
    update_R_global,
)

def make_game_from_graph(G, ko=False, coords=None, state=None, colors=(+1, -1), prey_of=None):
    if state is None:
        initial_state = {v: 0 for v in G.nodes()}
    else:
        initial_state = {v: int(state.get(v, 0)) for v in G.nodes()}
    gs = GameState(G=G, state=initial_state, coords=coords, ko=ko,
                   colors=colors, prey_of=prey_of)
    gs.n_nodes = len(G)
    gs.pass_streak = 0
    return gs


def compute_coords(G, seed=0, layout="spring", iterations=30):
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed, iterations=iterations)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError("Unknown layout")
    return {v: (float(pos[v][0]), float(pos[v][1])) for v in G.nodes()}


def compute_network_metrics(gs: GameState):
    """Compute metrics from a GameState (engine-style API)."""
    metrics = {
        'black': {'components': [], 'liberties': [], 'efficiency': []},
        'white': {'components': [], 'liberties': [], 'efficiency': []}
    }

    for color, label in [(+1, 'black'), (-1, 'white')]:
        for comp, libs in groups_and_liberties(gs, color):
            comp_size = len(comp)
            lib_count = len(libs)

            metrics[label]['components'].append(comp_size)
            metrics[label]['liberties'].append(lib_count)
            metrics[label]['efficiency'].append(lib_count / comp_size if comp_size > 0 else 0.0)

    return metrics

def size_entropy(sizes):
    sizes = np.asarray(sizes, dtype=float)
    if sizes.size == 0:
        return 0.0
    p = sizes / sizes.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))

def interblock_occupation_fractions(G, state, colors=None):
    """
    Returns:
      fracs : dict {color: fraction of inter-block edges where both endpoints == color}
      scalar: min(fracs.values())  — backward-compat summary

    Works for any number of species; pass colors=(+1,-1,+2,...) for N-player games.
    Falls back to (+1, -1) if colors is None (2-player backward compat).
    """
    inter_edges = [
        (u, v) for u, v in G.edges()
        if G.nodes[u]["block"] != G.nodes[v]["block"]
    ]
    if colors is None:
        colors = (+1, -1)
    if not inter_edges:
        return {c: 0.0 for c in colors}, 0.0

    counts = {c: 0 for c in colors}
    for u, v in inter_edges:
        su, sv = state.get(u, 0), state.get(v, 0)
        if su != 0 and su == sv and su in counts:
            counts[su] += 1

    n = len(inter_edges)
    fracs = {c: counts[c] / n for c in colors}
    return fracs, min(fracs.values())


def interblock_mix_index(G, state, colors=None):
    _, scalar = interblock_occupation_fractions(G, state, colors=colors)
    return scalar


def one_event(gs, color, t, rng, policy_main, random_every, eps):
    return step_mixed(
        gs, color, t, rng,
        policy_main=policy_main,
        random_every=random_every,
        eps=eps
    )


# def pick_color_from_occupancy(gs, rng, lambda0=0.2, lambda_occ=1.0):
#     n_b = gs.n_stones.get(+1, 0)
#     n_w = gs.n_stones.get(-1, 0)
#     lam_b = lambda0 + lambda_occ * (n_b / gs.n_nodes)
#     lam_w = lambda0 + lambda_occ * (n_w / gs.n_nodes)
#     p_black = lam_b / (lam_b + lam_w)
#     color = +1 if rng.random() < p_black else -1
#     return color, lam_b, lam_w


# def pick_color_balanced(
#     gs,
#     rng: np.random.Generator,
#     lambda0: float = 0.2,     # baseline activity so both can act even if equal
#     lambda_bal: float = 1.0,  # strength of balancing response
#     beta: float = 12.0,       # sensitivity to advantage (bigger = sharper switch)
#     eps: float = 1e-12,
# ):
#     """
#     Balanced scheduler: the UNDERDOG gets higher rate.
#     Uses sigmoid of (occ_opp - occ_self).

#     Returns (color, lam_b, lam_w, occ_b, occ_w).
#     """
#     occ_b = gs.n_stones.get(+1, 0) / (gs.n_nodes + eps)
#     occ_w = gs.n_stones.get(-1, 0) / (gs.n_nodes + eps)

#     # disadvantage signals
#     d_b = occ_w - occ_b
#     d_w = -d_b

#     sig_b = 1.0 / (1.0 + np.exp(-beta * d_b))
#     sig_w = 1.0 / (1.0 + np.exp(-beta * d_w))

#     lam_b = lambda0 + lambda_bal * sig_b
#     lam_w = lambda0 + lambda_bal * sig_w

#     p_black = lam_b / (lam_b + lam_w)
#     color = +1 if rng.random() < p_black else -1
#     return color, lam_b, lam_w, occ_b, occ_w


# def pick_color_energy(
#     rng: np.random.Generator,
#     E_b: float,
#     E_w: float,
#     *,
#     lambda_base: float = 1.0,  # overall event-rate scale
#     eta: float = 1.0,          # energy -> slowdown strength
#     lambda_floor: float = 1e-4 # prevents complete freezing
# ):
#     """
#     Pick next color from rates:
#       λ_c = lambda_base * exp(-eta * E_c) + lambda_floor

#     Returns:
#       color, lam_b, lam_w
#     """
#     lam_b = lambda_base * np.exp(-eta * E_b) + lambda_floor
#     lam_w = lambda_base * np.exp(-eta * E_w) + lambda_floor
#     p_black = lam_b / (lam_b + lam_w)
#     color = +1 if rng.random() < p_black else -1
#     return color, lam_b, lam_w

def update_energy(
    E: float,
    *,
    rho: float = 0.02,
    moved: bool = False,
    captured: int = 0,
    alpha: float = 0.15,
    gamma: float = 0.30
):
    """
    E(t+1) = (1-rho)*E(t) + alpha*1[moved] + gamma*1[captured>0]
    (You can use gamma * captured if you want captures to scale with size.)
    """
    E = (1.0 - rho) * E
    if moved:
        E += alpha
    if captured > 0:
        E += gamma
    return E

    
def run_one_game(
    G,
    coords=None,
    seed=0,
    n_moves=400,
    eps=0.15,
    random_every=0,
    ko=False,
    colors=(+1, -1),       # tuple of player color ints; default = two-player
    prey_of=None,          # None = all capture all others; dict = specific predation
    policy_kwargs=None,
    pass_threshold=None,  # unused in this version; keep for API stability
    # storage flags
    store_replay=True,
    store_metrics=True,
    store_timeseries=True,
    store_eff_ts=False,
    store_final_libs=True,       # default True since it's cheap & useful
    store_interblock=True,       # default True since it's cheap
    store_cap_ts: bool = False,          # store captures_per_turn (cap size per event)
    store_occ_ts: bool = False,          # store occupancy trajectories
    occ_every: int = 10,                 # sample occupancy every N turns
    store_R_ts: bool = False,
    store_sat_state: bool = True,   # new parameter, cheap
    R_every: int = 25,
    scheduler: Literal["sync", "gillespie_occ"] = "sync",
    use_gate: bool = False,
    gate_delta: float = 1.0,
    gate_hops: int = 2,
    resource_alpha: float = 0.002, # up to 0.01
    resource_beta: float = 0.001,  # up to 0.02
    resource_init: float = 1.0,
    ):
    
    rng = np.random.default_rng(seed)
    gs = make_game_from_graph(G, ko=ko, coords=coords, colors=colors, prey_of=prey_of)
    ensure_R(gs, init_value=resource_init)

    gs.use_gate = use_gate
    gs.gate_delta = gate_delta
    gs.gate_hops = gate_hops
    gs.resource_alpha = resource_alpha
    gs.resource_beta  = resource_beta
    gs.resource_init  = resource_init

    policy_kwargs = policy_kwargs or {}

    # Build per-color policies.
    # Backward-compat keys: "black"/+1 for color +1, "white"/-1 for color -1.
    # Generic key: color int (e.g. 1, 2, 3).
    # If none of the above match, fall back to a shared policy (empty kwargs).
    _compat = {"black": +1, "white": -1}
    _shared_pk = policy_kwargs if not any(isinstance(k, (int, str)) and (k in gs.colors or k in _compat) for k in policy_kwargs) else {}
    def _pk_for(c):
        if c in policy_kwargs:
            return policy_kwargs[c]
        label = {+1: "black", -1: "white"}.get(c)
        if label and label in policy_kwargs:
            return policy_kwargs[label]
        return _shared_pk
    policies = {c: policy_simple_heuristic(**_pk_for(c)) for c in gs.colors}
    # backward-compat aliases used below
    policy_black = policies.get(+1, policies.get(gs.colors[0]))
    policy_white = policies.get(-1, policies.get(gs.colors[1] if len(gs.colors) > 1 else gs.colors[0]))

    # policy_kwargs = policy_kwargs or {}
    # policy_main = policy_simple_heuristic(**policy_kwargs)

    # Optional containers
    history = [] if store_replay else None
    metrics_history = [] if store_metrics else None
    atari_counts = [] if store_timeseries else None
    pass_ts = [] if store_timeseries else None
    pass_with_legal_ts = [] if store_timeseries else None
    pass_no_legal_ts = [] if store_timeseries else None
    passes_failed_gate_ts = [] if store_timeseries else None
    attempt_ts = [] if store_timeseries else None
    passes_failed_gate_ts_by_color = {c: ([] if store_timeseries else None) for c in gs.colors}
    illegal_ts = [] if store_timeseries else None
    color_ts = [] if store_timeseries else None         # <-- strongly recommended for gillespie
    capture_sizes = [] if store_timeseries else None
    eff_b_ts = [] if store_eff_ts else None
    eff_w_ts = [] if store_eff_ts else None

    captures_per_turn = [] if (store_timeseries and store_cap_ts) else None

    occ_b_ts = [] if store_occ_ts else None
    occ_w_ts = [] if store_occ_ts else None
    occ_t_ts = [] if store_occ_ts else None  # timestamps
    # N-player generalization: one list per color
    occ_ts_by_color = {c: [] for c in gs.colors} if store_occ_ts else None

    R_mean_ts = [] if (store_timeseries and store_R_ts) else None
    R_occ_mean_ts = [] if (store_timeseries and store_R_ts) else None
    R_empty_mean_ts = [] if (store_timeseries and store_R_ts) else None
    R_occ_mass_ts = [] if (store_timeseries and store_R_ts) else None  # mean(R*occ) or sum(R*occ)

    # per-color counters (all keyed by color int)
    cap_total = {c: 0 for c in gs.colors}   # total stones captured by this color
    cap_moves = {c: 0 for c in gs.colors}   # moves actually played (denominator for capture rate)
    captures_n_count = 0                     # number of capture events (always tracked, store_timeseries-independent)

    E       = {c: 0.0 for c in gs.colors}   # energy per color
    lam_sum = {c: 0.0 for c in gs.colors}   # accumulated lambda per color
    lam_count = 0                            # number of scheduler samples

    # Always-correct counters (so we don't depend on history)
    turns = 0
    moves_count = 0
    passes_count = 0
    passes_in_row = 0

    events   = {c: 0 for c in gs.colors}
    moves_c  = {c: 0 for c in gs.colors}
    passes_c = {c: 0 for c in gs.colors}
    passes_failed_gate_c = {c: 0 for c in gs.colors}
    pass_with_legal = pass_no_legal = 0
    illegal_count = 0
    false_negative = 0
    exact_checks = 0
    t_pass2 = None

    gate_checks = 0
    gate_failed = 0

    # saturation detection
    sat_counter = 0
    saturated = False
    sat_t = None
    SAT_K = 50
    sat_state = None
    sat_R = None

    # extinction tracking: first turn each color drops to 0 stones
    extinction_turn: dict[int, int] = {}
    _had_stones: set[int] = set()          # colors that have placed at least one stone

    streak_color = None
    streak_len = 0
    streak_max = 0
    streak_sum = 0
    streak_count = 0

    # --- NEW: play-streaks (only when a legal move is actually played) ---
    play_streak_color = None
    play_streak_len = 0
    play_streak_max = 0
    play_streak_sum = 0
    play_streak_count = 0

    R_updates_local = 0
      
    def mean_eff(color_):
        groups = groups_and_liberties(gs, color_)
        effs = []
        for comp, libs in groups:
            s = len(comp)
            if s > 0:
                effs.append(len(libs) / s)
        return float(np.mean(effs)) if effs else 0.0

    _color_cycle = list(gs.colors)   # round-robin order for sync scheduler
    _color_idx = 0
    color = _color_cycle[0]
    
    for t in range(n_moves):
          
        if scheduler == "gillespie_occ":
            # Energy-based Gillespie for any number of colors.
            # λ_c = exp(-η·E_c) + floor  → underdog (low E) gets more turns.
            # Sensitivity parameter η controls how strongly energy imbalance translates into turn-frequency imbalance, it is implicitly set to 1.
            lams = {c: np.exp(-E[c]) + 1e-4 for c in gs.colors}
            total_lam = sum(lams.values())
            p = rng.random()
            cumulative = 0.0
            color = gs.colors[-1]          # fallback (never reached in practice)
            for c in gs.colors:
                cumulative += lams[c] / total_lam
                if p < cumulative:
                    color = c
                    break
            for c in gs.colors:
                lam_sum[c] += lams[c]
            lam_count += 1

        policy_main_c = policies.get(color, policy_black)

        status, v, cap, dead = one_event( 
            gs, color, t, rng,
            policy_main=policy_main_c,
            random_every=random_every,
            eps=eps
        )

        attempted   = (v is not None)
        true_pass   = (status == MoveStatus.PASS)
        illegal     = (status == MoveStatus.ILLEGAL)
        failed_gate = (status == MoveStatus.FAILED_GATE)
        executed    = (status == MoveStatus.EXECUTED)

        pass_like   = true_pass or failed_gate or illegal   # if you want illegal to consume a turn

        cap_eff     = cap if executed else 0

        # for flow accounting, treat illegal as pass-like (recommended)
        pass_like_flow = pass_like or bool(illegal)

        moved  = attempted and (executed or failed_gate)   # effort spent
        played = attempted and executed                    # stone placed + committed
        
        # --- captures_per_turn (per event) ---
        if store_timeseries and store_cap_ts:
            captures_per_turn.append(int(cap_eff) if cap_eff > 0 else 0)
            

        E[color] = update_energy(
            E[color],
            rho=0.02,
            moved=moved,
            captured=cap_eff,
            alpha=0.15,
            gamma=0.30,
        )

        # --- resource dynamics ---
         
        # local update: only if you actually attempted something (including failed gate)
        if attempted:
            update_R_local(
                gs,
                v_played=v,
                dead=(dead if executed else None),  # only real captures if executed
                alpha=gs.resource_alpha,
                beta=gs.resource_beta,
                neigh_wear=0.25,
            )
            R_updates_local += 1
        
        # optional global amortized update (every R_every turns)
        if store_timeseries and store_R_ts and (t % R_every == 0):
            update_R_global(gs, 
                            alpha=gs.resource_alpha, 
                            beta=gs.resource_beta
                           )
                
        if store_timeseries:
            color_ts.append(color)      

        # event counts
        events[color] += 1

        # Count capture attempts:
        if failed_gate:
            gate_checks += 1
            gate_failed += 1
        elif executed and cap_eff > 0:
            gate_checks += 1

        # --- compute pass classification (only if pass) ---
        any_legal = False
        if true_pass:
            stones = [u for u, s in gs.state.items() if s != 0]
            empties = empty_nodes(gs)
            frontier = [x for u in stones for x in gs.neigh[u] if gs.state.get(x, 0) == 0]
            frontier = list(dict.fromkeys(frontier))
            pool = frontier if frontier else empties
    
            if len(pool) == 0:
                any_legal = False
            else:
                K = min(50, len(pool))
                idx = rng.choice(len(pool), size=K, replace=False)
                any_legal = False
                for i in idx:
                    if is_legal(gs, pool[i], color):     # <-- FIXED
                        any_legal = True
                        break

            if not any_legal and frontier:
                # fallback sample from global empties
                pool2 = empties
                K2 = min(50, len(pool2))
                idx2 = rng.choice(len(pool2), size=K2, replace=False)
                for i in idx2:
                    if is_legal(gs, pool2[i], color):
                        any_legal = True
                        break

            # --- saturation detection ---
            if true_pass and (not any_legal):
                sat_counter += 1
            else:
                sat_counter = 0
            
            if (not saturated) and (sat_counter >= SAT_K):
                saturated = True
                sat_t = t - SAT_K + 1
                sat_state = dict(gs.state)
                sat_R = None if gs.R is None else {u: float(gs.R[u]) for u in gs.G.nodes()}            

            # inside (v is None) block, after computing any_legal from sampling:
            if rng.random() < 0.01:   # 1% of passes only
                exact_any = any(is_legal(gs, u, color) for u in empties)
                false_negative += int((not any_legal) and exact_any)
                exact_checks += 1
    
            pass_with_legal += int(any_legal)
            pass_no_legal   += int(not any_legal)
        
            passes_c[color] += 1

        elif failed_gate:
            passes_failed_gate_c[color] += 1

        elif executed: # move happened (legal move)
            moves_c[color] += 1
            cap_moves[color] += 1
            cap_total[color] += int(cap_eff)
            if cap_eff > 0:
                captures_n_count += 1
        
        else:
            illegal_count += int(illegal)
            
        # --- append PASS time series ON EVERY TURN (aligned with atari_counts) ---
        if store_timeseries:
            is_true_pass  = int(true_pass)
            is_failed_gate = int(failed_gate)
            is_pass_like  = int(pass_like)

            pass_ts.append(is_pass_like)

            # only meaningful for TRUE passes
            pass_with_legal_ts.append(int(is_true_pass and any_legal))
            pass_no_legal_ts.append(int(is_true_pass and (not any_legal)))
            
            if passes_failed_gate_ts is not None:
                passes_failed_gate_ts.append(is_failed_gate)
            if illegal_ts is not None:
                illegal_ts.append(illegal)

                # NEW: attempt indicator (policy returned a vertex)
            attempt_ts.append(int(attempted))
        
            # failed-gate split by color (N-player)
            for _c, _ts in passes_failed_gate_ts_by_color.items():
                _ts.append(int(failed_gate and (color == _c)))

        # streaks (run-lengths of same-color turns)
        if streak_color is None:
            streak_color = color
            streak_len = 1
        else:
            if color == streak_color:
                streak_len += 1
            else:
                # finalize previous streak
                streak_max = max(streak_max, streak_len)
                streak_sum += streak_len
                streak_count += 1
        
                # start new streak
                streak_color = color
                streak_len = 1

                # --- NEW: play-streaks (run-lengths of same-color PLAYED moves) ---
        if played:
            if play_streak_color is None:
                play_streak_color = color
                play_streak_len = 1
            else:
                if color == play_streak_color:
                    play_streak_len += 1
                else:
                    play_streak_max = max(play_streak_max, play_streak_len)
                    play_streak_sum += play_streak_len
                    play_streak_count += 1

                    play_streak_color = color
                    play_streak_len = 1       
   
        turns += 1
        if pass_like_flow:
            passes_count += 1
        else:
            moves_count += 1
    
        if store_replay:
            if true_pass:
                label = "pass"
            elif failed_gate:
                label = "failed_gate"
            elif executed:
                label = "move"
            else:
                label = "illegal"
            history.append((t, color, v, label))
    
        if store_timeseries:
            if executed and cap_eff > 0:
                capture_sizes.append(cap_eff)
    
            ataris = sum(
                1 for c in gs.colors
                for _, libs in groups_and_liberties(gs, c)
                if len(libs) == 1
            )
            atari_counts.append(ataris)
    
        # Invariant: no group should have 0 liberties AND a predator adjacent.
        # Under partial predation (RPS etc.) groups can survive with 0 libs if
        # surrounded only by non-predators (Option A), so skip the check then.
        if gs.prey_of is None:
            assert check_no_zero_liberty_groups(gs), f"Illegal zero-liberty group at turn {t}"
    
        if store_metrics:
            metrics_history.append(compute_network_metrics(gs))
    
        if store_eff_ts:
            eff_b_ts.append(mean_eff(+1))
            eff_w_ts.append(mean_eff(-1))

        # --- occupancy trajectory sampling ---
        if store_occ_ts and (t % occ_every == 0):
            n = len(gs.G)  # or gs.n_nodes if you have it
            occ_b_ts.append(gs.n_stones.get(+1, 0) / n)
            occ_w_ts.append(gs.n_stones.get(-1, 0) / n)
            occ_t_ts.append(t)
            for _c in gs.colors:
                occ_ts_by_color[_c].append(gs.n_stones.get(_c, 0) / n)

        # --- extinction tracking ---
        for _ec in gs.colors:
            _ns = gs.n_stones.get(_ec, 0)
            if _ns > 0:
                _had_stones.add(_ec)
            elif _ec in _had_stones and _ec not in extinction_turn:
                extinction_turn[_ec] = t

        # --- resource field trajectory sampling ---
        if store_timeseries and store_R_ts and (t % R_every == 0):
            R = gs.R
            if R is not None:
                vals = np.fromiter((R[u] for u in gs.G.nodes()), dtype=float)
                occ = np.fromiter((1.0 if gs.state.get(u,0)!=0 else 0.0 for u in gs.G.nodes()), dtype=float)
                R_mean_ts.append(float(np.mean(vals)))
                # occupied vs empty means (very diagnostic)
                occ_mask = (occ > 0.0)
                if occ_mask.any():
                    R_occ_mean_ts.append(float(vals[occ_mask].mean()))
                else:
                    R_occ_mean_ts.append(np.nan)
        
                if (~occ_mask).any():
                    R_empty_mean_ts.append(float(vals[~occ_mask].mean()))
                else:
                    R_empty_mean_ts.append(np.nan)

                # resource “mass” associated with occupied nodes
                R_occ_mass_ts.append(float((vals * occ).mean()))
    
        # # termination: keep classic only for synchronous (for now)
        # if scheduler == "sync":
        #     if v is None:
        #         passes_in_row += 1
        #     else:
        #         passes_in_row = 0
        #     if passes_in_row >= 2:
        #         break
                
        # no termination: record two passes but continue
        if scheduler == "sync":
            if pass_like_flow:
                passes_in_row += 1
            else:
                passes_in_row = 0

            if passes_in_row >= 2 and t_pass2 is None:
                t_pass2 = t  # record but DON'T break
            _color_idx = (_color_idx + 1) % len(_color_cycle)
            color = _color_cycle[_color_idx]

    # final tallies
    final_n_stones = dict(gs.n_stones)  # {color: count}
    black = final_n_stones.get(+1, 0)
    white = final_n_stones.get(-1, 0)
    N_players = len(gs.colors)
    n_total = sum(final_n_stones.values())
    n_nodes = len(gs.G)
    _occ = [final_n_stones.get(c, 0) / max(1, n_nodes) for c in gs.colors]

    # survivors_count: colors with at least one stone
    survivors_count = sum(1 for c in gs.colors if final_n_stones.get(c, 0) > 0)

    # largest_share: maximum occupancy fraction
    largest_share = float(max(_occ)) if _occ else 0.0

    # Jain's fairness index on final stone counts (all-zero → nan, not 1.0)
    _jain_sum1 = sum(_occ)
    _jain_sum2 = sum(x * x for x in _occ)
    if _jain_sum2 == 0.0:
        jain_fairness_final = float("nan")
    else:
        jain_fairness_final = float((_jain_sum1 ** 2) / (N_players * _jain_sum2))

    # coexistence_index: stones-normalised for 2-player; Jain for N>2.
    # 2-player formula: 1 - |black - white| / (black + white)
    #   = 0 when one player holds all stones (monopoly)
    #   = 1 when both hold equal stones (coexistence)
    # NOTE: two_p_*_v1.pkl files store the old board-normalised formula
    #   (1 - |Δ|/n_nodes).  Use analysis_helpers.ci_2p(r) to get the
    #   corrected value from those files without re-running the sweep.
    if N_players == 2:
        _n_total_2p = black + white
        _ci = float(1.0 - abs(black - white) / _n_total_2p) if _n_total_2p > 0 else float("nan")
    else:
        _ci = jain_fairness_final

    # final group stats
    b_groups = groups_and_liberties(gs, +1)
    w_groups = groups_and_liberties(gs, -1)
    b_sizes = [len(comp) for comp, _ in b_groups]
    w_sizes = [len(comp) for comp, _ in w_groups]

    frag = fragmentation_index(gs)

    # finalize last streak
    if streak_color is not None:
        streak_max = max(streak_max, streak_len)
        streak_sum += streak_len
        streak_count += 1
    
    streak_mean = streak_sum / max(1, streak_count)

    # finalize last play-streak (if any played moves occurred)
    if play_streak_color is not None:
        play_streak_max = max(play_streak_max, play_streak_len)
        play_streak_sum += play_streak_len
        play_streak_count += 1

    play_streak_mean = play_streak_sum / max(1, play_streak_count)

    # compute backward-compat scalar aliases from dicts (used in result dict below)
    events_b = events.get(+1, 0)
    events_w = events.get(-1, 0)
    moves_b  = moves_c.get(+1, 0)
    moves_w  = moves_c.get(-1, 0)
    passes_b = passes_c.get(+1, 0)
    passes_w = passes_c.get(-1, 0)
    passes_failed_gate_b = passes_failed_gate_c.get(+1, 0)
    passes_failed_gate_w = passes_failed_gate_c.get(-1, 0)
    cap_total_b = cap_total.get(+1, 0)
    cap_total_w = cap_total.get(-1, 0)
    cap_moves_b = cap_moves.get(+1, 0)
    cap_moves_w = cap_moves.get(-1, 0)

    assert streak_sum == turns, (streak_sum, turns)
    assert 1 <= streak_max <= turns
    assert sum(events.values()) == turns
    assert moves_count + passes_count == turns
    # total played moves equals play_streak_sum by construction
    assert play_streak_sum == moves_count, (play_streak_sum, moves_count)
    assert 0 <= play_streak_max <= moves_count

    result = {
        # --- run summary ---
        "turns": turns,
        "moves": moves_count,
        "passes": passes_count,

        # --- population ---
        "n_stones": final_n_stones,               # generic N-player: {color: count}
        "black_stones": black,                    # compat alias for color +1
        "white_stones": white,                    # compat alias for color -1
        "black_occupancy": float(black / n_nodes),
        "white_occupancy": float(white / n_nodes),
        "survivors_count": survivors_count,
        "jain_fairness_final": jain_fairness_final,
        "largest_share": largest_share,
        "coexistence_index": _ci,                 # 2-player: 1-|Δ|/N; N>2: Jain's index

        # --- captures / avalanches ---
        # These four scalars use always-tracked counters so they are correct
        # regardless of store_timeseries.  capture_sizes (the per-event list) is
        # only available when store_timeseries=True and is stored separately below.
        "captures_n": captures_n_count,
        "captures_total": sum(cap_total.values()),
        "captures_mean": float(sum(cap_total.values()) / max(1, captures_n_count)),
        "capture_rate_per_move": float(sum(cap_total.values()) / max(1, moves_count)),

        # --- criticality ---
        "ataris_mean": float(np.mean(atari_counts)) if atari_counts else 0.0,
        "ataris_max": int(np.max(atari_counts)) if atari_counts else 0,

        # --- structure ---
        "black_groups": len(b_groups),
        "white_groups": len(w_groups),
        "black_largest_group": max(b_sizes) if b_sizes else 0,
        "white_largest_group": max(w_sizes) if w_sizes else 0,
        "black_largest_frac": float(max(b_sizes) / black) if black > 0 and b_sizes else 0.0,
        "white_largest_frac": float(max(w_sizes) / white) if white > 0 and w_sizes else 0.0,
        "black_fragmentation": float(frag["black"]),
        "white_fragmentation": float(frag["white"]),

        # --- optional diversity ---
        "black_group_entropy": size_entropy(b_sizes),
        "white_group_entropy": size_entropy(w_sizes),
    }

    result["sat_fraction"] = (sat_t / turns) if sat_t is not None else None

    # cheap, useful per-run extras
    if store_final_libs:
        b_libs = [len(libs) for _, libs in b_groups]
        w_libs = [len(libs) for _, libs in w_groups]
        result["final_group_liberties_black"] = b_libs
        result["final_group_liberties_white"] = w_libs

    # time_to_first_extinction: earliest turn any species went extinct (None = all survived)
    result["time_to_first_extinction"] = min(extinction_turn.values()) if extinction_turn else None

    # occ_avg_last30p: per-color mean occupancy over the final 30 % of the occ trajectory
    if occ_ts_by_color is not None:
        _ts_len = len(next(iter(occ_ts_by_color.values()), []))
        _start = int(_ts_len * 0.70)
        result["occ_avg_last30p"] = {
            c: float(np.mean(occ_ts_by_color[c][_start:])) if occ_ts_by_color[c][_start:] else float("nan")
            for c in gs.colors
        }
    else:
        result["occ_avg_last30p"] = None

    if store_interblock:
        if hasattr(G, "graph") and "partition" in G.graph:
            _ib_fracs, _ib_min = interblock_occupation_fractions(G, gs.state, colors=gs.colors)
            result["interblock_mix"] = _ib_min                 # scalar, backward-compat
            result["interblock_rate_by_color"] = _ib_fracs     # per-species dict
        else:
            result["interblock_mix"] = None
            result["interblock_rate_by_color"] = None

    # final_state is always stored: it is a cheap shallow copy and is needed for
    # downstream regime/eye analysis regardless of replay storage.
    result["final_state"] = dict(gs.state)

    # optional big payloads
    if store_replay:
        result["history"] = history
        result["coords"] = coords
        result["final_R"] = (None if gs.R is None else {u: float(gs.R[u]) for u in gs.G.nodes()})
        # result["sat_state"] = sat_state
        # result["sat_R"] = sat_R

    if store_metrics:
        result["metrics_history"] = metrics_history

    if store_timeseries:
        result["atari_counts"] = atari_counts
        result["capture_sizes"] = capture_sizes

    if store_timeseries and store_cap_ts:
        result["captures_per_turn"] = captures_per_turn

    if store_occ_ts:
        result["occ_b_ts"] = occ_b_ts
        result["occ_w_ts"] = occ_w_ts
        result["occ_t_ts"] = occ_t_ts
        result["occ_ts_by_color"] = occ_ts_by_color  # N-player: {color: [fractions]}

    if store_eff_ts:
        result["eff_b_ts"] = eff_b_ts
        result["eff_w_ts"] = eff_w_ts

    if store_timeseries and store_R_ts:
        result["R_mean_ts"] = R_mean_ts
        result["R_occ_mean_ts"] = R_occ_mean_ts
        result["R_empty_mean_ts"] = R_empty_mean_ts
        result["R_occ_mass_ts"] = R_occ_mass_ts
        result["R_every"] = R_every

    if store_timeseries:
        result["pass_ts"] = pass_ts
        result["pass_with_legal_ts"] = pass_with_legal_ts
        result["pass_no_legal_ts"] = pass_no_legal_ts
        result["passes_failed_gate_ts"] = passes_failed_gate_ts
        result["attempt_ts"] = attempt_ts
        result["passes_failed_gate_ts_by_color"] = dict(passes_failed_gate_ts_by_color)
        # backward-compat aliases
        result["passes_failed_gate_ts_black"] = passes_failed_gate_ts_by_color.get(+1)
        result["passes_failed_gate_ts_white"] = passes_failed_gate_ts_by_color.get(-1)
        result["R_updates_local"] = R_updates_local
        result["color_ts"] = color_ts
        result["illegal_ts"] = illegal_ts

    if store_sat_state:
        result["sat_state"] = sat_state
        result["sat_R"] = sat_R
    
    result.update({
        # generic N-color event / move / pass dicts (always present)
        "events_by_color":              dict(events),
        "moves_by_color":               dict(moves_c),
        "passes_true_by_color":         dict(passes_c),
        "passes_failed_gate_by_color":  dict(passes_failed_gate_c),
        "cap_total_by_color":           dict(cap_total),
        "cap_moves_by_color":           dict(cap_moves),
        "event_share_by_color":         {c: events[c] / max(1, turns) for c in gs.colors},
        "move_rate_by_color":           {c: moves_c[c] / max(1, events[c]) for c in gs.colors},
        "capture_rate_by_color":        {c: cap_total[c] / max(1, cap_moves[c]) for c in gs.colors},

        # backward-compat scalar aliases (always present; 0 when color absent)
        "events_black": events_b,
        "events_white": events_w,
        "event_share_black": events_b / max(1, turns),
        "event_share_white": events_w / max(1, turns),
        "moves_black": moves_b,
        "moves_white": moves_w,
        "passes_true_black": passes_b,
        "passes_true_white": passes_w,
        "passes_like_black": passes_b + passes_failed_gate_b,
        "passes_like_white": passes_w + passes_failed_gate_w,
        "passes_failed_gate_black": passes_failed_gate_b,
        "passes_failed_gate_white": passes_failed_gate_w,
        "gate_checks": gate_checks,
        "gate_failed": gate_failed,
        "gate_bite_rate": (gate_failed / gate_checks) if gate_checks > 0 else 0.0,
        "attempt_rate": gate_checks / max(1, turns),

        "pass_with_legal": pass_with_legal,
        "pass_no_legal": pass_no_legal,
        "illegal_count": illegal_count,
        "false_negative_rate": false_negative / max(1, exact_checks),

        "move_rate_black": moves_b / max(1, events_b),
        "move_rate_white": moves_w / max(1, events_w),

        # streak diagnostics
        "streak_count": streak_count,
        "streak_max": int(streak_max),
        "streak_mean": float(streak_mean),
        "play_streak_count": int(play_streak_count),
        "play_streak_max": int(play_streak_max),
        "play_streak_mean": float(play_streak_mean),
    })

    # Generic N-color lambda and energy diagnostics.
    # Backward-compat keys preserved for 2-player case.
    _lam_means = {c: (lam_sum[c] / lam_count) if lam_count > 0 else np.nan
                  for c in gs.colors}
    result.update({
        # generic keys (always present)
        "lambda_means":   _lam_means,
        "E_final":        {c: float(E[c]) for c in gs.colors},
        "lambda_samples": lam_count if lam_count > 0 else np.nan,
    })
    # backward-compat aliases for 2-player notebooks
    if +1 in gs.colors and -1 in gs.colors:
        result.update({
            "lambda_black_mean": _lam_means.get(+1, np.nan),
            "lambda_white_mean": _lam_means.get(-1, np.nan),
            "E_black_final":     float(E.get(+1, 0.0)),
            "E_white_final":     float(E.get(-1, 0.0)),
        })

    result.update({
        # backward-compat capture aliases
        "captures_total_black": int(cap_total_b),
        "captures_total_white": int(cap_total_w),
        "capture_rate_per_move_black": float(cap_total_b / max(1, cap_moves_b)),
        "capture_rate_per_move_white": float(cap_total_w / max(1, cap_moves_w)),
        "moves_played_black": int(cap_moves_b),
        "moves_played_white": int(cap_moves_w),
    })

    result.update({
        "t_pass2": t_pass2,
        "saturated": saturated,
        "sat_t": sat_t,
        "sat_k": SAT_K,
    })


    return result


def build_graphs(grid_size=19, er_p=0.03, sbm_seed=1):
    # 2D lattice (non-periodic)
    G_lattice = nx.grid_2d_graph(grid_size, grid_size)

    # torus grid
    G_torus = nx.grid_2d_graph(grid_size, grid_size, periodic=True)

    # ER
    n = grid_size**2
    G_er = nx.erdos_renyi_graph(n, er_p, seed=1)

    # SBM
    n = grid_size**2
    unit = n / 6
    size_1 = int(round(3 * unit))
    size_2 = int(round(2 * unit))
    size_3 = n - size_1 - size_2
    sizes = [size_1, size_2, size_3]
    assert all(s > 0 for s in sizes), sizes
    P = [
        [0.08, 0.01, 0.005],
        [0.01, 0.06, 0.01],
        [0.005, 0.01, 0.07],
    ]
    G_sbm = nx.stochastic_block_model(sizes, P, seed=sbm_seed)

    return {
        "lattice": G_lattice,
        "torus": G_torus,
        "er": G_er,
        "sbm": G_sbm,
    }


# def run_benchmark(
#     n_runs=10,
#     grid_size=19,
#     n_moves=400,
#     eps=0.15,
#     random_every=0,
#     ko=False,
#     er_p=0.03,
#     layout_seed=1,
#     policy_kwargs=None,
#     store_finals=True,
# ):
#     graphs = build_graphs(grid_size=grid_size, er_p=er_p)

#     # coords: grids use natural coords; ER/SBM use fixed layout for stable drawings
#     coords = {
#         "lattice": {v: (float(v[0]), float(v[1])) for v in graphs["lattice"].nodes()},
#         "torus": {v: (float(v[0]), float(v[1])) for v in graphs["torus"].nodes()},
#         "er": compute_coords(graphs["er"], seed=layout_seed, layout="spring"),
#         "sbm": compute_coords(graphs["sbm"], seed=layout_seed, layout="spring"),
#     }

#     results = {k: [] for k in graphs.keys()}

#     for gname, G in graphs.items():
#         for i in range(n_runs):
#             graph_seed_base = {"lattice": 1000, "torus": 2000, "er": 3000, "sbm": 4000}
#             seed = graph_seed_base[gname] + i
#             # seed = 10_000 * (hash(gname) & 0xFFFF) + i  # deterministic per graph+run
#             r = run_one_game(
#                 G,
#                 coords=coords[gname],
#                 seed=seed,
#                 n_moves=n_moves,
#                 eps=eps,
#                 random_every=random_every,
#                 ko=ko,
#                 policy_kwargs=policy_kwargs,
#             )

#             if not store_finals:
#                 r.pop("final_state", None)
#                 r.pop("history", None)
#                 r.pop("coords", None)

#             r["graph"] = gname
#             r["run"] = i
#             results[gname].append(r)

#     return results
