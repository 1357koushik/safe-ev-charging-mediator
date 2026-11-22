"""Baseline charging policies.

Owner: Sura Sravan Kumar.

Each function returns a recommendation array of shape (N,) with values 0 or 1.
Disconnected EVs always receive 0.

Signature (all baselines):
    recommend_<name>(q, p_rated_kw, c_max_kw, state, rng) -> np.ndarray

Parameters
----------
q : np.ndarray, shape (N, 2)
    Q-values, column 0 = Q_off, column 1 = Q_on. Disconnected EVs have [0, 0].
p_rated_kw : np.ndarray, shape (N,)
    Rated charger power in kW.
c_max_kw : float
    Transformer capacity limit in kW.
state : dict with keys:
    'connected' : bool array (N,) – True if EV is present this slot.
    'k_left'    : int array (N,)  – slots remaining until departure.
    'k_req'     : int array (N,)  – minimum ON slots still needed.
    'soc'       : float array (N,) – current state of charge.
rng : np.random.Generator
    Passed for interface consistency; not used by deterministic baselines.

Returns
-------
np.ndarray shape (N,) of int64, values 0 or 1.
"""

from __future__ import annotations

import numpy as np

from evsim.common import EPS


# ------------------------------------------------------------------ helpers --

def _zero_rec(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int64)


def _greedy_fill(order: np.ndarray, p_rated_kw: np.ndarray,
                 c_max_kw: float) -> np.ndarray:
    """Return 0/1 recommendations by filling capacity greedily in `order`."""
    n = p_rated_kw.size
    rec = _zero_rec(n)
    remaining = c_max_kw
    for i in order:
        if p_rated_kw[i] <= remaining + EPS:
            rec[i] = 1
            remaining -= p_rated_kw[i]
    return rec


# --------------------------------------------------------------- baselines --

def recommend_immediate(q, p_rated_kw, c_max_kw, state, rng) -> np.ndarray:
    """Charge on arrival: fill transformer in EV index order (arrival order)."""
    p = np.asarray(p_rated_kw, dtype=float)
    n = p.size
    connected = np.asarray(state["connected"], dtype=bool)
    order = np.flatnonzero(connected)          # index order = arrival order
    return _greedy_fill(order, p, c_max_kw)


def recommend_price_only(q, p_rated_kw, c_max_kw, state, rng) -> np.ndarray:
    """Charge in cheapest available slots: highest Q_on first (lowest price)."""
    q_arr = np.asarray(q, dtype=float)
    p = np.asarray(p_rated_kw, dtype=float)
    connected = np.asarray(state["connected"], dtype=bool)
    k_req_arr = np.asarray(state["k_req"], dtype=int)

    # Only consider connected EVs that still need charging.
    eligible = connected & (k_req_arr > 0)
    eligible_idx = np.flatnonzero(eligible)
    if eligible_idx.size == 0:
        return _zero_rec(p.size)

    # Sort by Q_on descending (highest value = cheapest effective cost).
    order = eligible_idx[np.argsort(-q_arr[eligible_idx, 1])]
    return _greedy_fill(order, p, c_max_kw)


def recommend_urgency(q, p_rated_kw, c_max_kw, state, rng) -> np.ndarray:
    """Urgency-ranked: charge most urgent EVs first (k_req / k_left)."""
    p = np.asarray(p_rated_kw, dtype=float)
    connected = np.asarray(state["connected"], dtype=bool)
    k_left = np.asarray(state["k_left"], dtype=float)
    k_req_arr = np.asarray(state["k_req"], dtype=float)

    # EVs that need no more charging get rec = 0.
    eligible = connected & (k_req_arr > 0)
    eligible_idx = np.flatnonzero(eligible)
    if eligible_idx.size == 0:
        return _zero_rec(p.size)

    urgency = k_req_arr[eligible_idx] / np.maximum(k_left[eligible_idx], 1.0)
    order = eligible_idx[np.argsort(-urgency)]
    return _greedy_fill(order, p, c_max_kw)


def recommend_independent_mdp(q, p_rated_kw, c_max_kw, state, rng) -> np.ndarray:
    """Each EV follows its own Q-values; ignores the transformer limit."""
    q_arr = np.asarray(q, dtype=float)
    p = np.asarray(p_rated_kw, dtype=float)
    connected = np.asarray(state["connected"], dtype=bool)
    n = p.size
    rec = _zero_rec(n)
    for i in np.flatnonzero(connected):
        if q_arr[i, 1] > q_arr[i, 0]:    # Q_on > Q_off
            rec[i] = 1
    return rec


# ---------------------------------------------------------- dispatch table --

BASELINE_POLICIES: dict[str, object] = {
    "immediate":       recommend_immediate,
    "price_only":      recommend_price_only,
    "urgency":         recommend_urgency,
    "independent_mdp": recommend_independent_mdp,
}

__all__ = [
    "recommend_immediate",
    "recommend_price_only",
    "recommend_urgency",
    "recommend_independent_mdp",
    "BASELINE_POLICIES",
]
