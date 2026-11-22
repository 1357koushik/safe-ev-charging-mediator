"""Overload penalty and how it is split among defectors.

Owner: Moram Raghavendra Sri Koushik.

This file is the only place where the penalty formula lives. The mediator and
the simulator both call into it, so the number an EV is charged in the
simulator is the same number the mediator assumed when it wrote its
obedience constraints.

Conventions (docs/Work_Division_Plan.md, section 3.4):

* ``rec`` and ``act`` are integer arrays of 0/1 with shape (N,). Power drawn
  by EV i is ``act[i] * p_rated_kw[i]``.
* An EV that is not connected in the slot has ``rec[i] == act[i] == 0``. The
  simulator can check this with ``check_connected``.
* A defector is an EV with ``rec[i] == 0`` and ``act[i] == 1``. An EV that was
  told to charge and waited instead is not a defector and pays nothing.
* ``lam`` is in Rs per kW^2 and ``lam_lin`` in Rs per kW, so every penalty
  is in Rs.

Assumptions: the penalty is settled slot by slot with no memory; only
defectors pay; the Shapley bound c({i}) = rho_i assumes the EVs told to
charge actually charge (if some wait, a defector may pay less, because it is
using capacity they left free).

The penalty
-----------
For an overload x = max(0, L - c_max_kw) in kW the penalty is::

    P(L) = lam * x**2 + lam_lin * x

``lam_lin`` is 0 unless a caller passes it, which gives the quadratic penalty
in the report. The quadratic term alone makes a small overshoot almost free:
0.3 kW over the limit costs 0.45 Rs at lam = 5. The linear term puts a floor
under that. Both terms are convex and grow with the load, so every guarantee
below holds with or without it. A fixed charge for any overload would not be
convex and would break them, so it is deliberately not offered.

What gets split
---------------
Let B be the load the obedient EVs actually drew in the slot, that is, the
executed load of everyone who is not a defector. The cost of a set S of
defectors is the extra penalty they add on top of B::

    c(S) = P(B + sum of p_rated_kw over S) - P(B)

B uses what the obedient EVs did, not what they were told. An EV told to
charge may wait instead, and the load it leaves free must count, or a
defector would be charged for an overload that never happened. Two results
follow:

* If the transformer was not overloaded, nobody pays.
* The shares never add up to more than the real penalty P(L(act)). They are
  equal to it whenever B itself is under the limit, which is always the case
  with the mediator, since its recommendations are safe.

B can be over the limit only when the recommendation itself is unsafe, as a
baseline such as "charge immediately" can produce. That part of the overload
is the policy's fault, not the defectors', so it is not charged to them. All
three rules below split the same amount, c(D), so they can be compared
fairly.

When every obedient EV does exactly what it was told, B is the load of the
recommendation, and a lone defector's c({i}) equals the rho_i the mediator
uses (``lone_defection_penalty``).
"""

from __future__ import annotations

from math import factorial

import numpy as np

from evsim.common import EPS

# Exact Shapley costs d * 2^d evaluations. N is at most 12 in our experiments,
# so 16 leaves room without letting a mistake run for minutes.
MAX_EXACT_DEFECTORS = 16


# ---------------------------------------------------------------- checks ----

def _as_actions(x, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}")
    if arr.size and not np.all((arr == 0) | (arr == 1)):
        raise ValueError(
            f"{name} must contain only 0 and 1. Pass actions, not kW values."
        )
    return arr.astype(np.int64)


def _as_ratings(p_rated_kw, n: int) -> np.ndarray:
    p = np.asarray(p_rated_kw, dtype=float)
    if p.shape != (n,):
        raise ValueError(f"p_rated_kw must have shape ({n},), got {p.shape}")
    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("p_rated_kw must be positive and finite")
    return p


def _check_weight(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite number >= 0, got {value}")
    return value


def _check_c_max(c_max_kw) -> float:
    c_max_kw = float(c_max_kw)
    if not np.isfinite(c_max_kw) or c_max_kw < 0:
        raise ValueError(f"c_max_kw must be finite and >= 0, got {c_max_kw}")
    return c_max_kw


def _check_inputs(rec, act, p_rated_kw, c_max_kw, lam, lam_lin):
    rec = _as_actions(rec, "rec")
    act = _as_actions(act, "act")
    if rec.shape != act.shape:
        raise ValueError(f"rec and act differ in shape: {rec.shape} vs {act.shape}")
    p = _as_ratings(p_rated_kw, rec.size)
    return (rec, act, p, _check_c_max(c_max_kw), _check_weight(lam, "lam"),
            _check_weight(lam_lin, "lam_lin"))


def check_connected(rec, act, connected) -> None:
    """Raise if an EV that is not connected was told to charge or charged.

    The penalty functions cannot tell a disconnected EV from one that waited,
    so a disconnected EV with ``act == 1`` would be counted as a defector.
    The simulator should call this once per slot before splitting the penalty.
    """
    rec = _as_actions(rec, "rec")
    act = _as_actions(act, "act")
    conn = np.asarray(connected, dtype=bool)
    if not (rec.shape == act.shape == conn.shape):
        raise ValueError("rec, act and connected must have the same shape")
    bad = np.flatnonzero(~conn & ((rec == 1) | (act == 1)))
    if bad.size:
        raise ValueError(
            f"EVs {bad.tolist()} are not connected but have rec or act set to 1"
        )


# ------------------------------------------------------- basic formulas ----

def _overload_from_load(load_kw, c_max_kw):
    """Overload in kW for a scalar or an array of loads.

    Uses the project-wide tolerance: a load within EPS of the limit is safe.
    """
    excess = np.asarray(load_kw, dtype=float) - c_max_kw
    return np.where(excess > EPS, excess, 0.0)


def _penalty_from_load(load_kw, c_max_kw, lam, lam_lin):
    ov = _overload_from_load(load_kw, c_max_kw)
    return lam * ov ** 2 + lam_lin * ov


def overload_kw(act, p_rated_kw, c_max_kw) -> float:
    """Overload of an executed profile: max(0, L(act) - c_max_kw), in kW."""
    act = _as_actions(act, "act")
    p = _as_ratings(p_rated_kw, act.size)
    return float(_overload_from_load(act @ p, _check_c_max(c_max_kw)))


def penalty_total(overload_kw: float, lam: float, *, lam_lin: float = 0.0) -> float:
    """Total penalty in Rs for an overload in kW: lam * x^2 + lam_lin * x."""
    lam = _check_weight(lam, "lam")
    lam_lin = _check_weight(lam_lin, "lam_lin")
    ov = float(overload_kw)
    if not np.isfinite(ov) or ov < 0:
        raise ValueError(f"overload_kw must be finite and >= 0, got {ov}")
    return lam * ov * ov + lam_lin * ov


def lone_defection_penalty(load_others_kw, p_i_kw, c_max_kw, lam, *,
                           lam_lin: float = 0.0):
    """Penalty EV i pays if it is the only one to disobey.

    ``load_others_kw`` is the load of the recommended profile without EV i.
    This is rho_i in the report, the quantity the mediator uses in its
    "told to wait" obedience constraint. It accepts numpy arrays, so the
    mediator can evaluate all 2^N profiles in one call.
    """
    lam = _check_weight(lam, "lam")
    lam_lin = _check_weight(lam_lin, "lam_lin")
    c_max_kw = _check_c_max(c_max_kw)
    base = np.asarray(load_others_kw, dtype=float)
    p_i = np.asarray(p_i_kw, dtype=float)
    if not np.all(np.isfinite(base)) or np.any(base < 0):
        raise ValueError("load_others_kw must be finite and >= 0")
    if not np.all(np.isfinite(p_i)) or np.any(p_i <= 0):
        raise ValueError("p_i_kw must be positive and finite")
    out = (_penalty_from_load(base + p_i, c_max_kw, lam, lam_lin)
           - _penalty_from_load(base, c_max_kw, lam, lam_lin))
    return float(out) if np.ndim(out) == 0 else out


# ------------------------------------------------------------ defectors ----

def find_defectors(rec, act) -> np.ndarray:
    """Indices of EVs that were told to wait and charged anyway."""
    rec = _as_actions(rec, "rec")
    act = _as_actions(act, "act")
    if rec.shape != act.shape:
        raise ValueError(f"rec and act differ in shape: {rec.shape} vs {act.shape}")
    return np.flatnonzero((rec == 0) & (act == 1))


def _obedient_load(rec, act, p) -> float:
    """B: executed load of every EV that is not a defector."""
    obedient = ~((rec == 0) & (act == 1))
    return float(act[obedient] @ p[obedient])


def defection_cost(rec, act, p_rated_kw, c_max_kw, lam, *,
                   lam_lin: float = 0.0) -> float:
    """c(D): the extra penalty caused by all defectors together, in Rs.

    This is the amount every split rule divides. It is zero when the
    executed profile is within the limit, and equals the real penalty of the
    slot whenever the obedient EVs' own load is within the limit.
    """
    rec, act, p, c_max_kw, lam, lam_lin = _check_inputs(
        rec, act, p_rated_kw, c_max_kw, lam, lam_lin)
    base = _obedient_load(rec, act, p)
    d = find_defectors(rec, act)
    return float(_penalty_from_load(base + p[d].sum(), c_max_kw, lam, lam_lin)
                 - _penalty_from_load(base, c_max_kw, lam, lam_lin))


def coalition_costs(rec, act, p_rated_kw, c_max_kw, lam, *, lam_lin: float = 0.0):
    """Cost c(S) of every subset S of the defectors.

    Returns ``(defectors, costs)``. Subset ``mask`` contains defector
    ``defectors[k]`` when ``(mask >> k) & 1`` is 1, the same bitmask rule the
    mediator uses for joint profiles. ``costs[0]`` is always 0.
    """
    rec, act, p, c_max_kw, lam, lam_lin = _check_inputs(
        rec, act, p_rated_kw, c_max_kw, lam, lam_lin)
    dset = find_defectors(rec, act)
    d = dset.size
    if d > MAX_EXACT_DEFECTORS:
        raise ValueError(
            f"{d} defectors is more than the exact limit of {MAX_EXACT_DEFECTORS}"
        )
    base = _obedient_load(rec, act, p)
    masks = np.arange(1 << d)
    bits = (masks[:, None] >> np.arange(d)) & 1          # (2^d, d)
    sub_kw = bits @ p[dset] if d else np.zeros(1)
    costs = (_penalty_from_load(base + sub_kw, c_max_kw, lam, lam_lin)
             - _penalty_from_load(base, c_max_kw, lam, lam_lin))
    return dset, costs


# ---------------------------------------------------------- split rules ----

def shapley_split(rec, act, p_rated_kw, c_max_kw, lam, *,
                  lam_lin: float = 0.0) -> np.ndarray:
    """Penalty of each EV under the Shapley value over the defectors.

    Obedient EVs get 0. The shares add up to ``defection_cost``. Because the
    penalty is convex in the load, a defector's Shapley share is never below
    c({i}), the penalty it would pay if it were the only defector against the
    same obedient load (see test_penalty.py).
    """
    rec, act, p, c_max_kw, lam, lam_lin = _check_inputs(
        rec, act, p_rated_kw, c_max_kw, lam, lam_lin)
    pen = np.zeros(rec.size)
    dset, costs = coalition_costs(rec, act, p, c_max_kw, lam, lam_lin=lam_lin)
    d = dset.size
    if d == 0:
        return pen
    if d == 1:
        pen[dset[0]] = costs[1]
        return pen

    masks = np.arange(1 << d)
    size = np.zeros(1 << d, dtype=np.int64)
    for k in range(d):
        size += (masks >> k) & 1
    # Weight of a coalition of size s that does not contain the player.
    weight = np.array([factorial(s) * factorial(d - s - 1) / factorial(d)
                       for s in range(d)])
    for k in range(d):
        bit = 1 << k
        without = masks[(masks & bit) == 0]
        marginal = costs[without | bit] - costs[without]
        pen[dset[k]] = float(np.dot(weight[size[without]], marginal))
    return pen


def equal_split(rec, act, p_rated_kw, c_max_kw, lam, *,
                lam_lin: float = 0.0) -> np.ndarray:
    """Every defector pays the same share of c(D)."""
    rec, act, p, c_max_kw, lam, lam_lin = _check_inputs(
        rec, act, p_rated_kw, c_max_kw, lam, lam_lin)
    pen = np.zeros(rec.size)
    dset = find_defectors(rec, act)
    if dset.size:
        total = defection_cost(rec, act, p, c_max_kw, lam, lam_lin=lam_lin)
        pen[dset] = total / dset.size
    return pen


def proportional_split(rec, act, p_rated_kw, c_max_kw, lam, *,
                       lam_lin: float = 0.0) -> np.ndarray:
    """Defectors share c(D) in proportion to their charger rating."""
    rec, act, p, c_max_kw, lam, lam_lin = _check_inputs(
        rec, act, p_rated_kw, c_max_kw, lam, lam_lin)
    pen = np.zeros(rec.size)
    dset = find_defectors(rec, act)
    if dset.size:
        total = defection_cost(rec, act, p, c_max_kw, lam, lam_lin=lam_lin)
        pen[dset] = total * p[dset] / p[dset].sum()
    return pen


# Names match the penalty_rule argument of sim.run (plan, section 3.6).
SPLIT_RULES = {
    "shapley": shapley_split,
    "equal": equal_split,
    "proportional": proportional_split,
}


def split_penalty(rule: str, rec, act, p_rated_kw, c_max_kw, lam, *,
                  lam_lin: float = 0.0) -> np.ndarray:
    """Dispatch by rule name, for the simulator."""
    try:
        fn = SPLIT_RULES[rule]
    except KeyError:
        raise ValueError(
            f"unknown penalty rule {rule!r}; use one of {sorted(SPLIT_RULES)}"
        ) from None
    return fn(rec, act, p_rated_kw, c_max_kw, lam, lam_lin=lam_lin)
