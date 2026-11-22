"""Mediator LP and recommendation sampler.

Owner: Muppala Harshith.

This module implements the central mediator described in section 3.4 of
docs/Revised_Ideation_Plan.md and section 3.4 of docs/Work_Division_Plan.md.

The mediator coordinates charging across N electric vehicles under a shared
transformer capacity C_max. Each EV reports its local Q-values (q_off, q_on).
The mediator:
1. Enumerates all 2^N joint charging profiles using the bitmask convention.
2. Identifies and removes unsafe profiles whose total load exceeds c_max_kw.
3. Formulates and solves a two-phase LP using scipy.optimize.linprog (method "highs"):
   - Phase 1: Minimise the slack epsilon >= 0 to find the minimum incentive gap.
   - Phase 2: Fix epsilon = epsilon* and maximise expected total reported utility.
   - Phase 3: Keep that utility and maximise the expected load, so capacity
     is not left idle when EVs are indifferent between now and later.
4. Samples a joint recommendation profile from pi using the provided rng.
5. Returns a MediatorResult containing rec, pi, eps, status, and runtime_ms.

The single-defector penalty uses lone_defection_penalty from evsim.penalty,
which is the project's single source of truth for the penalty formula.
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from scipy.optimize import linprog

from evsim.common import EPS, MediatorResult
from evsim.penalty import lone_defection_penalty

MAX_N_PROFILES = 16


def _check_inputs(
    q,
    p_rated_kw,
    c_max_kw: float,
    lam: float,
    rng: np.random.Generator,
    lam_lin: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float, float, np.random.Generator, float]:
    """Validate inputs to the mediator."""
    q_arr = np.asarray(q, dtype=float)
    if q_arr.ndim != 2 or q_arr.shape[1] != 2:
        raise ValueError(f"q must have shape (N, 2), got {q_arr.shape}")
    if not np.all(np.isfinite(q_arr)):
        raise ValueError("q must contain finite numbers")

    n = q_arr.shape[0]
    if n > MAX_N_PROFILES:
        raise ValueError(
            f"N={n} exceeds maximum supported limit of {MAX_N_PROFILES} profiles"
        )

    p_arr = np.asarray(p_rated_kw, dtype=float)
    if p_arr.shape != (n,):
        raise ValueError(f"p_rated_kw must have shape ({n},), got {p_arr.shape}")
    if n > 0 and (not np.all(np.isfinite(p_arr)) or np.any(p_arr <= 0)):
        raise ValueError("p_rated_kw must be positive and finite")

    c_max = float(c_max_kw)
    if not np.isfinite(c_max) or c_max < 0:
        raise ValueError(f"c_max_kw must be finite and >= 0, got {c_max_kw}")

    lam_val = float(lam)
    if not np.isfinite(lam_val) or lam_val < 0:
        raise ValueError(f"lam must be finite and >= 0, got {lam}")

    lam_lin_val = float(lam_lin)
    if not np.isfinite(lam_lin_val) or lam_lin_val < 0:
        raise ValueError(f"lam_lin must be finite and >= 0, got {lam_lin}")

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator")

    return q_arr, p_arr, c_max, lam_val, rng, lam_lin_val


def enumerate_profiles(n: int, p_rated_kw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Enumerate all 2^N profiles with the bitmask convention.

    Profile b in [0, 2^N - 1] has EV i ON when (b >> i) & 1 == 1.
    Returns (bits, loads), where bits is of shape (2^N, N) and loads is of shape (2^N,).
    """
    m = 1 << n
    if n == 0:
        return np.empty((1, 0), dtype=np.int64), np.zeros(1, dtype=float)
    bits = (np.arange(m)[:, None] >> np.arange(n)) & 1
    loads = bits @ p_rated_kw
    return bits, loads


def profile_to_rec(b: int, n: int) -> np.ndarray:
    """Extract per-EV recommendations 0/1 from joint profile bitmask b."""
    if n == 0:
        return np.empty(0, dtype=np.int64)
    return ((b >> np.arange(n)) & 1).astype(np.int64)


def rec_to_profile(rec: np.ndarray) -> int:
    """Convert per-EV recommendations array into joint profile bitmask b."""
    rec_arr = np.asarray(rec, dtype=np.int64)
    n = rec_arr.size
    if n == 0:
        return 0
    powers = 1 << np.arange(n)
    return int(np.dot(rec_arr, powers))


def recommend(
    q,
    p_rated_kw,
    c_max_kw: float,
    lam: float,
    rng: np.random.Generator,
    *,
    lam_lin: float = 0.0,
) -> MediatorResult:
    """Compute optimal correlated recommendation distribution and sample an action.

    Parameters
    ----------
    q : array-like, shape (N, 2)
        Q-values for each EV: column 0 is OFF, column 1 is ON.
    p_rated_kw : array-like, shape (N,)
        Rated power of each EV in kW.
    c_max_kw : float
        Transformer capacity limit in kW.
    lam : float
        Quadratic overload penalty weight (Rs / kW^2).
    rng : np.random.Generator
        Random number generator for sampling the recommendation.
    lam_lin : float, optional
        Optional linear overload penalty weight (Rs / kW), default 0.0.

    Returns
    -------
    MediatorResult:
        rec : (N,) array of recommendations (0 or 1).
        pi : (2^N,) probability distribution over all joint profiles.
        eps : float, minimum incentive gap epsilon*.
        status : str, solver status.
        runtime_ms : float, solver runtime in milliseconds.
    """
    t_start = time.perf_counter()
    q_arr, p_arr, c_max, lam_val, rng, lam_lin_val = _check_inputs(
        q, p_rated_kw, c_max_kw, lam, rng, lam_lin=lam_lin
    )
    n = q_arr.shape[0]

    # Handle N = 0 edge case
    if n == 0:
        runtime_ms = (time.perf_counter() - t_start) * 1000.0
        return MediatorResult(
            rec=np.empty(0, dtype=np.int64),
            pi=np.ones(1, dtype=float),
            eps=0.0,
            status="optimal",
            runtime_ms=runtime_ms,
        )

    m = 1 << n
    bits, loads = enumerate_profiles(n, p_arr)

    # Safe profiles: L(a) <= c_max_kw + EPS. Profile 0 (all OFF) is always safe.
    safe_mask = loads <= (c_max + EPS)
    safe_indices = np.flatnonzero(safe_mask)
    k_safe = safe_indices.size

    bits_safe = bits[safe_indices]
    loads_safe = loads[safe_indices]

    # EV gains: g_i = Q_i(ON) - Q_i(OFF)
    g = q_arr[:, 1] - q_arr[:, 0]

    # Variables: x = [pi_safe (k_safe); eps (1)]
    # A_eq: sum(pi_safe) = 1
    a_eq = np.zeros((1, k_safe + 1), dtype=float)
    a_eq[0, :k_safe] = 1.0
    b_eq = np.array([1.0], dtype=float)

    # Inequality constraints (2 * N):
    # For each EV i:
    # 1. Told ON: -g_i * sum_{k: a_i=1} pi(k) - eps <= 0
    # 2. Told OFF: sum_{k: a_i=0} pi(k) * (g_i - rho_i(k)) - eps <= 0
    a_ub = np.zeros((2 * n, k_safe + 1), dtype=float)
    b_ub = np.zeros(2 * n, dtype=float)

    for i in range(n):
        # ON constraint
        on_mask = bits_safe[:, i] == 1
        a_ub[2 * i, :k_safe] = np.where(on_mask, -g[i], 0.0)
        a_ub[2 * i, -1] = -1.0

        # OFF constraint
        off_mask = bits_safe[:, i] == 0
        loads_without_i = loads_safe[off_mask]
        rho_i = lone_defection_penalty(
            loads_without_i, p_arr[i], c_max, lam_val, lam_lin=lam_lin_val
        )
        a_ub[2 * i + 1, :k_safe][off_mask] = g[i] - rho_i
        a_ub[2 * i + 1, -1] = -1.0

    bounds_phase1 = [(0.0, 1.0) for _ in range(k_safe)] + [(0.0, None)]

    # Phase 1: minimise eps
    c_phase1 = np.zeros(k_safe + 1, dtype=float)
    c_phase1[-1] = 1.0

    res1 = linprog(
        c_phase1,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds_phase1,
        method="highs",
    )

    if not res1.success:
        # Fallback: everyone waits
        eps_star = float(max(0.0, np.max(g)))
        pi = np.zeros(m, dtype=float)
        pi[0] = 1.0
        rec = np.zeros(n, dtype=np.int64)
        runtime_ms = (time.perf_counter() - t_start) * 1000.0
        return MediatorResult(
            rec=rec,
            pi=pi,
            eps=eps_star,
            status=f"phase1_failed: {res1.message}",
            runtime_ms=runtime_ms,
        )

    eps_star = max(0.0, float(res1.x[-1]))
    if eps_star < EPS:
        eps_star = 0.0

    # Phase 2: fix eps = eps_star and maximise total utility sum_k pi(k) * U(k)
    # Total utility for each safe profile k: sum_{i=0}^{N-1} q[i, a_i]
    util_safe = np.sum(np.where(bits_safe == 1, q_arr[:, 1], q_arr[:, 0]), axis=1)

    c_phase2 = np.zeros(k_safe + 1, dtype=float)
    c_phase2[:k_safe] = -util_safe

    bounds_phase2 = bounds_phase1[:-1] + [(0.0, eps_star)]

    res2 = linprog(
        c_phase2,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds_phase2,
        method="highs",
    )

    if not res2.success:
        # Allow tiny numerical relaxation on eps_star bound if needed
        bounds_phase2[-1] = (0.0, eps_star + 1e-7)
        res2 = linprog(
            c_phase2,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds_phase2,
            method="highs",
        )

    if res2.success:
        pi_safe = res2.x[:k_safe]
        status = "optimal"

        # Phase 3: among the distributions that reach the phase-2 utility,
        # pick one that uses as much of the transformer as possible.
        # Without this, an EV that is indifferent between charging now and
        # later (g_i = 0, common at a flat off-peak price) is almost always
        # told to wait. Capacity left unused in a slot is lost for good, and
        # the local Q-values cannot see that, so cars pile up near departure
        # and miss their targets. Obedience constraints are unchanged.
        u_star = float(-res2.fun)
        tol = 1e-7 * max(1.0, abs(u_star))
        a_ub3 = np.vstack([a_ub, np.append(-util_safe, 0.0)])
        b_ub3 = np.append(b_ub, -(u_star - tol))
        c_phase3 = np.zeros(k_safe + 1, dtype=float)
        c_phase3[:k_safe] = -loads_safe
        res3 = linprog(
            c_phase3,
            A_ub=a_ub3,
            b_ub=b_ub3,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds_phase2,
            method="highs",
        )
        if res3.success:
            pi_safe = res3.x[:k_safe]
    else:
        # If phase 2 failed, fall back to phase 1 solution
        pi_safe = res1.x[:k_safe]
        status = f"phase2_failed: {res2.message}"

    # Assemble full distribution pi over all 2^N profiles
    pi = np.zeros(m, dtype=float)
    pi[safe_indices] = np.maximum(pi_safe, 0.0)
    pi_sum = pi.sum()
    if pi_sum > 0:
        pi = pi / pi_sum
    else:
        pi[0] = 1.0

    # Sample one joint profile
    chosen_idx = int(rng.choice(m, p=pi))
    rec = profile_to_rec(chosen_idx, n)

    runtime_ms = (time.perf_counter() - t_start) * 1000.0
    return MediatorResult(
        rec=rec,
        pi=pi,
        eps=eps_star,
        status=status,
        runtime_ms=runtime_ms,
    )
