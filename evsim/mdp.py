"""Deterministic local EV charging model and finite-horizon value function.

Q-values are utilities (rewards), not costs: a larger Q-value is better.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from evsim.common import DT_H, EPS, OFF, ON, M_UNMET, GAMMA, EVParams


def _validate_ev(ev: EVParams) -> None:
    """Validate physical EV parameters used by the local model."""
    if not np.isfinite(ev.e_cap_kwh) or ev.e_cap_kwh <= 0:
        raise ValueError("e_cap_kwh must be positive and finite")
    if not np.isfinite(ev.p_rated_kw) or ev.p_rated_kw <= 0:
        raise ValueError("p_rated_kw must be positive and finite")
    if not np.isfinite(ev.eta) or not 0 < ev.eta <= 1:
        raise ValueError("eta must be finite and in (0, 1]")
    if not 0 <= ev.soc_target <= 1:
        raise ValueError("soc_target must be between 0 and 1")
    if ev.arrive_slot < 0 or ev.depart_slot <= ev.arrive_slot:
        raise ValueError("departure must be after a non-negative arrival slot")


def _validate_soc(soc: float) -> None:
    if not np.isfinite(soc) or not 0.0 <= soc <= 1.0:
        raise ValueError("soc must be finite and between 0 and 1")


def step_soc(ev: EVParams, soc: float, action: int) -> float:
    """Return the EV's SoC after one 15-minute OFF/ON action."""
    _validate_ev(ev)
    _validate_soc(soc)
    if action not in (OFF, ON):
        raise ValueError("action must be OFF(0) or ON(1)")
    if action == OFF:
        return float(soc)
    increment = ev.eta * ev.p_rated_kw * DT_H / ev.e_cap_kwh
    return min(1.0, float(soc) + increment)


def k_req(ev: EVParams, soc: float) -> int:
    """Return the minimum number of ON slots needed to reach target SoC."""
    _validate_ev(ev)
    _validate_soc(soc)
    deficit = max(0.0, ev.soc_target - soc)
    if deficit <= EPS:
        return 0
    energy_needed_kwh = deficit * ev.e_cap_kwh
    energy_added_per_slot_kwh = ev.eta * ev.p_rated_kw * DT_H
    return math.ceil(energy_needed_kwh / energy_added_per_slot_kwh - EPS)



def solve_q(
    ev: EVParams,
    soc: float,
    t: int,
    price: np.ndarray,
) -> tuple[float, float]:
    """Return (Q_OFF, Q_ON) using backward dynamic programming."""

    # Validate EV parameters and current SoC.
    _validate_ev(ev)
    _validate_soc(soc)

    # Validate the current time slot.
    if t < ev.arrive_slot or t >= ev.depart_slot:
        raise ValueError("t must be an active EV time slot")

    # Convert prices before checking their shape or length.
    price = np.asarray(price, dtype=float)

    if price.ndim != 1:
        raise ValueError("price must be one-dimensional")

    if len(price) < ev.depart_slot:
        raise ValueError("price array is too short")

    if not np.all(np.isfinite(price)):
        raise ValueError("price must contain finite values")

    @lru_cache(maxsize=None)
    def value(current_soc: float, current_t: int) -> float:
        if current_t >= ev.depart_slot:
            unmet_kwh = max(
                0.0, ev.soc_target - current_soc
            ) * ev.e_cap_kwh

            return -M_UNMET * unmet_kwh

        soc_off = step_soc(ev, current_soc, OFF)
        q_off = GAMMA * value(soc_off, current_t + 1)

        soc_on = step_soc(ev, current_soc, ON)
        reward_on = -ev.p_rated_kw * DT_H * price[current_t]
        q_on = reward_on + GAMMA * value(soc_on, current_t + 1)

        return max(q_off, q_on)

    # Calculate the Q-values for the two possible current actions.
    q_off = GAMMA * value(step_soc(ev, soc, OFF), t + 1)

    reward_on = -ev.p_rated_kw * DT_H * price[t]
    q_on = reward_on + GAMMA * value(
        step_soc(ev, soc, ON), t + 1
    )

    return q_off, q_on
