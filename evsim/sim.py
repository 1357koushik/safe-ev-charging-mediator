"""Overnight EV charging simulation loop.

Owner: Sura Sravan Kumar.

The public interface is a single function:

    run(scenario, policy, penalty_rule, lam, defect_rate, seed) -> pd.DataFrame

It steps through every slot from 0 to N_SLOTS-1 and returns one row per EV
per slot with the columns specified in Work_Division_Plan.md section 3.7.

Assumptions (also listed in the report, section "Assumptions and limitations")
-------------------------------------------------------------------------------
* Information: an EV sees only its own state, its departure slot and the
  public price of every slot. It never sees other EVs. Under the mediator it
  also receives one number per slot, alpha_t = min(1, C_max / total rating of
  the connected EVs), computed here from what the mediator knows.
* Planning: each EV recomputes its Q-values every slot. Under the mediator it
  treats alpha_t as the chance that a future request to charge is granted,
  the same for every future slot and independent across slots. Baselines plan
  with alpha_t = 1.
* Defection is not strategic: an EV told to wait with g > 0 charges anyway
  with fixed probability defect_rate, whatever the penalty would be.
* Only defectors pay. An EV told to charge that waits pays nothing. The
  penalty is settled per slot with no memory across slots.
* An EV that is not connected has rec = act = 0 and draws nothing.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

from evsim.common import (
    DT_H,
    N_SLOTS,
    EPS,
    Scenario,
)
from evsim.mdp import solve_q, k_req, step_soc
from evsim.mediator import recommend as mediator_recommend
from evsim import penalty as pen_mod
from evsim.baselines import BASELINE_POLICIES


POLICY_NAMES = ["immediate", "price_only", "urgency", "independent_mdp", "ce_mediator"]
PENALTY_RULE_NAMES = ["shapley", "equal", "proportional"]

# Log columns (Work_Division_Plan.md §3.7)
_LOG_COLS = [
    "run_id", "policy", "penalty_rule", "seed", "n_ev", "c_max_kw", "lam",
    "defect_rate", "slot", "ev_id", "soc_before", "k_left", "q_off", "q_on",
    "rec", "act", "defected", "load_kw", "overload_kw", "pen_rs",
    "energy_cost_rs", "eps", "lp_status", "mediator_ms",
]


def run(
    scenario: Scenario,
    policy: str,
    penalty_rule: str,
    lam: float,
    defect_rate: float,
    seed: int,
) -> pd.DataFrame:
    """Simulate one overnight charging session.

    Parameters
    ----------
    scenario : Scenario
        Contains EV parameters, initial SoCs, transformer limit and price.
    policy : str
        One of ``POLICY_NAMES``.
    penalty_rule : str
        One of ``PENALTY_RULE_NAMES``.
    lam : float
        Quadratic overload penalty weight (Rs / kW^2).
    defect_rate : float
        Probability that a connected EV told OFF with g > 0 charges anyway.
    seed : int
        Seed for the run RNG (use BASE_SEED + run_index).

    Returns
    -------
    pd.DataFrame
        One row per EV per slot with columns in ``_LOG_COLS``.
    """
    if policy not in POLICY_NAMES:
        raise ValueError(f"Unknown policy {policy!r}; use one of {POLICY_NAMES}")
    if penalty_rule not in PENALTY_RULE_NAMES:
        raise ValueError(f"Unknown penalty rule {penalty_rule!r}; use one of {PENALTY_RULE_NAMES}")

    evs = scenario.evs
    n = len(evs)
    p_rated = np.array([ev.p_rated_kw for ev in evs], dtype=float)
    c_max = scenario.c_max_kw
    price = scenario.price
    rng = np.random.default_rng(seed)
    run_id = seed

    # Mutable SoC state, one entry per EV.
    soc = scenario.soc_init.copy().astype(float)

    rows: list[dict] = []

    for t in range(N_SLOTS):
        # 1. Which EVs are connected this slot?
        connected = np.array(
            [ev.arrive_slot <= t < ev.depart_slot for ev in evs], dtype=bool
        )

        # 2. Q-values and urgency metrics.
        q = np.zeros((n, 2), dtype=float)
        k_left_arr = np.zeros(n, dtype=int)
        k_req_arr = np.zeros(n, dtype=int)

        # Share of the connected EVs' total rating the transformer can carry.
        # The mediator announces it, and each EV plans with it as the chance
        # that a future request to charge is granted.
        demand_kw = float(p_rated[connected].sum())
        avail = min(1.0, c_max / demand_kw) if demand_kw > 0 else 1.0
        if policy != "ce_mediator":
            avail = 1.0        # baselines keep the plain local plan

        for i, ev in enumerate(evs):
            if connected[i]:
                q[i, 0], q[i, 1] = solve_q(ev, soc[i], t, price, avail=avail)
                k_left_arr[i] = ev.depart_slot - t
                k_req_arr[i] = k_req(ev, soc[i])

        # 3. Build state dict for baselines.
        state = {
            "connected": connected,
            "k_left": k_left_arr,
            "k_req": k_req_arr,
            "soc": soc.copy(),
            "t": t,
            "price": price,
        }

        # 4. Get recommendations.
        eps_val = float("nan")
        lp_status = ""
        mediator_ms_val = float("nan")

        if policy == "ce_mediator":
            # Only connected EVs take part. A disconnected EV passed to the
            # LP would take up capacity in the sampled profile and inflate
            # the lone-defection penalty the others are told to fear.
            rec = np.zeros(n, dtype=np.int64)
            idx = np.flatnonzero(connected)
            if idx.size:
                result = mediator_recommend(q[idx], p_rated[idx], c_max, lam, rng)
                rec[idx] = result.rec.astype(np.int64)
                eps_val = result.eps
                lp_status = result.status
                mediator_ms_val = result.runtime_ms
            else:
                eps_val = 0.0
                lp_status = "no_ev_connected"
                mediator_ms_val = 0.0
        else:
            fn = BASELINE_POLICIES[policy]
            rec = fn(q, p_rated, c_max, state, rng).astype(np.int64)

        # Force rec = 0 for disconnected EVs.
        rec[~connected] = 0

        # 5. Defection: connected EV told OFF with g > 0 charges with prob defect_rate.
        act = rec.copy()
        defected = np.zeros(n, dtype=bool)
        for i in range(n):
            if connected[i] and rec[i] == 0:
                g = q[i, 1] - q[i, 0]
                if g > EPS and rng.random() < defect_rate:
                    act[i] = 1
                    defected[i] = True

        # 6. Validate: no disconnected EV may have rec or act = 1.
        pen_mod.check_connected(rec, act, connected)

        # 7. Slot-level quantities.
        load_kw = float(act @ p_rated)
        overload = pen_mod.overload_kw(act, p_rated, c_max)
        pen_array = pen_mod.split_penalty(penalty_rule, rec, act, p_rated, c_max, lam)

        # 8. Per-EV energy cost.
        energy_cost = act * p_rated * DT_H * price[t]   # (N,) in Rs

        # 9. Write log rows.
        for i, ev in enumerate(evs):
            rows.append({
                "run_id":        run_id,
                "policy":        policy,
                "penalty_rule":  penalty_rule,
                "seed":          seed,
                "n_ev":          n,
                "c_max_kw":      c_max,
                "lam":           lam,
                "defect_rate":   defect_rate,
                "slot":          t,
                "ev_id":         ev.ev_id,
                "soc_before":    soc[i],
                "k_left":        int(k_left_arr[i]),
                "q_off":         q[i, 0],
                "q_on":          q[i, 1],
                "rec":           int(rec[i]),
                "act":           int(act[i]),
                "defected":      bool(defected[i]),
                "load_kw":       load_kw,
                "overload_kw":   overload,
                "pen_rs":        float(pen_array[i]),
                "energy_cost_rs": float(energy_cost[i]),
                "eps":           eps_val,
                "lp_status":     lp_status,
                "mediator_ms":   mediator_ms_val,
            })

        # 10. Update SoC for connected EVs.
        for i, ev in enumerate(evs):
            if connected[i]:
                soc[i] = step_soc(ev, soc[i], int(act[i]))

    df = pd.DataFrame(rows, columns=_LOG_COLS)
    return df
