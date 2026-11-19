"""Synthetic defection events for studying the penalty rules on their own.

Owner: Moram Raghavendra Sri Koushik.

A full simulator run mixes three things: what the EVs want, what the
mediator recommends, and who disobeys. To compare split rules we only need
the last step, so this module builds slots directly:

1. Draw N charger ratings from RATINGS_KW and a transformer limit that binds.
2. Recommend a maximal safe set: go through the EVs in random order and turn
   each one ON if it still fits. This is the recommendation the existence
   argument in the report builds, with every EV wanting to charge.
3. Every EV told to wait disobeys with probability ``defect_rate``.

Because every EV wants to charge, every defector here overloads the
transformer even alone. That is the hardest case for a split rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evsim import penalty as pn
from evsim.common import RATINGS_KW

RULES = tuple(pn.SPLIT_RULES)      # ("shapley", "equal", "proportional")


@dataclass
class Event:
    rec: np.ndarray
    act: np.ndarray
    p_rated_kw: np.ndarray
    c_max_kw: float


def sample_event(rng: np.random.Generator, n_ev: int, defect_rate: float,
                 tightness=(0.3, 0.7)) -> Event:
    """One slot. ``tightness`` is the range of c_max as a fraction of total
    charger power, so that the limit always binds."""
    p = rng.choice(np.array(RATINGS_KW), size=n_ev)
    c_max = float(rng.uniform(*tightness) * p.sum())
    rec = np.zeros(n_ev, dtype=np.int64)
    load = 0.0
    for i in rng.permutation(n_ev):
        if load + p[i] <= c_max:
            rec[i] = 1
            load += p[i]
    act = rec.copy()
    act[(rec == 0) & (rng.random(n_ev) < defect_rate)] = 1
    return Event(rec, act, p, c_max)


def evaluate_event(ev: Event, lam: float) -> dict:
    """Shares under every rule, plus the fairness checks.

    ``max_subsidy_rs`` is the largest amount by which some group of
    defectors pays less than the penalty it would have caused by itself.
    Zero means no group is subsidised by the others.
    """
    dset, costs = pn.coalition_costs(ev.rec, ev.act, ev.p_rated_kw, ev.c_max_kw, lam)
    d = dset.size
    load_rec = float(ev.rec @ ev.p_rated_kw)
    lone = pn.lone_defection_penalty(load_rec, ev.p_rated_kw[dset], ev.c_max_kw, lam)
    lone = np.atleast_1d(lone)

    masks = np.arange(1 << d)
    bits = ((masks[:, None] >> np.arange(d)) & 1).astype(float)   # (2^d, d)

    out = {"defectors": dset, "lone_rs": lone, "total_rs": float(costs[-1]) if d else 0.0,
           "load_rec_kw": load_rec, "shares": {}, "max_subsidy_rs": {},
           "n_below_lone": {}}
    for rule in RULES:
        pen = pn.split_penalty(rule, ev.rec, ev.act, ev.p_rated_kw, ev.c_max_kw, lam)
        share = pen[dset]
        paid = bits @ share if d else np.zeros(1)
        out["shares"][rule] = share
        out["max_subsidy_rs"][rule] = float(max(0.0, np.max(costs - paid)))
        out["n_below_lone"][rule] = int(np.sum(share < lone - 1e-9))
    return out
