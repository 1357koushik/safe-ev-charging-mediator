"""Synthetic defection events for studying the penalty rules on their own.

Owner: Moram Raghavendra Sri Koushik.

A full simulator run mixes three things: what the EVs want, what the
mediator recommends, and who disobeys. To compare split rules we only need
the last step, so this module builds slots directly:

1. Draw N charger ratings from RATINGS_KW and a transformer limit that binds.
2. Build a safe recommendation in one of two ways:

   * ``"full"``: go through the EVs in random order and turn each one ON if
     it still fits. This is the recommendation the existence argument in the
     report builds when every EV wants to charge. Nobody told to wait fits in
     what is left, so every defector overloads the transformer even alone.
   * ``"spare"``: the same, but an EV that fits is only turned ON with
     probability ``fill_prob``. This stands for EVs that do not need to
     charge now, or a mediator that leaves room. A small defector can then
     fit in the spare capacity and be harmless on its own.

3. Every EV told to wait disobeys with probability ``defect_rate``.

The two kinds matter because split rules differ most in how they treat a
defector that is harmless alone. Testing only ``"full"`` hides that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evsim import penalty as pn
from evsim.common import RATINGS_KW

RULES = tuple(pn.SPLIT_RULES)      # ("shapley", "equal", "proportional")
REC_MODES = ("full", "spare")
FILL_PROB = 0.6


@dataclass
class Event:
    rec: np.ndarray
    act: np.ndarray
    p_rated_kw: np.ndarray
    c_max_kw: float


def sample_event(rng: np.random.Generator, n_ev: int, defect_rate: float,
                 tightness=(0.3, 0.7), rec_mode: str = "full",
                 fill_prob: float = FILL_PROB) -> Event:
    """One slot. ``tightness`` is the range of c_max as a fraction of total
    charger power, so that the limit always binds."""
    if rec_mode not in REC_MODES:
        raise ValueError(f"rec_mode must be one of {REC_MODES}, got {rec_mode!r}")
    p = rng.choice(np.array(RATINGS_KW), size=n_ev)
    c_max = float(rng.uniform(*tightness) * p.sum())
    rec = np.zeros(n_ev, dtype=np.int64)
    load = 0.0
    for i in rng.permutation(n_ev):
        # Draw for every EV so the random stream does not depend on fit.
        keep = rng.random() < fill_prob if rec_mode == "spare" else True
        if load + p[i] <= c_max and keep:
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
    # c({i}): penalty if i were the only defector, against the load the
    # obedient EVs actually drew (here they all follow their recommendation).
    obedient = np.ones(ev.rec.size, dtype=bool)
    obedient[dset] = False
    load_obedient = float(ev.act[obedient] @ ev.p_rated_kw[obedient])
    lone = pn.lone_defection_penalty(load_obedient, ev.p_rated_kw[dset],
                                     ev.c_max_kw, lam)
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
