"""Reproducible synthetic EV charging scenarios.

The project fixes the scenario interface and the quantities to generate, but
not their probability distributions. The defaults below are therefore
explicit modelling assumptions, not values prescribed by the project plan.

Assumptions
-----------
* Battery capacities are sampled uniformly from 40, 50 and 60 kWh.
* Charger ratings are sampled uniformly from common.RATINGS_KW.
* Initial SoC is uniform on [0.20, 0.60); target SoC is uniform on [0.70, 0.90).
* Arrival is sampled from slots 0--8 (18:00--20:00); departure from slots
  36--52 (03:00--07:00), with departure always after arrival.
* Efficiency is common.ETA_DEFAULT.
* The illustrative time-of-use tariff is Rs 8/kWh for slots 0--15,
  Rs 5/kWh for slots 16--47, and Rs 7/kWh for slots 48--51.

All random sampling uses a local numpy.random.Generator created from `seed`.
The tariff is deterministic so that all scenarios share the same price shape.
"""

from __future__ import annotations

import numpy as np

from evsim.common import (
    ETA_DEFAULT,
    N_SLOTS,
    RATINGS_KW,
    EVParams,
    Scenario,
)

# These are generator assumptions, not constants defined by the project plan.
BATTERY_CAPACITIES_KWH = (40.0, 50.0, 60.0)
INITIAL_SOC_RANGE = (0.20, 0.60)
TARGET_SOC_RANGE = (0.70, 0.90)
ARRIVAL_SLOT_RANGE = (0, 9)       # inclusive start, exclusive stop
DEPARTURE_SLOT_RANGE = (36, 53)  # inclusive start, exclusive stop


def _price_profile() -> np.ndarray:
    """Return the illustrative 52-slot time-of-use tariff (Rs/kWh)."""
    price = np.empty(N_SLOTS, dtype=float)
    price[:16] = 8.0       # 18:00--22:00: evening peak
    price[16:48] = 5.0     # 22:00--06:00: overnight off-peak
    price[48:] = 7.0       # 06:00--07:00: morning shoulder
    return price


def make_scenario(n_ev: int, c_max_kw: float, seed: int) -> Scenario:
    """Build one reproducible overnight charging scenario.

    Args:
        n_ev: Number of EVs. Must be a positive integer.
        c_max_kw: Transformer capacity in kW. Must be finite and positive.
        seed: Integer seed used to initialise the scenario's random generator.

    Returns:
        A ``Scenario`` with EV IDs in list order, ``soc_init`` of shape (N,),
        and a 52-element price vector.

    The caller supplies transformer capacity so experiments can vary it while
    holding the EV population and random seed fixed.
    """
    if isinstance(n_ev, (bool, np.bool_)) or not isinstance(n_ev, (int, np.integer)):
        raise TypeError("n_ev must be an integer")
    if n_ev <= 0:
        raise ValueError("n_ev must be positive")

    c_max_kw = float(c_max_kw)
    if not np.isfinite(c_max_kw) or c_max_kw <= 0.0:
        raise ValueError("c_max_kw must be finite and positive")

    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")

    rng = np.random.default_rng(int(seed))
    evs: list[EVParams] = []
    soc_init = rng.uniform(*INITIAL_SOC_RANGE, size=n_ev).astype(float)

    for i in range(n_ev):
        e_cap_kwh = float(rng.choice(BATTERY_CAPACITIES_KWH))
        p_rated_kw = float(rng.choice(RATINGS_KW))
        soc_target = float(rng.uniform(*TARGET_SOC_RANGE))
        arrive_slot = int(rng.integers(*ARRIVAL_SLOT_RANGE))
        depart_slot = int(rng.integers(*DEPARTURE_SLOT_RANGE))

        evs.append(
            EVParams(
                ev_id=i,
                e_cap_kwh=e_cap_kwh,
                p_rated_kw=p_rated_kw,
                eta=ETA_DEFAULT,
                soc_target=soc_target,
                arrive_slot=arrive_slot,
                depart_slot=depart_slot,
            )
        )

    return Scenario(
        evs=evs,
        soc_init=soc_init,
        c_max_kw=c_max_kw,
        price=_price_profile(),
        seed=int(seed),
    )


__all__ = ["make_scenario"]
