"""Shared constants and data classes.

Everything here follows section 3.5 of docs/Work_Division_Plan.md. Change a
value only in this file, and tell the group first, because every module
reads from here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DT_H = 0.25            # slot length in hours
N_SLOTS = 52           # 18:00 to 07:00
GAMMA = 1.0
M_UNMET = 100.0        # Rs per kWh not delivered at departure
LAMBDA_DEFAULT = 5.0   # Rs per kW^2
ETA_DEFAULT = 0.9
RATINGS_KW = (3.3, 7.0, 11.0)
EPS = 1e-9
BASE_SEED = 2025
OFF, ON = 0, 1


@dataclass(frozen=True)
class EVParams:
    ev_id: int
    e_cap_kwh: float
    p_rated_kw: float
    eta: float
    soc_target: float
    arrive_slot: int
    depart_slot: int


@dataclass
class Scenario:
    evs: list[EVParams]
    soc_init: np.ndarray       # (N,)
    c_max_kw: float
    price: np.ndarray          # (N_SLOTS,) Rs per kWh
    seed: int


@dataclass
class MediatorResult:
    rec: np.ndarray            # (N,) of 0/1
    pi: np.ndarray             # (2^N,)
    eps: float
    status: str
    runtime_ms: float
