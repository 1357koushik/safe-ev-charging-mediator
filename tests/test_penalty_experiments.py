"""Checks on the synthetic events used by the penalty experiments."""

import numpy as np
import pandas as pd

from evsim import penalty as pn
from experiments import run_defection_rate, run_penalty_rules
from experiments.penalty_events import evaluate_event, sample_event


def test_recommendation_is_safe_and_maximal():
    rng = np.random.default_rng(0)
    for _ in range(500):
        ev = sample_event(rng, int(rng.integers(2, 13)), 0.3)
        load = ev.rec @ ev.p_rated_kw
        assert load <= ev.c_max_kw + 1e-9
        off = ev.rec == 0
        # No EV that was told to wait would have fitted.
        assert np.all(load + ev.p_rated_kw[off] > ev.c_max_kw)


def test_every_defector_overloads_alone():
    rng = np.random.default_rng(1)
    for _ in range(300):
        ev = sample_event(rng, 8, 0.5)
        res = evaluate_event(ev, 5.0)
        assert np.all(res["lone_rs"] > 0)


def test_zero_defect_rate_means_no_overload():
    rng = np.random.default_rng(2)
    for _ in range(200):
        ev = sample_event(rng, 12, 0.0)
        assert pn.overload_kw(ev.act, ev.p_rated_kw, ev.c_max_kw) == 0.0


def test_experiments_are_reproducible(tmp_path):
    a1, b1 = run_penalty_rules.run(events_per_n=50, out_dir=tmp_path / "a")
    a2, b2 = run_penalty_rules.run(events_per_n=50, out_dir=tmp_path / "b")
    pd.testing.assert_frame_equal(a1, a2)
    pd.testing.assert_frame_equal(b1, b2)
    d1 = run_defection_rate.run(events=20, out_dir=tmp_path / "c")
    d2 = run_defection_rate.run(events=20, out_dir=tmp_path / "d")
    pd.testing.assert_frame_equal(d1, d2)
