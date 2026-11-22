"""Tests for evsim/sim.py and evsim/baselines.py.

Owner: Sura Sravan Kumar.

Run with:
    pytest tests/test_sim.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evsim.common import N_SLOTS, BASE_SEED, LAMBDA_DEFAULT, DT_H
from evsim.scenarios import make_scenario
from evsim.sim import run, POLICY_NAMES, PENALTY_RULE_NAMES
from evsim.baselines import (
    recommend_immediate,
    recommend_price_only,
    recommend_urgency,
    recommend_independent_mdp,
    BASELINE_POLICIES,
)


# ------------------------------------------------------------------ fixtures --

@pytest.fixture
def scenario4():
    return make_scenario(4, 20.0, BASE_SEED)


@pytest.fixture
def scenario8():
    return make_scenario(8, 20.0, BASE_SEED)


# ---------------------------------------------------------------- sim tests --

class TestSimRun:
    def test_output_shape(self, scenario4):
        """DataFrame must have exactly N_SLOTS * N_EV rows and the right columns."""
        df = run(scenario4, "immediate", "shapley", LAMBDA_DEFAULT, 0.0, BASE_SEED)
        n = len(scenario4.evs)
        assert df.shape == (N_SLOTS * n, 24), f"Expected {N_SLOTS * n} rows, got {df.shape}"

    def test_all_policies_run(self, scenario4):
        """Every policy name must run without error."""
        for policy in POLICY_NAMES:
            df = run(scenario4, policy, "shapley", LAMBDA_DEFAULT, 0.0, BASE_SEED)
            assert len(df) == N_SLOTS * len(scenario4.evs)

    def test_all_penalty_rules_run(self, scenario4):
        """Every penalty rule name must run without error."""
        for rule in PENALTY_RULE_NAMES:
            df = run(scenario4, "immediate", rule, LAMBDA_DEFAULT, 0.0, BASE_SEED)
            assert len(df) == N_SLOTS * len(scenario4.evs)

    def test_soc_monotone_when_charging(self, scenario4):
        """SoC must never decrease between consecutive slots for a given EV."""
        df = run(scenario4, "immediate", "shapley", LAMBDA_DEFAULT, 0.0, BASE_SEED)
        for ev_id in df["ev_id"].unique():
            ev_df = df[df["ev_id"] == ev_id].sort_values("slot")
            soc = ev_df["soc_before"].values
            # soc can stay flat (OFF) or increase (ON), never decrease
            assert np.all(np.diff(soc) >= -1e-9), \
                f"EV {ev_id} SoC decreased at some slot"

    def test_no_overload_mediator_zero_defection(self, scenario4):
        """With defect_rate=0 the mediator must never exceed C_max."""
        df = run(scenario4, "ce_mediator", "shapley", LAMBDA_DEFAULT, 0.0, BASE_SEED)
        assert (df["overload_kw"] <= 1e-6).all(), \
            "Mediator produced overload with zero defection"

    def test_zero_penalty_no_defectors(self, scenario4):
        """With defect_rate=0 all pen_rs must be 0."""
        df = run(scenario4, "ce_mediator", "shapley", LAMBDA_DEFAULT, 0.0, BASE_SEED)
        assert (df["pen_rs"].abs() < 1e-9).all(), \
            "Non-zero penalty despite zero defection"

    def test_defection_increases_overload(self, scenario8):
        """High defection rate should produce more overload than zero defection."""
        df0 = run(scenario8, "ce_mediator", "shapley", LAMBDA_DEFAULT, 0.0, BASE_SEED)
        df1 = run(scenario8, "ce_mediator", "shapley", LAMBDA_DEFAULT, 0.5, BASE_SEED + 1)
        ov0 = df0["overload_kw"].sum()
        ov1 = df1["overload_kw"].sum()
        assert ov1 >= ov0, "Higher defection rate should not reduce total overload"

    def test_disconnected_ev_has_zero_rec_act(self, scenario4):
        """Disconnected EVs (not yet arrived or already left) must have rec=act=0."""
        df = run(scenario4, "immediate", "shapley", LAMBDA_DEFAULT, 0.0, BASE_SEED)
        evs = scenario4.evs
        for ev in evs:
            before = df[(df["ev_id"] == ev.ev_id) & (df["slot"] < ev.arrive_slot)]
            after = df[(df["ev_id"] == ev.ev_id) & (df["slot"] >= ev.depart_slot)]
            assert (before["rec"] == 0).all(), \
                f"EV {ev.ev_id} has rec!=0 before arrival"
            assert (before["act"] == 0).all(), \
                f"EV {ev.ev_id} has act!=0 before arrival"
            assert (after["rec"] == 0).all(), \
                f"EV {ev.ev_id} has rec!=0 after departure"
            assert (after["act"] == 0).all(), \
                f"EV {ev.ev_id} has act!=0 after departure"

    def test_load_matches_act_sum(self, scenario4):
        """load_kw in each slot must equal sum of act[i] * p_rated[i]."""
        df = run(scenario4, "immediate", "shapley", LAMBDA_DEFAULT, 0.0, BASE_SEED)
        p_rated = {ev.ev_id: ev.p_rated_kw for ev in scenario4.evs}
        for slot in df["slot"].unique():
            slot_df = df[df["slot"] == slot]
            expected = (slot_df["act"] * slot_df["ev_id"].map(p_rated)).sum()
            actual = slot_df["load_kw"].iloc[0]
            assert abs(actual - expected) < 1e-6, \
                f"slot {slot}: load_kw={actual}, expected {expected}"

    def test_deterministic_with_same_seed(self, scenario4):
        """Two runs with the same seed must produce identical DataFrames (excluding wall-clock timing)."""
        df1 = run(scenario4, "ce_mediator", "shapley", LAMBDA_DEFAULT, 0.1, BASE_SEED)
        df2 = run(scenario4, "ce_mediator", "shapley", LAMBDA_DEFAULT, 0.1, BASE_SEED)
        # mediator_ms is a wall-clock timer and will differ between runs; exclude it.
        skip = {"mediator_ms"}
        cols = [c for c in df1.columns if c not in skip]
        pd.testing.assert_frame_equal(df1[cols], df2[cols])

    def test_invalid_policy_raises(self, scenario4):
        with pytest.raises(ValueError, match="Unknown policy"):
            run(scenario4, "not_a_policy", "shapley", LAMBDA_DEFAULT, 0.0, BASE_SEED)

    def test_invalid_rule_raises(self, scenario4):
        with pytest.raises(ValueError, match="Unknown penalty rule"):
            run(scenario4, "immediate", "not_a_rule", LAMBDA_DEFAULT, 0.0, BASE_SEED)


# ------------------------------------------------------------- baseline tests --

class TestBaselines:
    """Tests for the four baseline recommendation functions."""

    def _make_state(self, n, connected=None, k_left=None, k_req=None, soc=None):
        rng = np.random.default_rng(42)
        if connected is None:
            connected = np.ones(n, dtype=bool)
        return {
            "connected": np.asarray(connected, dtype=bool),
            "k_left": np.asarray(k_left if k_left is not None else np.full(n, 10), dtype=int),
            "k_req": np.asarray(k_req if k_req is not None else np.full(n, 5), dtype=int),
            "soc": np.asarray(soc if soc is not None else np.full(n, 0.3), dtype=float),
        }

    def _rng(self):
        return np.random.default_rng(0)

    def test_immediate_respects_capacity(self):
        """Immediate policy must not exceed c_max_kw."""
        p = np.array([7.0, 7.0, 7.0, 7.0])
        q = np.zeros((4, 2))
        state = self._make_state(4)
        rec = recommend_immediate(q, p, 10.0, state, self._rng())
        assert rec.sum() <= 2  # at most 1 or 2 EVs (7+7=14>10, so at most 1)
        assert (rec * p).sum() <= 10.0 + 1e-9

    def test_disconnected_always_zero(self):
        """All baseline policies must give 0 to disconnected EVs."""
        n = 4
        connected = np.array([True, False, True, False])
        p = np.array([7.0, 7.0, 7.0, 7.0])
        q = np.zeros((n, 2))
        q[:, 1] = 10.0  # strong incentive to charge
        state = self._make_state(n, connected=connected, k_req=[3, 0, 3, 0])

        for fn in BASELINE_POLICIES.values():
            rec = fn(q, p, 50.0, state, self._rng())
            assert rec[1] == 0, f"{fn.__name__}: disconnected EV 1 got rec=1"
            assert rec[3] == 0, f"{fn.__name__}: disconnected EV 3 got rec=1"

    def test_independent_mdp_ignores_capacity(self):
        """Independent MDP charges all EVs with Q_on > Q_off regardless of limit."""
        n = 4
        p = np.array([7.0, 7.0, 7.0, 7.0])
        q = np.array([[0.0, 1.0]] * 4)   # all want to charge
        state = self._make_state(n)
        rec = recommend_independent_mdp(q, p, 10.0, state, self._rng())
        assert rec.sum() == 4   # all 4 EVs get rec=1 despite 28 kW > 10 kW limit

    def test_urgency_charges_most_urgent_first(self):
        """Urgency policy must recommend the EV with highest k_req/k_left first."""
        n = 3
        p = np.array([7.0, 7.0, 7.0])
        q = np.zeros((n, 2))
        # EV 0: k_req=1, k_left=10 → urgency=0.1 (least urgent)
        # EV 1: k_req=9, k_left=10 → urgency=0.9 (most urgent)
        # EV 2: k_req=5, k_left=10 → urgency=0.5
        state = self._make_state(n, k_req=[1, 9, 5], k_left=[10, 10, 10])
        c_max = 10.0  # only room for one 7 kW EV
        rec = recommend_urgency(q, p, c_max, state, self._rng())
        assert rec[1] == 1, "Most urgent EV (index 1) should get rec=1"
        assert (rec * p).sum() <= c_max + 1e-9

    def test_price_only_charges_cheapest_first(self):
        """Price-only charges EV with highest Q_on (cheapest slot) first."""
        n = 2
        p = np.array([7.0, 7.0])
        # EV 0 has lower Q_on (more expensive now), EV 1 has higher Q_on (cheaper)
        q = np.array([[0.0, -5.0], [0.0, -2.0]])
        state = self._make_state(n, k_req=[3, 3])
        c_max = 10.0  # only room for one
        rec = recommend_price_only(q, p, c_max, state, self._rng())
        assert rec[1] == 1, "EV 1 (higher Q_on = cheaper) should be preferred"
        assert rec[0] == 0

    def test_all_baselines_return_correct_shape(self):
        n = 6
        p = np.full(n, 3.3)
        q = np.zeros((n, 2))
        q[:, 1] = 1.0
        state = self._make_state(n)
        for name, fn in BASELINE_POLICIES.items():
            rec = fn(q, p, 15.0, state, self._rng())
            assert rec.shape == (n,), f"{name}: wrong shape {rec.shape}"
            assert set(rec.tolist()).issubset({0, 1}), f"{name}: non-binary output"
