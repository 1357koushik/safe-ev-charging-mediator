import numpy as np
import pytest

from evsim.common import EVParams, OFF, ON
from evsim.mdp import k_req, solve_q, step_soc


@pytest.fixture
def ev():
    return EVParams(
        ev_id=0,
        e_cap_kwh=50.0,
        p_rated_kw=7.0,
        eta=0.9,
        soc_target=0.8,
        arrive_slot=0,
        depart_slot=20,
    )


def test_step_soc_off(ev):
    assert step_soc(ev, 0.4, OFF) == pytest.approx(0.4)


def test_step_soc_on(ev):
    assert step_soc(ev, 0.4, ON) == pytest.approx(0.4315)


def test_soc_does_not_exceed_one(ev):
    assert step_soc(ev, 0.99, ON) == 1.0


def test_step_soc_rejects_invalid_action(ev):
    with pytest.raises(ValueError, match="action"):
        step_soc(ev, 0.4, 2)


def test_step_soc_rejects_invalid_soc(ev):
    with pytest.raises(ValueError, match="soc"):
        step_soc(ev, 1.1, ON)


def test_k_req_already_at_target(ev):
    assert k_req(ev, 0.8) == 0


def test_k_req_below_target(ev):
    assert k_req(ev, 0.4) == 13


def test_exactly_k_req_slots_remaining_requires_charging(ev):
    # At SoC 0.4, 13 ON slots are needed. Exactly 13 slots remain.
    exact_deadline = EVParams(
        ev_id=ev.ev_id,
        e_cap_kwh=ev.e_cap_kwh,
        p_rated_kw=ev.p_rated_kw,
        eta=ev.eta,
        soc_target=ev.soc_target,
        arrive_slot=0,
        depart_slot=13,
    )
    assert exact_deadline.depart_slot == k_req(exact_deadline, 0.4)
    q_off, q_on = solve_q(exact_deadline, 0.4, 0, np.ones(52))
    assert q_on > q_off


def test_ev_already_at_target_prefers_waiting(ev):
    q_off, q_on = solve_q(ev, 0.8, 0, np.ones(52))
    assert q_off >= q_on


def test_flat_price_and_spare_time_gives_equal_q_values(ev):
    q_off, q_on = solve_q(ev, 0.4, 0, np.ones(52))
    assert q_off == pytest.approx(q_on)


def test_urgent_ev_q_values_are_compatible_with_mediator():
    # Integration check: MDP utility Q-values should make the mediator prefer
    # charging when the EV has exactly enough slots to meet its target.
    from evsim.mediator import recommend

    exact_deadline = EVParams(0, 50.0, 7.0, 0.9, 0.8, 0, 13)
    q_off, q_on = solve_q(exact_deadline, 0.4, 0, np.ones(52))
    result = recommend(
        np.array([[q_off, q_on]]),
        np.array([7.0]),
        7.0,
        5.0,
        np.random.default_rng(2025),
    )
    np.testing.assert_array_equal(result.rec, np.array([ON]))


def test_solve_q_rejects_non_1d_price(ev):
    with pytest.raises(ValueError, match="one-dimensional"):
        solve_q(ev, 0.4, 0, np.ones((52, 1)))


def test_solve_q_rejects_non_finite_price(ev):
    prices = np.ones(52)
    prices[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        solve_q(ev, 0.4, 0, prices)

def test_infeasible_deadline_still_prefers_charging(ev):
    # The EV needs 13 charging slots but only 1 slot remains.
    infeasible_ev = EVParams(
        ev_id=ev.ev_id,
        e_cap_kwh=ev.e_cap_kwh,
        p_rated_kw=ev.p_rated_kw,
        eta=ev.eta,
        soc_target=ev.soc_target,
        arrive_slot=0,
        depart_slot=1,
    )

    current_soc = 0.4
    current_slot = 0

    required_slots = k_req(infeasible_ev, current_soc)
    remaining_slots = infeasible_ev.depart_slot - current_slot

    # Confirm that reaching the target is impossible.
    assert required_slots == 13
    assert remaining_slots == 1
    assert remaining_slots < required_slots

    # Even though the target is unreachable, charging reduces
    # the unavoidable unmet-energy penalty.
    q_off, q_on = solve_q(
        infeasible_ev,
        current_soc,
        current_slot,
        np.ones(52),
    )

    assert q_on > q_off


def test_avail_one_matches_plain_plan():
    from evsim.common import EVParams
    from evsim.mdp import solve_q
    import numpy as np
    ev = EVParams(0, 50.0, 7.0, 0.9, 0.8, 0, 40)
    price = np.array([8.0] * 16 + [5.0] * 36)
    assert solve_q(ev, 0.3, 2, price) == solve_q(ev, 0.3, 2, price, avail=1.0)


def test_lower_avail_makes_a_tight_ev_want_to_charge_earlier():
    """With little slack, an EV that expects to be told to wait sometimes
    should value charging in the expensive evening more."""
    from evsim.common import EVParams
    from evsim.mdp import solve_q
    import numpy as np
    ev = EVParams(0, 60.0, 3.3, 0.9, 0.9, 0, 44)
    price = np.array([8.0] * 16 + [5.0] * 36)
    g_full = np.subtract(*solve_q(ev, 0.3, 4, price)[::-1])
    g_half = np.subtract(*solve_q(ev, 0.3, 4, price, avail=0.5)[::-1])
    assert g_half > g_full


def test_avail_must_be_a_probability():
    import pytest
    from evsim.common import EVParams
    from evsim.mdp import solve_q
    import numpy as np
    ev = EVParams(0, 50.0, 7.0, 0.9, 0.8, 0, 40)
    for bad in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            solve_q(ev, 0.3, 2, np.full(52, 5.0), avail=bad)
