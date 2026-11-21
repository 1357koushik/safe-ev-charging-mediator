import pytest
from evsim.common import EVParams,OFF,ON
from evsim.mdp import step_soc,k_req
import numpy as np
from evsim.mdp import solve_q
@pytest.fixture
def ev():
    return EVParams(ev_id=0,e_cap_kwh=50.0,p_rated_kw=7.0,eta=0.9,soc_target=0.8,arrive_slot=0,depart_slot=10)
def test_step_soc_off(ev):
    assert step_soc(ev,0.4,OFF)==pytest.approx(0.4)
def test_step_soc_on(ev):
    assert step_soc(ev,0.4,ON)==pytest.approx(0.4315)
def test_soc_does_not_exceed_one(ev):
    assert step_soc(ev,0.99,ON)==1.0
def test_k_req_already_at_target(ev):
    assert k_req(ev,0.8)==0
def test_k_req_below_target(ev):
    assert k_req(ev,0.4)==13
def test_must_charge_when_no_time_left(ev):
    # Only one charging slot remains
    ev_short = EVParams(
        ev_id=0,
        e_cap_kwh=50.0,
        p_rated_kw=7.0,
        eta=0.9,
        soc_target=0.8,
        arrive_slot=0,
        depart_slot=1,
    )

    price = np.ones(52)

    q_off, q_on = solve_q(ev_short, 0.78, 0, price)

    assert q_on < q_off


def test_no_need_to_charge_at_target(ev):
    price = np.ones(52)

    q_off, q_on = solve_q(ev, 0.8, 0, price)

    assert q_off < q_on


def test_flat_price_and_spare_time(ev):
    # Give the EV 20 slots; it needs only 13.
    ev_spare = EVParams(
        ev_id=0,
        e_cap_kwh=50.0,
        p_rated_kw=7.0,
        eta=0.9,
        soc_target=0.8,
        arrive_slot=0,
        depart_slot=20,
    )

    price = np.ones(52)

    q_off, q_on = solve_q(ev_spare, 0.4, 0, price)

    assert q_off == pytest.approx(q_on)

