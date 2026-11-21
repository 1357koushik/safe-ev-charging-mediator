import numpy as np
import pytest

from evsim.common import ETA_DEFAULT, N_SLOTS, RATINGS_KW
from evsim.scenarios import make_scenario


def test_scenario_shapes_and_shared_fields():
    scenario = make_scenario(4, 22.0, 123)

    assert len(scenario.evs) == 4
    assert scenario.soc_init.shape == (4,)
    assert scenario.price.shape == (N_SLOTS,)
    assert scenario.c_max_kw == 22.0
    assert scenario.seed == 123


def test_scenario_values_respect_documented_ranges():
    scenario = make_scenario(30, 40.0, 7)

    assert np.all((scenario.soc_init >= 0.20) & (scenario.soc_init < 0.60))
    assert np.all(np.isfinite(scenario.price))
    assert np.all(scenario.price > 0.0)

    for i, ev in enumerate(scenario.evs):
        assert ev.ev_id == i
        assert ev.e_cap_kwh in (40.0, 50.0, 60.0)
        assert ev.p_rated_kw in RATINGS_KW
        assert ev.eta == ETA_DEFAULT
        assert 0.70 <= ev.soc_target < 0.90
        assert 0 <= ev.arrive_slot <= 8
        assert 36 <= ev.depart_slot <= 52
        assert ev.arrive_slot < ev.depart_slot


def test_tariff_has_expected_time_of_use_bands():
    price = make_scenario(1, 10.0, 1).price
    assert np.all(price[:16] == 8.0)
    assert np.all(price[16:48] == 5.0)
    assert np.all(price[48:] == 7.0)


def test_same_seed_gives_identical_scenario():
    a = make_scenario(12, 50.0, 2025)
    b = make_scenario(12, 50.0, 2025)

    assert a.evs == b.evs
    np.testing.assert_array_equal(a.soc_init, b.soc_init)
    np.testing.assert_array_equal(a.price, b.price)


def test_different_seeds_change_random_ev_parameters():
    a = make_scenario(12, 50.0, 1)
    b = make_scenario(12, 50.0, 2)
    assert a.evs != b.evs or not np.array_equal(a.soc_init, b.soc_init)


@pytest.mark.parametrize("n_ev", [0, -1, 1.5, True])
def test_invalid_ev_count(n_ev):
    with pytest.raises((TypeError, ValueError)):
        make_scenario(n_ev, 20.0, 1)


@pytest.mark.parametrize("capacity", [0.0, -1.0, np.inf, np.nan])
def test_invalid_transformer_capacity(capacity):
    with pytest.raises(ValueError, match="c_max_kw"):
        make_scenario(2, capacity, 1)


def test_seed_must_be_integer():
    with pytest.raises(TypeError, match="seed"):
        make_scenario(2, 20.0, 1.2)
