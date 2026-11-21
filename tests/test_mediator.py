"""Tests for the mediator LP and recommendation sampler.

Owner: Muppala Harshith.
"""

from __future__ import annotations

import numpy as np
import pytest

from evsim import mediator, penalty
from evsim.common import EPS, RATINGS_KW


def test_golden_test_a():
    """Golden Test A from Section 5 of docs/Work_Division_Plan.md.

    Two EVs, 7 kW each, C_max = 10 kW, both with gain g = 8.
    Since 16 * 1 >= 8, the LP returns eps = 0 and only the profiles
    (ON, OFF) and (OFF, ON) have positive probability.
    With g = 20 instead, eps > 0.
    """
    p_rated = np.array([7.0, 7.0])
    c_max = 10.0
    lam = 1.0
    rng = np.random.default_rng(2025)

    # g = 8
    q_8 = np.array([[0.0, 8.0], [0.0, 8.0]])
    res_8 = mediator.recommend(q_8, p_rated, c_max, lam, rng)

    assert res_8.status == "optimal"
    assert res_8.eps == 0.0
    assert np.isclose(np.sum(res_8.pi), 1.0)
    # Profiles: b=0 (OFF, OFF), b=1 (ON, OFF), b=2 (OFF, ON), b=3 (ON, ON)
    assert res_8.pi[0] <= EPS
    assert res_8.pi[3] <= EPS
    assert np.isclose(res_8.pi[1] + res_8.pi[2], 1.0)
    # The sampled recommendation must be either (ON, OFF) or (OFF, ON)
    assert (np.array_equal(res_8.rec, [1, 0]) or np.array_equal(res_8.rec, [0, 1]))

    # g = 20
    q_20 = np.array([[0.0, 20.0], [0.0, 20.0]])
    res_20 = mediator.recommend(q_20, p_rated, c_max, lam, rng)
    assert res_20.status == "optimal"
    assert res_20.eps > 0.0
    assert np.isclose(res_20.eps, 2.0)
    assert res_20.pi[0] <= EPS
    assert res_20.pi[3] <= EPS
    assert np.isclose(res_20.pi[1] + res_20.pi[2], 1.0)


def test_bitmask_enumeration_and_helpers():
    """Verify bitmask encoding: EV i is ON iff (b >> i) & 1 == 1."""
    p_rated = np.array([3.3, 7.0, 11.0])
    bits, loads = mediator.enumerate_profiles(3, p_rated)

    assert bits.shape == (8, 3)
    assert loads.shape == (8,)

    for b in range(8):
        rec = mediator.profile_to_rec(b, 3)
        assert mediator.rec_to_profile(rec) == b
        for i in range(3):
            expected_bit = (b >> i) & 1
            assert bits[b, i] == expected_bit
            assert rec[i] == expected_bit
        expected_load = sum(((b >> i) & 1) * p_rated[i] for i in range(3))
        assert np.isclose(loads[b], expected_load)


def test_recommendation_always_safe():
    """No returned profile or positive probability profile may exceed c_max_kw."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        n = int(rng.integers(2, 7))
        p = rng.choice(RATINGS_KW, size=n)
        c_max = float(rng.uniform(0.2, 0.8) * p.sum())
        q = rng.uniform(-10.0, 30.0, size=(n, 2))
        lam = 5.0

        res = mediator.recommend(q, p, c_max, lam, rng)

        # Probabilities sum to 1
        assert np.isclose(res.pi.sum(), 1.0)
        assert np.all(res.pi >= -EPS)

        # Unsafe profiles must have zero probability
        _, loads = mediator.enumerate_profiles(n, p)
        unsafe = loads > (c_max + EPS)
        assert np.all(res.pi[unsafe] <= EPS)

        # Sampled recommendation must be safe
        rec_load = float(res.rec @ p)
        assert rec_load <= c_max + EPS


def test_obedience_conditions_hold_within_eps():
    """Verify that obedience constraints (eq 131 and 132) hold within eps."""
    rng = np.random.default_rng(123)
    for _ in range(30):
        n = int(rng.integers(2, 6))
        p = rng.choice(RATINGS_KW, size=n)
        c_max = float(rng.uniform(0.3, 0.7) * p.sum())
        q = rng.uniform(-5.0, 20.0, size=(n, 2))
        lam = float(rng.uniform(0.5, 10.0))

        res = mediator.recommend(q, p, c_max, lam, rng)
        bits, loads = mediator.enumerate_profiles(n, p)
        g = q[:, 1] - q[:, 0]

        for i in range(n):
            # ON constraint: g_i * P(i told ON) >= -eps
            on_prob = float(np.sum(res.pi[bits[:, i] == 1]))
            assert g[i] * on_prob >= -res.eps - 1e-6

            # OFF constraint: sum_{a_i=0} pi(a) * (rho_i(a_-i) - g_i) >= -eps
            off_mask = bits[:, i] == 0
            if np.any(off_mask):
                loads_others = loads[off_mask]
                rho_i = penalty.lone_defection_penalty(loads_others, p[i], c_max, lam)
                off_val = float(np.dot(res.pi[off_mask], rho_i - g[i]))
                assert off_val >= -res.eps - 1e-6


def test_negative_gain_ev_not_recommended():
    """An EV that strictly prefers waiting (g_i < 0) should not be recommended ON."""
    rng = np.random.default_rng(999)
    p = np.array([3.3, 7.0, 7.0])
    c_max = 20.0
    lam = 5.0

    # EV 0 prefers waiting: Q(OFF) = 10, Q(ON) = 2 -> g = -8
    # EVs 1 and 2 prefer charging: g = +10
    q = np.array([
        [10.0, 2.0],
        [0.0, 10.0],
        [0.0, 10.0],
    ])

    res = mediator.recommend(q, p, c_max, lam, rng)
    bits, _ = mediator.enumerate_profiles(3, p)

    # EV 0 should have probability 0 of being told ON
    p_ev0_on = np.sum(res.pi[bits[:, 0] == 1])
    assert np.isclose(p_ev0_on, 0.0, atol=EPS)
    assert res.rec[0] == 0


def test_lambda_threshold_behavior():
    """When lambda is sufficiently large, eps must reach 0."""
    p = np.array([7.0, 7.0])
    c_max = 10.0
    q = np.array([[0.0, 12.0], [0.0, 12.0]])  # g = 12
    # delta = 4 kW, delta^2 = 16. Threshold is lambda = 12 / 16 = 0.75.
    rng = np.random.default_rng(77)

    # Below threshold: lambda = 0.5 -> 16 * 0.5 = 8 < 12 -> eps > 0
    res_low = mediator.recommend(q, p, c_max, 0.5, rng)
    assert res_low.eps > 0.0

    # Above threshold: lambda = 1.0 -> 16 * 1.0 = 16 >= 12 -> eps == 0
    res_high = mediator.recommend(q, p, c_max, 1.0, rng)
    assert res_high.eps == 0.0


def test_reproducibility():
    """Passing identical rng instances yields identical recommendations."""
    n = 4
    p = np.array([3.3, 7.0, 7.0, 11.0])
    c_max = 15.0
    lam = 5.0
    q = np.array([[0.0, 5.0], [0.0, 8.0], [0.0, 6.0], [0.0, 9.0]])

    rng1 = np.random.default_rng(2025)
    rng2 = np.random.default_rng(2025)

    res1 = mediator.recommend(q, p, c_max, lam, rng1)
    res2 = mediator.recommend(q, p, c_max, lam, rng2)

    np.testing.assert_allclose(res1.pi, res2.pi)
    np.testing.assert_array_equal(res1.rec, res2.rec)
    assert res1.eps == res2.eps


def test_input_validation():
    """Check invalid shapes, negative values, and wrong types are rejected."""
    rng = np.random.default_rng(0)
    p = np.array([7.0, 7.0])
    q = np.array([[0.0, 5.0], [0.0, 5.0]])

    # Wrong shape q
    with pytest.raises(ValueError, match="shape"):
        mediator.recommend(np.zeros((2, 3)), p, 10.0, 1.0, rng)

    # Wrong shape p_rated
    with pytest.raises(ValueError, match="shape"):
        mediator.recommend(q, np.array([7.0]), 10.0, 1.0, rng)

    # Negative p_rated
    with pytest.raises(ValueError, match="positive"):
        mediator.recommend(q, np.array([7.0, -1.0]), 10.0, 1.0, rng)

    # Negative c_max
    with pytest.raises(ValueError, match="c_max_kw"):
        mediator.recommend(q, p, -5.0, 1.0, rng)

    # Negative lam
    with pytest.raises(ValueError, match="lam"):
        mediator.recommend(q, p, 10.0, -1.0, rng)

    # Non-generator rng
    with pytest.raises(TypeError, match="rng"):
        mediator.recommend(q, p, 10.0, 1.0, "not_an_rng")  # type: ignore

    # Too many EVs (exceeds MAX_N_PROFILES)
    with pytest.raises(ValueError, match="exceeds"):
        mediator.recommend(np.zeros((17, 2)), np.ones(17), 10.0, 1.0, rng)


def test_runtime_measurement_n_values():
    """Verify solver runs and measures runtime for N = 4, 8, 12."""
    rng = np.random.default_rng(2025)
    for n in (4, 8, 12):
        p = rng.choice(RATINGS_KW, size=n)
        c_max = float(0.5 * p.sum())
        q = rng.uniform(0.0, 10.0, size=(n, 2))
        res = mediator.recommend(q, p, c_max, 5.0, rng)
        assert res.status == "optimal"
        assert res.runtime_ms > 0.0
        assert res.rec.shape == (n,)
        assert res.pi.shape == (1 << n,)
        assert res.rec @ p <= c_max + EPS
