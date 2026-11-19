"""Tests for evsim/penalty.py.

Golden tests B and C are the hand-worked examples in section 5 of
docs/Work_Division_Plan.md. The property tests check the claims we make in
the report against a slow, independent Shapley computation.
"""

from itertools import permutations
from math import factorial

import numpy as np
import pytest

from evsim.common import RATINGS_KW
from evsim import penalty as pn

TOL = 1e-9


def brute_shapley(rec, act, p, c_max, lam):
    """Shapley by averaging marginal costs over every ordering of defectors."""
    rec, act, p = np.asarray(rec), np.asarray(act), np.asarray(p, float)
    d = list(np.flatnonzero((rec == 0) & (act == 1)))
    # Base load: what every non-defector actually drew.
    base = sum(act[j] * p[j] for j in range(len(rec)) if j not in d)

    def cost(members):
        load = base + sum(p[j] for j in members)
        return (lam * max(0.0, load - c_max) ** 2
                - lam * max(0.0, base - c_max) ** 2)

    out = np.zeros(len(rec))
    for order in permutations(d):
        seen = []
        for j in order:
            out[j] += cost(seen + [j]) - cost(seen)
            seen.append(j)
    return out / factorial(len(d)) if d else out


def random_event(rng, n=None, safe=True, defect_rate=0.5, wait_rate=0.2):
    """A random slot: ratings, limit, a recommendation and the actions taken.

    Some EVs told to charge wait instead (``wait_rate``), so the tests also
    cover slots where the real load is below the recommended one.
    """
    n = n or int(rng.integers(2, 8))
    p = rng.choice(RATINGS_KW, size=n)
    c_max = float(rng.uniform(0.3, 0.8) * p.sum())
    rec = np.zeros(n, dtype=int)
    for i in rng.permutation(n):
        if not safe or rec @ p + p[i] <= c_max:
            rec[i] = int(rng.random() < 0.7) if not safe else 1
    act = rec.copy()
    act[(rec == 0) & (rng.random(n) < defect_rate)] = 1
    act[(rec == 1) & (rng.random(n) < wait_rate)] = 0
    lam = float(rng.uniform(0.5, 10.0))
    return rec, act, p, c_max, lam


# ------------------------------------------------------------ golden ----

def test_golden_b_single_defector():
    rec, act, p = [1, 0], [1, 1], [7.0, 7.0]
    assert pn.overload_kw(act, p, 10.0) == pytest.approx(4.0)
    assert pn.penalty_total(4.0, 1.0) == pytest.approx(16.0)
    for rule in pn.SPLIT_RULES:
        pen = pn.split_penalty(rule, rec, act, p, 10.0, 1.0)
        assert pen == pytest.approx([0.0, 16.0]), rule


def test_golden_c_shapley():
    rec, act, p = [1, 0, 0], [1, 1, 1], [7.0, 3.3, 7.0]
    assert pn.overload_kw(act, p, 10.0) == pytest.approx(7.3)
    pen = pn.shapley_split(rec, act, p, 10.0, 1.0)
    assert pen[0] == 0.0
    assert round(pen[1], 2) == 18.69
    assert round(pen[2], 2) == 34.60
    assert pen.sum() == pytest.approx(53.29)


def test_golden_c_other_rules():
    rec, act, p = [1, 0, 0], [1, 1, 1], [7.0, 3.3, 7.0]
    eq = pn.equal_split(rec, act, p, 10.0, 1.0)
    assert eq == pytest.approx([0.0, 26.645, 26.645])
    pr = pn.proportional_split(rec, act, p, 10.0, 1.0)
    assert pr == pytest.approx([0.0, 53.29 * 3.3 / 10.3, 53.29 * 7.0 / 10.3])


# ----------------------------------------------------- basic behaviour ----

def test_no_defectors_means_no_penalty():
    rec, act, p = [1, 1, 0], [1, 1, 0], [7.0, 7.0, 3.3]
    for rule in pn.SPLIT_RULES:
        assert np.all(pn.split_penalty(rule, rec, act, p, 20.0, 5.0) == 0)


def test_told_to_charge_but_waited_is_not_a_defector():
    rec, act = [1, 1, 0], [0, 1, 1]
    assert list(pn.find_defectors(rec, act)) == [2]


def test_defection_inside_the_headroom_costs_nothing():
    # 7 kW recommended, 10 kW limit, a 3 kW charger fits in the gap.
    rec, act, p = [1, 0], [1, 1], [7.0, 3.0]
    for rule in pn.SPLIT_RULES:
        assert np.all(pn.split_penalty(rule, rec, act, p, 10.0, 5.0) == 0)


def test_load_exactly_at_the_limit_is_safe():
    assert pn.overload_kw([1, 1], [3.3, 6.7], 10.0) == 0.0


def test_lone_penalty_matches_report_formula():
    # rho_i = lam * max(0, L(others) + P_i - C)^2 when the others are safe.
    others = np.array([0.0, 3.3, 7.0, 9.9])
    got = pn.lone_defection_penalty(others, 7.0, 10.0, 2.0)
    want = 2.0 * np.maximum(0.0, others + 7.0 - 10.0) ** 2
    assert got == pytest.approx(want)
    assert isinstance(pn.lone_defection_penalty(7.0, 7.0, 10.0, 1.0), float)


def test_zero_lambda_charges_nobody():
    rec, act, p = [1, 0, 0], [1, 1, 1], [7.0, 3.3, 7.0]
    for rule in pn.SPLIT_RULES:
        assert np.all(pn.split_penalty(rule, rec, act, p, 10.0, 0.0) == 0)


def test_identical_defectors_pay_the_same():
    rec, act, p = [1, 0, 0, 0], [1, 1, 1, 1], [7.0, 7.0, 7.0, 3.3]
    pen = pn.shapley_split(rec, act, p, 12.0, 1.0)
    assert pen[1] == pytest.approx(pen[2])


def test_unsafe_recommendation_only_extra_cost_is_split():
    # The recommendation itself is 14 kW on a 10 kW limit (a bad baseline).
    rec, act, p = [1, 1, 0], [1, 1, 1], [7.0, 7.0, 3.3]
    base_penalty = pn.penalty_total(4.0, 1.0)
    full_penalty = pn.penalty_total(7.3, 1.0)
    for rule in pn.SPLIT_RULES:
        pen = pn.split_penalty(rule, rec, act, p, 10.0, 1.0)
        assert pen[:2] == pytest.approx([0.0, 0.0])
        assert pen.sum() == pytest.approx(full_penalty - base_penalty)


def test_no_real_overload_means_no_penalty_even_if_someone_waited():
    # Told ON, A waits. Told OFF, B charges. Real load 7 kW on a 10 kW limit.
    rec, act, p = [1, 0], [0, 1], [7.0, 7.0]
    assert pn.overload_kw(act, p, 10.0) == 0.0
    for rule in pn.SPLIT_RULES:
        assert np.all(pn.split_penalty(rule, rec, act, p, 10.0, 1.0) == 0), rule


def test_waiting_ev_frees_capacity_for_defectors():
    # EV 0 (7 kW) and EV 1 (3 kW) are told ON; EV 0 waits. EV 2 (3 kW) and
    # EV 3 (7 kW) defect. Real load is 3 + 3 + 7 = 13 kW on a 10 kW limit.
    rec, act, p = [1, 1, 0, 0], [0, 1, 1, 1], [7.0, 3.0, 3.0, 7.0]
    real = pn.penalty_total(pn.overload_kw(act, p, 10.0), 1.0)
    assert real == pytest.approx(9.0)          # 13 kW on 10 kW
    for rule in pn.SPLIT_RULES:
        pen = pn.split_penalty(rule, rec, act, p, 10.0, 1.0)
        assert pen.sum() == pytest.approx(real), rule
        assert pen[0] == 0 and pen[1] == 0, rule


def test_golden_b_with_spare_ev_waiting():
    # Golden B plus a third EV that was told ON and waited. Nothing changes
    # for the defector because the waiting EV was not adding load anyway.
    rec, act, p = [1, 0, 1], [1, 1, 0], [7.0, 7.0, 3.0]
    assert pn.shapley_split(rec, act, p, 10.0, 1.0) == pytest.approx([0, 16, 0])


# ---------------------------------------------------------- properties ----

@pytest.mark.parametrize("seed", range(200))
def test_shapley_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    rec, act, p, c_max, lam = random_event(rng, safe=bool(seed % 2))
    fast = pn.shapley_split(rec, act, p, c_max, lam)
    assert fast == pytest.approx(brute_shapley(rec, act, p, c_max, lam), abs=1e-7)


@pytest.mark.parametrize("seed", range(200))
def test_every_rule_splits_exactly_the_defection_cost(seed):
    rng = np.random.default_rng(1000 + seed)
    rec, act, p, c_max, lam = random_event(rng, safe=bool(seed % 2))
    total = pn.defection_cost(rec, act, p, c_max, lam)
    obedient = ~((np.asarray(rec) == 0) & (np.asarray(act) == 1))
    for rule in pn.SPLIT_RULES:
        pen = pn.split_penalty(rule, rec, act, p, c_max, lam)
        assert pen.sum() == pytest.approx(total, abs=1e-7), rule
        assert np.all(pen[obedient] == 0), rule
        assert np.all(pen >= -TOL), rule


@pytest.mark.parametrize("seed", range(200))
def test_shapley_never_below_lone_penalty_or_any_group_cost(seed):
    """A group of defectors always pays at least what it would cause alone.

    For a single EV this means its share is at least the lone-defector
    penalty the mediator used, so the obedience constraint still holds when
    several EVs disobey together.
    """
    rng = np.random.default_rng(5000 + seed)
    rec, act, p, c_max, lam = random_event(rng, safe=bool(seed % 2))
    dset, costs = pn.coalition_costs(rec, act, p, c_max, lam)
    pen = pn.shapley_split(rec, act, p, c_max, lam)
    for mask in range(1, 1 << dset.size):
        members = [dset[k] for k in range(dset.size) if (mask >> k) & 1]
        assert pen[members].sum() >= costs[mask] - 1e-7


@pytest.mark.parametrize("seed", range(300))
def test_charges_never_exceed_the_real_penalty(seed):
    rng = np.random.default_rng(13000 + seed)
    rec, act, p, c_max, lam = random_event(rng, safe=bool(seed % 2))
    real = pn.penalty_total(pn.overload_kw(act, p, c_max), lam)
    b = sum(a * q for r, a, q in zip(rec, act, p) if not (r == 0 and a == 1))
    for rule in pn.SPLIT_RULES:
        pen = pn.split_penalty(rule, rec, act, p, c_max, lam)
        assert pen.sum() <= real + 1e-7, rule
        if b <= c_max:
            # Obedient load is safe, so the defectors carry the whole penalty.
            assert pen.sum() == pytest.approx(real, abs=1e-7), rule


def test_equal_split_can_undercharge_a_big_defector():
    # One 11 kW and three 3.3 kW chargers defect from a recommendation that
    # fills the transformer exactly. Alone, the 11 kW car would cause 121 Rs
    # of penalty. Equal split charges it less than that.
    rec = [1, 0, 0, 0, 0]
    act = [1, 1, 1, 1, 1]
    p = [10.0, 11.0, 3.3, 3.3, 3.3]
    lone = pn.lone_defection_penalty(10.0, 11.0, 10.0, 1.0)
    assert lone == pytest.approx(121.0)
    assert pn.equal_split(rec, act, p, 10.0, 1.0)[1] < lone
    assert pn.shapley_split(rec, act, p, 10.0, 1.0)[1] >= lone


def test_twelve_defectors_runs_quickly():
    rec = np.zeros(12, dtype=int)
    act = np.ones(12, dtype=int)
    p = np.resize(np.array(RATINGS_KW), 12)
    pen = pn.shapley_split(rec, act, p, 20.0, 1.0)
    assert pen.sum() == pytest.approx(pn.defection_cost(rec, act, p, 20.0, 1.0))


# --------------------------------------------------------- bad inputs ----

@pytest.mark.parametrize("bad", [
    dict(rec=[1, 0], act=[7.0, 7.0]),          # kW passed as actions
    dict(rec=[1, 0, 0], act=[1, 1]),           # shape mismatch
    dict(rec=[[1, 0]], act=[[1, 1]]),          # 2-D
])
def test_rejects_malformed_actions(bad):
    with pytest.raises(ValueError):
        pn.shapley_split(bad["rec"], bad["act"], [7.0, 7.0], 10.0, 1.0)


def test_rejects_bad_numbers():
    with pytest.raises(ValueError):
        pn.shapley_split([1, 0], [1, 1], [7.0, -7.0], 10.0, 1.0)
    with pytest.raises(ValueError):
        pn.shapley_split([1, 0], [1, 1], [7.0, 7.0], 10.0, -1.0)
    with pytest.raises(ValueError):
        pn.penalty_total(-1.0, 1.0)
    with pytest.raises(ValueError):
        pn.split_penalty("median", [1, 0], [1, 1], [7.0, 7.0], 10.0, 1.0)


def test_too_many_defectors_is_refused():
    n = pn.MAX_EXACT_DEFECTORS + 1
    with pytest.raises(ValueError):
        pn.shapley_split(np.zeros(n, int), np.ones(n, int), np.ones(n), 1.0, 1.0)


@pytest.mark.parametrize("seed", range(200))
def test_proportional_also_never_subsidises_a_group(seed):
    """The cost of a group depends only on its total kW and grows faster than
    linearly, so cost per kW rises with group size. Charging by kW therefore
    never asks a group for less than its own cost. This is why proportional
    split passes the same check as Shapley in the experiment."""
    rng = np.random.default_rng(9000 + seed)
    rec, act, p, c_max, lam = random_event(rng, safe=bool(seed % 2))
    dset, costs = pn.coalition_costs(rec, act, p, c_max, lam)
    pen = pn.proportional_split(rec, act, p, c_max, lam)
    for mask in range(1, 1 << dset.size):
        members = [dset[k] for k in range(dset.size) if (mask >> k) & 1]
        assert pen[members].sum() >= costs[mask] - 1e-7
