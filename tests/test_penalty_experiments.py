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


def test_spare_mode_is_safe_and_sometimes_leaves_room():
    rng = np.random.default_rng(3)
    room = 0
    for _ in range(500):
        ev = sample_event(rng, 8, 0.3, rec_mode="spare")
        load = ev.rec @ ev.p_rated_kw
        assert load <= ev.c_max_kw + 1e-9
        off = ev.rec == 0
        room += bool(np.any(load + ev.p_rated_kw[off] <= ev.c_max_kw))
    assert room > 100        # an EV told to wait would often have fitted


def test_spare_mode_produces_defectors_harmless_alone():
    rng = np.random.default_rng(4)
    harmless = 0
    for _ in range(500):
        res = evaluate_event(sample_event(rng, 8, 0.5, rec_mode="spare"), 5.0)
        harmless += int(np.sum(res["lone_rs"] == 0))
    assert harmless > 50


def test_unknown_mode_is_refused():
    import pytest
    with pytest.raises(ValueError):
        sample_event(np.random.default_rng(0), 4, 0.3, rec_mode="half")


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


def test_table_percentages_do_not_depend_on_lambda(tmp_path):
    """The report says the percentages hold for any lambda > 0."""
    a = run_penalty_rules.summarise(*run_penalty_rules.run(
        events_per_n=60, lam=1.0, out_dir=tmp_path / "a"))
    b = run_penalty_rules.summarise(*run_penalty_rules.run(
        events_per_n=60, lam=37.0, out_dir=tmp_path / "b"))
    cols = [c for c in a.columns if c.startswith("pct_") or "gap" in c]
    assert np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float),
                       atol=1e-9, equal_nan=True)
    assert np.allclose(b["rs_per_kw_7"], 37.0 * a["rs_per_kw_7"])


def test_figures_and_table_are_written(tmp_path):
    from experiments import figures_penalty as fp
    res, figs = tmp_path / "results", tmp_path / "figures"
    dfd, dfe = run_penalty_rules.run(events_per_n=40, out_dir=res)
    run_penalty_rules.summarise(dfd, dfe).to_csv(
        res / "penalty_rules_summary.csv", index=False)
    run_defection_rate.run(events=10, out_dir=res)
    fp.make_all(results=res, out_dir=figs)
    for name in ("penalty_rs_per_kw.pdf", "defection_rate.pdf",
                 "penalty_rules_table.tex"):
        assert (figs / name).stat().st_size > 0
    table = (figs / "penalty_rules_table.tex").read_text()
    assert table.count("Shapley &") == 2 and "nan" not in table


def test_figures_survive_a_missing_rating_class(tmp_path):
    """If no 11 kW car defected, the table shows -- and the plot has a gap."""
    from experiments import figures_penalty as fp
    res = tmp_path / "results"
    dfd, dfe = run_penalty_rules.run(events_per_n=40, out_dir=res)
    summary = run_penalty_rules.summarise(dfd, dfe)
    summary.loc[summary.rec_mode == "spare", "rs_per_kw_11"] = float("nan")
    fp.plot_rs_per_kw(summary, tmp_path / "f")
    fp.write_rules_table(summary, tmp_path / "f")
    assert "--" in (tmp_path / "f" / "penalty_rules_table.tex").read_text()
