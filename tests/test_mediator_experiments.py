"""Tests for mediator experiment runners and figure generators.

Owner: Muppala Harshith.
"""

from __future__ import annotations

import pandas as pd
import pytest

from experiments import figures_mediator, run_mediator_lambda, run_mediator_runtime


def test_mediator_lambda_runner_and_reproducibility(tmp_path):
    """Verify run_mediator_lambda produces correct structure and is reproducible."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    df_a = run_mediator_lambda.run(
        lambdas=(0.0, 2.0, 5.0),
        scenarios_per_n={4: 3},
        n_values=(4,),
        out_dir=out_a,
    )
    df_b = run_mediator_lambda.run(
        lambdas=(0.0, 2.0, 5.0),
        scenarios_per_n={4: 3},
        n_values=(4,),
        out_dir=out_b,
    )

    pd.testing.assert_frame_equal(df_a, df_b)
    assert (out_a / "mediator_lambda.csv").exists()
    assert len(df_a) == 3
    assert set(df_a.columns) == {
        "n_ev",
        "lam",
        "n_scenarios",
        "mean_eps",
        "median_eps",
        "max_eps",
        "pct_exact_ce",
    }
    # Epsilon should be non-increasing with lambda
    assert df_a.loc[df_a.lam == 5.0, "mean_eps"].iloc[0] <= df_a.loc[df_a.lam == 0.0, "mean_eps"].iloc[0] + 1e-9


def test_mediator_runtime_runner(tmp_path):
    """Verify run_mediator_runtime records timing and profile stats."""
    out_dir = tmp_path / "runtime"
    df = run_mediator_runtime.run(
        n_values=(2, 4),
        repeats={2: 2, 4: 2},
        out_dir=out_dir,
    )

    assert (out_dir / "mediator_runtime.csv").exists()
    assert len(df) == 4
    assert set(df["n_ev"]) == {2, 4}
    assert (df["runtime_ms"] > 0).all()
    assert (df["n_safe_profiles"] <= df["n_profiles"]).all()


def test_mediator_figures_generation(tmp_path):
    """Verify figures_mediator generates valid .pdf and .png files."""
    res_dir = tmp_path / "results"
    fig_dir = tmp_path / "figures"

    df_lambda = run_mediator_lambda.run(
        lambdas=(0.0, 1.0, 3.0),
        scenarios_per_n={4: 2, 8: 2},
        n_values=(4, 8),
        out_dir=res_dir,
    )
    df_runtime = run_mediator_runtime.run(
        n_values=(4, 8),
        repeats={4: 2, 8: 2},
        out_dir=res_dir,
    )

    figures_mediator.make_all(results=res_dir, out_dir=fig_dir)

    for name in ("mediator_eps_lambda.pdf", "mediator_eps_lambda.png",
                 "mediator_runtime.pdf", "mediator_runtime.png"):
        file_path = fig_dir / name
        assert file_path.exists(), f"Missing expected file {name}"
        assert file_path.stat().st_size > 0
