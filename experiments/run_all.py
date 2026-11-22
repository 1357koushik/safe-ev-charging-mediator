"""Run all experiments in order and regenerate every figure.

Usage (from repo root):
    python -m experiments.run_all

Total run time is noted in the README.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(label: str, module_name: str) -> None:
    t0 = time.perf_counter()
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    import importlib
    mod = importlib.import_module(module_name)
    mod.main()
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")


def main() -> None:
    wall_start = time.perf_counter()

    # 1. Existing team experiments (Koushik & Harshith)
    _run("Penalty rules comparison", "experiments.run_penalty_rules")
    _run("Defection rate sweep", "experiments.run_defection_rate")
    _run("Mediator eps vs lambda", "experiments.run_mediator_lambda")
    _run("Mediator runtime vs N", "experiments.run_mediator_runtime")

    # 2. Sravan's experiments
    _run("Policy comparison (N=4,8,12)", "experiments.run_policy_comparison")
    _run("C_max sweep", "experiments.run_cmax_sweep")
    _run("Sensitivity: tariff, charger mix, defection", "experiments.run_sensitivity")

    # 3. All figures
    _run("Make figures (penalty)", "experiments.figures_penalty")
    _run("Make figures (mediator)", "experiments.figures_mediator")
    _run("Make figures (Sravan)", "experiments.make_figures")

    total = time.perf_counter() - wall_start
    print(f"\n{'='*60}")
    print(f"  All done in {total:.0f}s ({total/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
