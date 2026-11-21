"""Study equilibrium incentive gap (epsilon) as a function of lambda.

Owner: Muppala Harshith.

Run from repository root:
    python -m experiments.run_mediator_lambda

Writes results/mediator_lambda.csv, containing epsilon and the fraction of
slots achieving exact correlated equilibrium (eps = 0) as penalty weight lambda varies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evsim import mediator
from evsim.common import BASE_SEED, RATINGS_KW

N_VALUES = (4, 8, 12)
LAMBDAS = tuple(np.round(np.arange(0.0, 8.1, 0.5), 1))
RESULTS = Path("results")

# Distinct offset for mediator lambda experiments
RUN_INDEX_OFFSET = 200

# Scenarios per N value to balance statistical power and runtime
DEFAULT_SCENARIOS = {4: 25, 8: 25, 12: 5}


def run(
    lambdas=LAMBDAS,
    scenarios_per_n=None,
    n_values=N_VALUES,
    out_dir=RESULTS,
) -> pd.DataFrame:
    """Run the lambda sweep across N values and write CSV results."""
    if scenarios_per_n is None:
        scenarios_per_n = DEFAULT_SCENARIOS
    elif isinstance(scenarios_per_n, int):
        scenarios_per_n = {n: scenarios_per_n for n in n_values}

    rows = []
    run_idx = RUN_INDEX_OFFSET

    for n_ev in n_values:
        n_scenarios = scenarios_per_n.get(n_ev, 10)
        seed = BASE_SEED + run_idx
        run_idx += 1
        rng = np.random.default_rng(seed)

        # Pre-generate scenarios (ratings, c_max, Q-values)
        scenarios = []
        for _ in range(n_scenarios):
            p = rng.choice(np.array(RATINGS_KW), size=n_ev)
            c_max = float(rng.uniform(0.3, 0.7) * p.sum())
            # Q-values: OFF around 0-5, ON with positive gain
            q_off = rng.uniform(0.0, 5.0, size=n_ev)
            g = rng.uniform(0.0, 15.0, size=n_ev)
            q = np.column_stack([q_off, q_off + g])
            scenarios.append((p, c_max, q))

        for lam in lambdas:
            lam_val = float(lam)
            eps_list = []
            for p, c_max, q in scenarios:
                res = mediator.recommend(q, p, c_max, lam_val, rng)
                eps_list.append(res.eps)

            eps_arr = np.array(eps_list)
            exact_count = int(np.sum(eps_arr <= 1e-9))
            pct_exact = 100.0 * exact_count / n_scenarios if n_scenarios else 0.0

            rows.append(
                dict(
                    n_ev=n_ev,
                    lam=lam_val,
                    n_scenarios=n_scenarios,
                    mean_eps=float(np.mean(eps_arr)),
                    median_eps=float(np.median(eps_arr)),
                    max_eps=float(np.max(eps_arr)),
                    pct_exact_ce=pct_exact,
                )
            )

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "mediator_lambda.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Run lambda sweep for mediator LP")
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    args = parser.parse_args()

    df = run(out_dir=args.out_dir)
    print("Lambda sweep complete. Summary preview:")
    with pd.option_context("display.width", 120, "display.precision", 3):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
