"""Measure mediator LP runtime scaling against number of EVs (N).

Owner: Muppala Harshith.

Run from repository root:
    python -m experiments.run_mediator_runtime

Writes results/mediator_runtime.csv, recording solver runtimes for N = 2, 4, 6, 8, 10, 12.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evsim import mediator
from evsim.common import BASE_SEED, EPS, RATINGS_KW

N_VALUES = (2, 4, 6, 8, 10, 12)
REPEATS_PER_N = {2: 10, 4: 10, 6: 10, 8: 10, 10: 5, 12: 3}
RESULTS = Path("results")

RUN_INDEX_OFFSET = 300


def run(
    n_values=N_VALUES,
    repeats=None,
    lam=5.0,
    out_dir=RESULTS,
) -> pd.DataFrame:
    """Measure mediator solve time across N values and write CSV results."""
    if repeats is None:
        repeats = REPEATS_PER_N
    elif isinstance(repeats, int):
        repeats = {n: repeats for n in n_values}

    rows = []
    run_idx = RUN_INDEX_OFFSET

    for n_ev in n_values:
        n_repeats = repeats.get(n_ev, 3)
        seed = BASE_SEED + run_idx
        run_idx += 1
        rng = np.random.default_rng(seed)

        for rep in range(n_repeats):
            p = rng.choice(np.array(RATINGS_KW), size=n_ev)
            c_max = float(rng.uniform(0.3, 0.7) * p.sum())
            q = rng.uniform(-5.0, 15.0, size=(n_ev, 2))

            res = mediator.recommend(q, p, c_max, lam, rng)

            _, loads = mediator.enumerate_profiles(n_ev, p)
            n_safe = int(np.sum(loads <= c_max + EPS))

            rows.append(
                dict(
                    n_ev=n_ev,
                    rep=rep,
                    seed=seed,
                    n_profiles=1 << n_ev,
                    n_safe_profiles=n_safe,
                    c_max_kw=c_max,
                    lam=lam,
                    eps=res.eps,
                    status=res.status,
                    runtime_ms=res.runtime_ms,
                )
            )

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "mediator_runtime.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Measure mediator LP runtime vs N")
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    args = parser.parse_args()

    df = run(out_dir=args.out_dir)
    print("Runtime benchmarking complete. Summary preview:")
    summary = df.groupby("n_ev").agg(
        n_profiles=("n_profiles", "first"),
        mean_safe=("n_safe_profiles", "mean"),
        mean_runtime_ms=("runtime_ms", "mean"),
        median_runtime_ms=("runtime_ms", "median"),
        min_runtime_ms=("runtime_ms", "min"),
        max_runtime_ms=("runtime_ms", "max"),
    ).reset_index()
    with pd.option_context("display.width", 120, "display.precision", 2):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
