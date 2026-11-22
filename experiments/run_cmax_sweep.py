"""C_max sweep experiment.

Varies the transformer limit for each policy to show how overload grows as the
limit tightens.  Uses N = 8 EVs, 30 seeds.
Seed range: 230 to 259.

Usage:
    python -m experiments.run_cmax_sweep

Writes: results/cmax_sweep.csv
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from evsim.common import BASE_SEED, LAMBDA_DEFAULT
from evsim.scenarios import make_scenario
from evsim.sim import run, POLICY_NAMES

# ------------------------------------------------------------------ config --
N_EV = 8
C_MAX_LIST = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0]
LAM = LAMBDA_DEFAULT
DEFECT_RATE = 0.0
PENALTY_RULE = "shapley"
SEEDS = list(range(BASE_SEED + 230, BASE_SEED + 260))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    frames: list[pd.DataFrame] = []

    total = len(POLICY_NAMES) * len(C_MAX_LIST) * len(SEEDS)
    done = 0
    for policy in POLICY_NAMES:
        for c_max in C_MAX_LIST:
            for seed in SEEDS:
                sc = make_scenario(N_EV, c_max, seed)
                df = run(sc, policy, PENALTY_RULE, LAM, DEFECT_RATE, seed)
                frames.append(df)
                done += 1
                if done % 30 == 0:
                    print(f"  {done}/{total} runs done")

    result = pd.concat(frames, ignore_index=True)
    out = os.path.join(RESULTS_DIR, "cmax_sweep.csv")
    result.to_csv(out, index=False)
    print(f"Saved {len(result)} rows to {out}")


if __name__ == "__main__":
    main()
