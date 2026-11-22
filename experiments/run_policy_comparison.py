"""Policy comparison experiment.

Runs all five policies x three N values x 30 seeds.
Seed range: 200 to 229 (does not clash with penalty/defection experiments).

Usage:
    python -m experiments.run_policy_comparison

Writes: results/policy_comparison.csv
"""

from __future__ import annotations

import os
import sys

# Allow running as script from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from evsim.common import BASE_SEED, LAMBDA_DEFAULT
from evsim.scenarios import make_scenario
from evsim.sim import run, POLICY_NAMES

# ------------------------------------------------------------------ config --
N_EV_LIST = [4, 8, 12]
C_MAX_KW = 20.0          # binds for immediate but mediator can still serve all EVs
LAM = LAMBDA_DEFAULT     # 5.0  Rs / kW^2
DEFECT_RATE = 0.0        # zero defection for the main comparison table
PENALTY_RULE = "shapley"
SEEDS = list(range(BASE_SEED + 200, BASE_SEED + 230))   # 2225 .. 2254

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    frames: list[pd.DataFrame] = []

    total = len(POLICY_NAMES) * len(N_EV_LIST) * len(SEEDS)
    done = 0
    for policy in POLICY_NAMES:
        for n_ev in N_EV_LIST:
            for seed in SEEDS:
                sc = make_scenario(n_ev, C_MAX_KW, seed)
                df = run(sc, policy, PENALTY_RULE, LAM, DEFECT_RATE, seed)
                frames.append(df)
                done += 1
                if done % 20 == 0:
                    print(f"  {done}/{total} runs done")

    result = pd.concat(frames, ignore_index=True)
    out = os.path.join(RESULTS_DIR, "policy_comparison.csv")
    result.to_csv(out, index=False)
    print(f"Saved {len(result)} rows to {out}")


if __name__ == "__main__":
    main()
