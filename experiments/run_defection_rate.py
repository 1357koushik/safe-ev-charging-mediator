"""Overload and penalties as more EVs ignore the recommendation.

Owner: Moram Raghavendra Sri Koushik.

Run from the repository root:

    python -m experiments.run_defection_rate

Writes results/defection_rate.csv, one row per (N, defect rate, rule).

This uses the synthetic events of experiments/penalty_events.py, where every
EV wants to charge. When sim.run is ready, the same sweep through the full
simulator gives the realistic version, in which only EVs with a positive gain
consider disobeying.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evsim import penalty as pn
from evsim.common import BASE_SEED, LAMBDA_DEFAULT
from experiments.penalty_events import RULES, evaluate_event, sample_event

N_VALUES = (4, 8, 12)
RATES = tuple(np.round(np.arange(0.0, 0.501, 0.05), 2))
EVENTS_PER_POINT = 1000
RESULTS = Path("results")

# Run indices 0-2 are used by run_penalty_rules.py; start after them so the
# two experiments never share a random stream.
RUN_INDEX_OFFSET = 100


def run(events=EVENTS_PER_POINT, lam=LAMBDA_DEFAULT, out_dir=RESULTS):
    rows = []
    run_index = RUN_INDEX_OFFSET
    for n_ev in N_VALUES:
        for rate in RATES:
            seed = BASE_SEED + run_index
            run_index += 1
            rng = np.random.default_rng(seed)
            overload, total, n_def = [], [], []
            below = {r: 0 for r in RULES}
            for _ in range(events):
                ev = sample_event(rng, n_ev, float(rate))
                overload.append(pn.overload_kw(ev.act, ev.p_rated_kw, ev.c_max_kw))
                res = evaluate_event(ev, lam)
                total.append(res["total_rs"])
                n_def.append(res["defectors"].size)
                for r in RULES:
                    below[r] += res["n_below_lone"][r]
            overload = np.array(overload)
            all_def = int(np.sum(n_def))
            for r in RULES:
                rows.append(dict(
                    seed=seed, n_ev=n_ev, defect_rate=float(rate), lam=lam,
                    events=events, rule=r,
                    pct_slots_overloaded=100 * float(np.mean(overload > 0)),
                    mean_overload_kw=float(overload.mean()),
                    mean_penalty_rs=float(np.mean(total)),
                    mean_defectors=float(np.mean(n_def)),
                    pct_defectors_below_lone=(100 * below[r] / all_def) if all_def else 0.0))
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "defection_rate.csv", index=False)
    return df


def main():
    df = run()
    view = df[df.rule == "shapley"][["n_ev", "defect_rate", "pct_slots_overloaded",
                                     "mean_overload_kw", "mean_penalty_rs"]]
    with pd.option_context("display.width", 120, "display.precision", 2):
        print(view.to_string(index=False))


if __name__ == "__main__":
    main()
