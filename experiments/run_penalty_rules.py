"""Compare Shapley, equal and proportional splits on the same defection events.

Owner: Moram Raghavendra Sri Koushik.

Run from the repository root:

    python -m experiments.run_penalty_rules

Writes:
    results/penalty_rules_defectors.csv   one row per defector, event and rule
    results/penalty_rules_events.csv      one row per event and rule
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evsim.common import BASE_SEED, LAMBDA_DEFAULT
from experiments.penalty_events import RULES, evaluate_event, sample_event

N_VALUES = (4, 8, 12)
EVENTS_PER_N = 2000
DEFECT_RATE = 0.3
RESULTS = Path("results")


def run(events_per_n=EVENTS_PER_N, defect_rate=DEFECT_RATE, lam=LAMBDA_DEFAULT,
        out_dir=RESULTS):
    per_defector, per_event = [], []
    event_id = 0
    for run_index, n_ev in enumerate(N_VALUES):
        seed = BASE_SEED + run_index
        rng = np.random.default_rng(seed)
        for _ in range(events_per_n):
            ev = sample_event(rng, n_ev, defect_rate)
            res = evaluate_event(ev, lam)
            d = res["defectors"].size
            if d == 0:
                event_id += 1
                continue
            for rule in RULES:
                per_event.append(dict(
                    event_id=event_id, seed=seed, n_ev=n_ev, lam=lam,
                    defect_rate=defect_rate, c_max_kw=ev.c_max_kw,
                    load_rec_kw=res["load_rec_kw"], n_defectors=d, rule=rule,
                    total_rs=res["total_rs"],
                    max_subsidy_rs=res["max_subsidy_rs"][rule],
                    n_below_lone=res["n_below_lone"][rule]))
                for k, i in enumerate(res["defectors"]):
                    per_defector.append(dict(
                        event_id=event_id, seed=seed, n_ev=n_ev,
                        n_defectors=d, ev_id=int(i),
                        p_rated_kw=float(ev.p_rated_kw[i]), rule=rule,
                        lone_rs=float(res["lone_rs"][k]),
                        share_rs=float(res["shares"][rule][k])))
            event_id += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    dfd = pd.DataFrame(per_defector)
    dfe = pd.DataFrame(per_event)
    dfd.to_csv(out_dir / "penalty_rules_defectors.csv", index=False)
    dfe.to_csv(out_dir / "penalty_rules_events.csv", index=False)
    return dfd, dfe


def summarise(dfd: pd.DataFrame, dfe: pd.DataFrame) -> pd.DataFrame:
    """Numbers for the report table. Only events with two or more defectors
    count, because with one defector every rule gives the same answer."""
    multi_e = dfe[dfe.n_defectors >= 2]
    multi_d = dfd[dfd.n_defectors >= 2].assign(
        rs_per_kw=lambda x: x.share_rs / x.p_rated_kw)
    # How far each rule is from Shapley: half the total absolute difference in
    # shares, as a percentage of the event's penalty. 0 means identical, 100
    # means the whole penalty has moved to different EVs.
    key = ["event_id", "ev_id"]
    ref = multi_d[multi_d.rule == "shapley"].set_index(key).share_rs
    rows = []
    for rule in RULES:
        e = multi_e[multi_e.rule == rule]
        d = multi_d[multi_d.rule == rule]
        diff = (d.set_index(key).share_rs - ref).abs().groupby("event_id").sum()
        tot = e.set_index("event_id").total_rs
        pos = tot > 0
        gap = 100 * 0.5 * diff[pos] / tot[pos]
        row = dict(rule=rule, events=len(e),
                   mean_gap_vs_shapley_pct=gap.mean(),
                   max_gap_vs_shapley_pct=gap.max(),
                   pct_events_someone_below_lone=100 * (e.n_below_lone > 0).mean(),
                   pct_events_group_subsidised=100 * (e.max_subsidy_rs > 1e-9).mean(),
                   pct_defectors_below_lone=100 * (d.share_rs < d.lone_rs - 1e-9).mean())
        for kw, g in d.groupby("p_rated_kw"):
            row[f"rs_per_kw_{kw:g}"] = g.rs_per_kw.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", type=int, default=EVENTS_PER_N)
    args = ap.parse_args()
    dfd, dfe = run(events_per_n=args.events)
    summary = summarise(dfd, dfe)
    summary.to_csv(RESULTS / "penalty_rules_summary.csv", index=False)
    with pd.option_context("display.width", 140, "display.precision", 2):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
