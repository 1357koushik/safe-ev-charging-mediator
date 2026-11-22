"""Sensitivity of the mediator to tariff, charger mix and defection.

Owner: Moram Raghavendra Sri Koushik.

The proposal asked us to vary the tariff, the charger mix and the defection
rate. This runs the full simulator with the CE mediator, N = 8 EVs and
C_max = 20 kW (the policy-comparison setting), changing one thing at a time:

* base:        time-of-use tariff, mixed ratings, nobody defects
* flat tariff: 6 Rs/kWh all night
* all 7 kW:    every charger rated 7 kW
* defect 0.1 / 0.3: EVs told to wait with g > 0 charge anyway with that
  probability, and the Shapley rule splits the penalty

Seeds are BASE_SEED + 400 to 409.

    python -m experiments.run_sensitivity

Writes results/sensitivity.csv and report/figures/sensitivity_table.tex.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from evsim.common import BASE_SEED, LAMBDA_DEFAULT
from evsim.mdp import step_soc
from evsim.scenarios import make_scenario
from evsim.sim import run

N_EV = 8
C_MAX_KW = 20.0
SEEDS = list(range(BASE_SEED + 400, BASE_SEED + 410))
RESULTS = Path("results")
FIGURES = Path("report/figures")

CASES = [
    ("Base (time-of-use tariff, mixed ratings)", "base", 0.0),
    ("Flat tariff, 6 Rs/kWh", "flat", 0.0),
    ("All chargers 7 kW", "seven", 0.0),
    ("Defection rate 0.1", "base", 0.1),
    ("Defection rate 0.3", "base", 0.3),
]


def _scenario(kind: str, seed: int):
    sc = make_scenario(N_EV, C_MAX_KW, seed)
    if kind == "flat":
        sc.price = np.full_like(sc.price, 6.0)
    elif kind == "seven":
        sc.evs = [dataclasses.replace(ev, p_rated_kw=7.0) for ev in sc.evs]
    return sc


def _outcome(sc, df: pd.DataFrame) -> tuple[float, float]:
    reached, unmet = [], 0.0
    for ev in sc.evs:
        last = df[(df.ev_id == ev.ev_id) & (df.slot == ev.depart_slot - 1)].iloc[0]
        final = step_soc(ev, float(last.soc_before), int(last.act))
        reached.append(final >= ev.soc_target - 1e-9)
        unmet += max(0.0, ev.soc_target - final) * ev.e_cap_kwh
    return 100.0 * float(np.mean(reached)), unmet


def run_cases(seeds=SEEDS, out_dir=RESULTS) -> pd.DataFrame:
    rows = []
    for label, kind, rate in CASES:
        for seed in seeds:
            sc = _scenario(kind, seed)
            df = run(sc, "ce_mediator", "shapley", LAMBDA_DEFAULT, rate, seed)
            slots = df.drop_duplicates("slot")
            reached, unmet = _outcome(sc, df)
            rows.append(dict(case=label, kind=kind, defect_rate=rate, seed=seed,
                             overloaded_slots=int((slots.overload_kw > 1e-9).sum()),
                             overload_kwh=float(slots.overload_kw.sum() * 0.25),
                             pct_reached=reached, unmet_kwh=unmet,
                             energy_cost_rs=float(df.energy_cost_rs.sum()),
                             penalty_rs=float(df.pen_rs.sum()),
                             defections=int(df.defected.sum())))
    out = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "sensitivity.csv", index=False)
    return out


def write_table(df: pd.DataFrame, out_dir=FIGURES) -> None:
    order = [c[0] for c in CASES]
    m = df.groupby("case").mean(numeric_only=True).loc[order]
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Case ($N=8$, $\cmax=20$~kW) & Overloaded slots & EVs at target (\%) & "
        r"Unmet energy (kWh) & Electricity cost (Rs) & Penalty (Rs) \\",
        r"\midrule",
    ]
    for case, r in m.iterrows():
        lines.append(f"{case} & {r.overloaded_slots:.1f} & {r.pct_reached:.1f} & "
                     f"{r.unmet_kwh:.1f} & {r.energy_cost_rs:.1f} & {r.penalty_rs:.1f} "
                     + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sensitivity_table.tex").write_text("\n".join(lines) + "\n")


def main():
    df = run_cases()
    write_table(df)
    with pd.option_context("display.width", 160, "display.precision", 1):
        print(df.groupby("case").mean(numeric_only=True)[
            ["overloaded_slots", "pct_reached", "unmet_kwh", "energy_cost_rs",
             "penalty_rs", "defections"]].to_string())


if __name__ == "__main__":
    main()
