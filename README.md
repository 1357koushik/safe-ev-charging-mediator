# Safe Neighbourhood EV Charging: Local Planning, a Correlated Mediator, and Shapley Penalties

**Group 25 — Foundations of AI & Multi-Agent Systems (Semester 5, 2026-27)**

Several electric vehicles (EVs) in one neighbourhood share a distribution transformer. This project implements a slot-by-slot mediator that keeps the total load within the transformer's capacity limit. Each EV solves its own finite-horizon MDP and reports two Q-values per slot; the mediator finds a correlated-equilibrium recommendation over all 2^N joint charging profiles of the connected EVs using a three-phase LP (minimise the incentive gap, maximise reported value, then use as much capacity as possible at that value). The report also shows where this fails: when the EVs together need more energy than the transformer can give in the cheap overnight hours, many cars miss their target, because each EV's own plan ignores congestion. When some EVs disobey, the resulting overload penalty is split by the Shapley value.

---

## Team

| Member | Roll | Module |
|---|---|---|
| Busa Bhava Ram | 24110083 | Local EV model (`evsim/mdp.py`), scenario generator (`evsim/scenarios.py`) |
| Muppala Harshith | 24110213 | Mediator LP (`evsim/mediator.py`), lambda and runtime experiments |
| Moram Raghavendra Sri Koushik | 24110210 | Penalty module (`evsim/penalty.py`), Shapley split, penalty-rule experiments |
| Sura Sravan Kumar | 24110362 | Simulator (`evsim/sim.py`), baselines (`evsim/baselines.py`), experiment scripts, figures, README |

---

## Install

```bash
git clone https://github.com/1357koushik/safe-ev-charging-mediator
cd safe-ev-charging-mediator
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.12+. All dependencies are in `requirements.txt`.

---

## Reproduce all results

```bash
python -m experiments.run_all
```

This runs every experiment script in order, writes CSVs to `results/`, and writes all figures and LaTeX table files to `report/figures/`. Typical total run time: **about 13 minutes** on a laptop (the policy comparison and C_max sweep dominate).

To run individual parts:

```bash
# Tests
python -m pytest

# Individual experiments
python -m experiments.run_policy_comparison     # policy comparison tables, load profile
python -m experiments.run_cmax_sweep            # overload vs C_max figure
python -m experiments.run_sensitivity           # tariff, charger mix, defection table
python -m experiments.run_penalty_rules         # penalty split comparison
python -m experiments.run_defection_rate        # defection rate sweep
python -m experiments.run_mediator_lambda       # eps vs lambda figure
python -m experiments.run_mediator_runtime      # runtime vs N figure

# Figures only (after all CSVs exist)
python -m experiments.make_figures
python -m experiments.figures_penalty
python -m experiments.figures_mediator
```

---

## Figure and table map

| Report item | Script | CSV | Output |
|---|---|---|---|
| Policy comparison tables, N = 4, 8, 12 | `run_policy_comparison.py` | `results/policy_comparison.csv` | `report/figures/policy_table_N{4,8,12}.tex` |
| Load profile figure | `run_policy_comparison.py` | `results/policy_comparison.csv` | `report/figures/load_profile.pdf` |
| Overload vs C_max figure | `run_cmax_sweep.py` | `results/cmax_sweep.csv` | `report/figures/overload_vs_cmax.pdf` |
| Tariff, charger mix and defection table | `run_sensitivity.py` | `results/sensitivity.csv` | `report/figures/sensitivity_table.tex` |
| Penalty rules table | `run_penalty_rules.py` | `results/penalty_rules_summary.csv` | `report/figures/penalty_rules_table.tex` |
| Penalty per kW figure | `run_penalty_rules.py` | `results/penalty_rules_summary.csv` | `report/figures/penalty_rs_per_kw.pdf` |
| Defection rate figure | `run_defection_rate.py` | `results/defection_rate.csv` | `report/figures/defection_rate.pdf` |
| eps vs lambda figure | `run_mediator_lambda.py` | `results/mediator_lambda.csv` | `report/figures/mediator_eps_lambda.pdf` |
| Runtime vs N figure | `run_mediator_runtime.py` | `results/mediator_runtime.csv` | `report/figures/mediator_runtime.pdf` |

Every table and figure in the report is written by these scripts. The numbers quoted in the report text are read from the same CSV files. Runtimes depend on the machine; everything else is identical on every run because all randomness is seeded.

---

## Seed registry

| Seed range | Experiment |
|---|---|
| BASE\_SEED + 0 .. +5 (2025–2030) | `run_penalty_rules.py` |
| BASE\_SEED + 100 .. +132 (2125–2157) | `run_defection_rate.py` |
| BASE\_SEED + 200 .. +229 (2225–2254) | `run_policy_comparison.py` |
| BASE\_SEED + 230 .. +259 (2255–2284) | `run_cmax_sweep.py` |
| BASE\_SEED + 300 .. +305 (2325–2330) | `run_mediator_runtime.py` |
| BASE\_SEED + 400 .. +409 (2425–2434) | `run_sensitivity.py` |

`run_mediator_lambda.py` uses BASE_SEED + 200 onwards for its synthetic Q-values. That overlaps the policy-comparison seeds, but the two experiments draw different things from them, so their results are independent.

`BASE_SEED = 2025` (defined in `evsim/common.py`).

---

## Screenshots and logs

In `docs/screenshots/`:

- `load_profile.png`, `overload_vs_cmax.png`, `penalty_rs_per_kw.png`, `defection_rate.png`, `mediator_eps_lambda.png`, `mediator_runtime.png`: the report figures, copied from `report/figures/`
- `pytest_output.txt`: output of `python -m pytest` for the submitted commit
- `run_all_output.txt`: output of `python -m experiments.run_all` for the submitted commit

---

## Repository link

<https://github.com/1357koushik/safe-ev-charging-mediator>

Submission commit tag: `v1.0-submission`
