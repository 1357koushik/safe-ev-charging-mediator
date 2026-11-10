# Safe Neighbourhood EV Charging

Course project for Foundations of Multi-Agent Systems, Group 25.

Several EVs share one distribution transformer. In every 15-minute slot each EV either charges at its rated power or waits. Each EV plans for itself with a small finite-horizon MDP and reports its value for waiting and for charging. A mediator solves a linear program and sends each EV a private recommendation that keeps the total load under the transformer limit. An EV that is told to wait and charges anyway pays for the overload it causes. When several do this in the same slot, the penalty is split by the Shapley value.

## Status

Planning stage. No code yet. Final submission is 22 November.

## Documents

- [docs/Revised_Ideation_Plan.md](docs/Revised_Ideation_Plan.md): the formulation after the review remarks.
- [docs/Work_Division_Plan.md](docs/Work_Division_Plan.md): who builds what, shared names and interfaces, schedule, GitHub workflow, submission checklist.
- [report/main.tex](report/main.tex): the LaTeX report (uses `arxiv.sty`). Compiled copy: [report/Revised_Ideation_Report.pdf](report/Revised_Ideation_Report.pdf).

## Planned layout

    evsim/          source: common.py, mdp.py, scenarios.py, mediator.py, penalty.py, baselines.py, sim.py
    tests/          pytest tests, one file per module
    experiments/    one script per experiment, make_figures.py, run_all.py
    results/        CSV output of the experiments
    report/         LaTeX report and generated figures
    docs/           plans and README screenshots

## To be added before submission

- Install steps and the single command that reproduces all results.
- A table mapping each figure and table in the report to its script and CSV file.
- Screenshots of the results.

## Team

| Member | Roll no. | Part |
|---|---|---|
| Busa Bhava Ram | 24110083 | Local EV model and scenarios |
| Muppala Harshith | 24110213 | Mediator LP |
| Moram Raghavendra Sri Koushik | 24110210 | Penalty and Shapley split |
| Sura Sravan Kumar | 24110362 | Simulator, baselines, experiments |
