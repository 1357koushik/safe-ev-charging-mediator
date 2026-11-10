# Work Division Plan: Group 25

Read `Revised_Ideation_Plan.md` first. This file says who builds what, what everyone must keep identical, and in what order we work.

Final submission is Sunday 22 November. We have 12 days from 10 November, so the schedule in section 4 is by date.

What the submission guidelines ask for, and where this plan covers it:

| Guideline | Covered in |
|---|---|
| Complete the LaTeX report from the review, 8 pages | Section 6 |
| Add Methodology (algorithm and solution in detail) | Section 6, owners per subsection |
| Add Results (plots, tables, short discussion) | Sections 6 and 8 |
| Contribution section: task per member and whether it was fulfilled | Section 6.2 |
| Repository link with a README and screenshots of results | Sections 7 and 8 |
| Results in the paper will be checked against the code | Section 8 |

## 1. Before anyone writes code

1. All four read section 3 of the revised plan and work the two-EV example by hand.
2. Meet the instructor with the three questions in section 3.7. Harshith and Koushik should both be there, since the answer changes the mediator and the penalty module.
3. Sravan adds the folder layout and `common.py` from section 3 below to the repository. Nobody starts their module until that file is merged.

## 2. Who owns what

| Member | Module | File |
|---|---|---|
| Busa Bhava Ram | Local EV model and scenario generator | `evsim/mdp.py`, `evsim/scenarios.py` |
| Muppala Harshith | Mediator LP | `evsim/mediator.py` |
| Moram Raghavendra Sri Koushik | Overload, penalty and Shapley split | `evsim/penalty.py` |
| Sura Sravan Kumar | Simulator loop, baselines, experiments, plots | `evsim/sim.py`, `evsim/baselines.py`, `experiments/` |

Each person also writes the report section for their own module, in their own words, and must be able to explain it without notes.

### Bhava Ram: local model and scenarios

- `step_soc`: SoC update for one slot.
- `k_req`: minimum ON slots still needed, and the ratio K / K_req. Handle the zero-deficit case separately.
- `solve_q`: backward induction that returns (Q_off, Q_on) for the current slot. Include the terminal penalty M times unmet kWh. Tip: SoC moves in fixed steps, so the table can be indexed by "ON slots used so far" instead of a SoC grid.
- `make_scenario`: builds a reproducible scenario from a seed (ratings, capacities, initial SoC, targets, arrival and departure slots, price array).
- Tests: an EV with K = K_req must get Q_on > Q_off. An EV already at target must get Q_off >= Q_on. With a flat price and spare time, Q_on and Q_off should be equal.

### Harshith: mediator

- Enumerate profiles with the bitmask convention in section 3.4 and precompute the load of each.
- Build the LP from section 3.4 of the revised plan with `scipy.optimize.linprog` (method `highs`). Two solves: minimise eps, then maximise total utility with eps fixed.
- For the single-defector penalty, call `penalty.penalty_total`. Do not rewrite the formula inside the mediator.
- Sample one profile from pi using the rng passed in.
- Return pi, the sampled recommendation, eps, solver status and runtime.
- Tests: the two-EV example (7 kW, 7 kW, C_max 10) must return eps = 0 once 16 * lambda >= g, and eps > 0 below that. No returned profile may exceed C_max. Probabilities sum to 1.
- Runtime measurement for N = 4, 8, 12.

### Koushik: penalty module

- `overload_kw` and `penalty_total` (lambda times overload squared). These two functions are the single source of the penalty formula for the whole project.
- `find_defectors`: EVs with recommendation 0 and action 1.
- `shapley_split`: exact Shapley over the defector set, using the coalition value from the old section 4.2. Exact enumeration is fine since defectors are at most 12.
- `equal_split` and `proportional_split` with the same signature, for comparison.
- Tests: obedient EVs pay 0. Shares add up to the total. Two identical defectors pay the same. The worked example in section 5 below matches to 2 decimals.

### Sravan: simulator, baselines, experiments

- `run`: the slot loop. For each slot get Q-values, get a recommendation from the chosen policy, apply defection, update SoC, compute penalties, write one log row per EV per slot.
- Defection model: an EV told OFF with g > 0 charges anyway with probability `defect_rate`.
- Baselines: immediate, price-only, urgency, independent MDP. They return a recommendation in the same format as the mediator so the loop does not care which policy is running.
- Experiment scripts that sweep C_max, N, lambda, defect rate and tariff, and write CSV files.
- Plots and tables from the CSV files only, never from in-memory results.
- Start with stub versions of the other three modules (fixed return values) so the loop runs end to end by Wednesday 11 November.

## 3. Things that must be the same for everyone

### 3.1 Tools

- Python 3.11, `numpy`, `scipy`, `matplotlib`, `pandas`, `pytest`. Versions pinned in `requirements.txt`.
- No other solver or library without telling the group.

### 3.2 Folder layout

    evsim/
        common.py        shared constants and data classes
        mdp.py
        scenarios.py
        mediator.py
        penalty.py
        baselines.py
        sim.py
    tests/
        test_mdp.py  test_mediator.py  test_penalty.py  test_sim.py
    experiments/
        run_*.py         one script per experiment, writes CSV
        make_figures.py  reads CSV, writes every figure and table
        run_all.py       runs everything above in order
    results/             CSV output, committed, not edited by hand
    report/
        main.tex  arxiv.sty
        figures/         written by make_figures.py only
    docs/
        screenshots/     images used in the README
    README.md
    requirements.txt

### 3.3 Units

| Quantity | Unit |
|---|---|
| Power | kW |
| Energy | kWh |
| Time | slots (integer). Hours only through `DT_H` |
| SoC | fraction 0 to 1, never percent |
| Price | Rs per kWh |
| lambda | Rs per kW^2 |
| Money | Rs |

### 3.4 Encoding rules

- Actions and recommendations are integers 0 or 1. Power is always `action * p_rated_kw`. Do not pass kW values as actions.
- EVs are indexed 0 to N-1 in the order of the scenario list. That order never changes during a run.
- A joint profile is an integer b from 0 to 2^N - 1. EV i is ON when `(b >> i) & 1` is 1.
- Slot 0 is 18:00 to 18:15. There are 52 slots. `depart_slot` is the first slot in which the EV is gone, so `k_left = depart_slot - t`.
- Per-EV vectors are numpy arrays of shape (N,). Q-values are shape (N, 2), column 0 is OFF and column 1 is ON.
- Randomness: every function that needs it takes a `numpy.random.Generator` argument. Nobody calls `np.random.seed` or the global `np.random` functions. Run seed is `BASE_SEED + run_index`.
- Float comparisons use `EPS`. A profile is safe when `load <= c_max_kw + EPS`.

### 3.5 Names in `common.py`

Constants (the numbers are starting values, change them only in this file):

    DT_H = 0.25
    N_SLOTS = 52
    GAMMA = 1.0
    M_UNMET = 100.0        # Rs per kWh not delivered at departure
    LAMBDA_DEFAULT = 5.0   # Rs per kW^2
    ETA_DEFAULT = 0.9
    RATINGS_KW = (3.3, 7.0, 11.0)
    EPS = 1e-9
    BASE_SEED = 2025
    OFF, ON = 0, 1

Data classes:

    EVParams:  ev_id, e_cap_kwh, p_rated_kw, eta, soc_target, arrive_slot, depart_slot
    Scenario:  evs (list of EVParams), soc_init (N,), c_max_kw, price (N_SLOTS,), seed
    MediatorResult:  rec (N,), pi (2^N,), eps, status, runtime_ms

Symbol to variable name:

| Report symbol | Code name |
|---|---|
| SoC_i,t | `soc` |
| K_i,t | `k_left` |
| K_req | `k_req` |
| P_wattage,i | `p_rated_kw` |
| E_capacity,i | `e_cap_kwh` |
| eta_i | `eta` |
| C_max | `c_max_kw` |
| c_electricity(t) | `price[t]` |
| U_i,t | `q` (row i is `[q_off, q_on]`) |
| g_i | `gain` |
| a*_t | `rec` |
| a_t | `act` |
| L_total | `load_kw` |
| Delta C | `overload_kw` |
| lambda | `lam` (`lambda` is a Python keyword) |
| phi_i | `pen` |

### 3.6 Function signatures

These are fixed. If one has to change, post it in the group and update this file in the same commit.

    # mdp.py
    step_soc(ev: EVParams, soc: float, action: int) -> float
    k_req(ev: EVParams, soc: float) -> int
    solve_q(ev: EVParams, soc: float, t: int, price: np.ndarray) -> tuple[float, float]

    # scenarios.py
    make_scenario(n_ev: int, c_max_kw: float, seed: int) -> Scenario

    # penalty.py
    overload_kw(act, p_rated_kw, c_max_kw) -> float
    penalty_total(overload_kw: float, lam: float) -> float
    find_defectors(rec, act) -> np.ndarray            # indices
    shapley_split(rec, act, p_rated_kw, c_max_kw, lam) -> np.ndarray   # (N,)
    equal_split(...)  and  proportional_split(...)    # same arguments and return

    # mediator.py
    recommend(q, p_rated_kw, c_max_kw, lam, rng) -> MediatorResult

    # baselines.py
    recommend_<name>(q, p_rated_kw, c_max_kw, state, rng) -> np.ndarray   # (N,)

    # sim.py
    run(scenario, policy: str, penalty_rule: str, lam, defect_rate, seed) -> pd.DataFrame

Policy names: `immediate`, `price_only`, `urgency`, `independent_mdp`, `ce_mediator`.
Penalty rule names: `shapley`, `equal`, `proportional`.

### 3.7 Log columns

One row per EV per slot, in this order:

    run_id, policy, penalty_rule, seed, n_ev, c_max_kw, lam, defect_rate,
    slot, ev_id, soc_before, k_left, q_off, q_on, rec, act, defected,
    load_kw, overload_kw, pen_rs, energy_cost_rs, eps, lp_status, mediator_ms

Slot-level values (load, overload, eps) are repeated on each EV row of that slot. For baselines, `eps`, `lp_status` and `mediator_ms` are left empty.

### 3.8 Git

The full GitHub workflow is in section 7. The two rules that matter most: nobody pushes to `main` directly, and `common.py` changes need a message in the group first because they affect everyone.

## 4. Schedule (10 to 22 November)

| Dates | Bhava Ram | Harshith | Koushik | Sravan |
|---|---|---|---|---|
| Tue 10, Wed 11 | `step_soc`, `k_req` with tests | Profile enumeration, LP on the two-EV example | `overload_kw`, `penalty_total`, `find_defectors` | `common.py`, sim loop running on stubs |
| Thu 12 to Sat 14 | `solve_q`, scenario generator, price profile | Full LP with eps, sampler | `shapley_split`, the two comparison rules, help wire the penalty into the LP | Baselines, swap stubs for real modules |
| Sun 15 to Tue 17 | Check odd cases in results (infeasible targets, K < K_req) | lambda threshold and runtime experiments | Penalty rule comparison experiment | Full sweeps, `make_figures.py`, `run_all.py` |
| Wed 18 to Fri 20 | Report: preliminaries, local model | Report: formulation, mediator method and its results | Report: penalty method and its results | Report: experiment setup, baseline results, assembly; README |
| Sat 21 | Fresh-clone check (section 8) | Read the full report for errors in the maths | Page count, figures, references | Fix whatever the fresh-clone check finds |
| Sun 22 | Submit before evening. Tag the commit. | | | |

The instructor meeting about section 3.7 of the revised plan should happen on 10 or 11 November. If he asks for the congestion-game version, only `mediator.py` and `penalty.py` change, and the interfaces stay as they are.

Checkpoints:

- Wed 11, night: one night with 4 EVs runs end to end on stubs.
- Sat 14, night: the same run on real modules, and all three golden tests in section 5 pass.
- Tue 17, night: code freeze. Every figure and table exists in `report/figures/`. After this only bug fixes go in, and any fix means re-running `run_all.py`.
- Fri 20, night: report is complete at 8 pages. Each person explains their section to the other three. If the others cannot follow it, rewrite it.
- Sat 21: fresh-clone check passes.

Short call every evening until the 22nd, 10 minutes. Each person says what is done, what is blocked, and whether any shared name or signature needs to change.

If we fall behind, cut in this order: N = 12 runs, the tariff sweep, the proportional split rule. Do not cut the baselines or the lambda experiment, because the main claim rests on them.

## 5. Golden tests

Everyone should get these numbers. Use lambda = 1.

**Test A (mediator).** Two EVs, 7 kW each, C_max = 10, both with gain g = 8. Since 16 * 1 >= 8, the LP returns eps = 0 and only the profiles (ON, OFF) and (OFF, ON) have positive probability. With g = 20 instead, eps > 0.

**Test B (single defector).** Same two EVs. Recommendation (ON, OFF), both charge. Load 14 kW, overload 4 kW, total penalty 16. EV 1 pays 16, EV 0 pays 0.

**Test C (Shapley).** Three EVs with ratings 7, 3.3 and 7 kW, C_max = 10. Recommendation (ON, OFF, OFF), all three charge. Load 17.3 kW, overload 7.3 kW, total penalty 53.29.

- EV 1 alone: overload 0.3, value 0.09.
- EV 2 alone: overload 4, value 16.
- Both: 53.29.
- EV 1 pays (0.09 + 53.29 - 16) / 2 = 18.69.
- EV 2 pays (16 + 53.29 - 0.09) / 2 = 34.60.
- EV 0 pays 0. The shares add up to 53.29.

## 6. Writing the report

### 6.1 Page budget and owners

The report is the review document completed to 8 pages, in the same `arxiv.sty` template.

| Section | Pages | Owner |
|---|---|---|
| Abstract, introduction | 1 | Sravan |
| Preliminaries: inputs, local model | 1 | Bhava Ram |
| Problem formulation: penalty game, mediator LP | 1 | Harshith |
| Methodology: algorithm for one slot, LP solution steps, existence condition | 1 | Harshith |
| Methodology: penalty and Shapley split, baselines, simulator | 1 | Koushik (penalty), Sravan (baselines) |
| Results: plots, tables, short discussion | 2 | each person writes up their own experiment |
| Contributions, limitations, references | 1 | Sravan assembles |

Results we plan to show. Each one is produced by one script and one CSV file.

| Item | Shows | Owner |
|---|---|---|
| Table: policy comparison | Overloaded slots, overload size, cost, share reaching target, for 5 policies and N = 4, 8, 12 | Sravan |
| Figure: load over the night | Total load against C_max for immediate charging and for the mediator, one scenario | Sravan |
| Figure: overload against C_max | How each policy degrades as the transformer gets tighter | Sravan |
| Figure: eps against lambda | The lambda at which the exact equilibrium appears | Harshith |
| Figure: mediator runtime against N | Growth with 2^N | Harshith |
| Table or figure: penalty rules | Shapley, equal and proportional split on the same defection events | Koushik |
| Figure: effect of defection rate | Overload and penalties as more EVs ignore the recommendation | Koushik |
| Table: infeasible targets | Cases with K < K_req and the unmet energy | Bhava Ram |

### 6.2 Contribution section

The guidelines ask for the assigned task and whether it was fulfilled. Use this table in the report and fill the last column honestly on 21 November. If something was only partly done, say which part.

| Member | Assigned task | Fulfilled |
|---|---|---|
| Busa Bhava Ram | Local EV model, scenario generator, tests | to fill |
| Muppala Harshith | Mediator LP, sampler, lambda and runtime experiments | to fill |
| Moram Raghavendra Sri Koushik | Penalty module, Shapley split, penalty comparison | to fill |
| Sura Sravan Kumar | Simulator, baselines, experiment scripts, figures, README | to fill |

The commit history should agree with this table, so each of us commits our own work from our own GitHub account.

### 6.3 Writing in our own words

Sir flagged three lines as AI text and asked for our own words. So the report text is written by us, section by section, by the person who built that part. Notes for when we write:

- Say what we did and what we got, with the number. "The LP was infeasible in 31 of 52 slots" is better than "the mediator often struggles".
- Do not describe the document itself. Lines such as "these limitations will be reported clearly" or "this is more than a scheduling demo" say nothing about the project. Sir quoted two of this kind.
- Do not leave words that only make sense as a reply to someone else. "Following the stated project rule" refers to a rule the reader has never seen.
- Avoid the "not just X, but Y" pattern, closing sentences that repeat the paragraph, and adjectives such as robust, comprehensive, seamless.
- Mix sentence lengths the way we normally write. Keep it plain. Simple English that is ours is safer than polished English that is not.
- State every assumption once, where it is used.
- Remove the "PRIME AI paper" page header from the template.
- Before submitting, each of us reads our section aloud to another member and answers "why" for every equation in it.

Questions each owner should be ready for:

- Bhava Ram: why is the terminal penalty needed, and what happens if M is too small?
- Harshith: why was the old LP infeasible, and what exactly does eps measure?
- Koushik: why does a lone defector pay the full penalty, and why Shapley over an equal split?
- Sravan: how is defection simulated, and what does each baseline know that the others do not?

## 7. Working on GitHub

Repository: `safe-ev-charging-mediator`. Koushik owns it and adds the other three as collaborators with write access.

### 7.1 One-time setup

- Everyone accepts the invite, clones the repository, and sets `git config user.name` and `user.email` to match their GitHub account. Otherwise commits will not show under the right person.
- Koushik turns on branch protection for `main`: pull request required, one approval required, no force push.
- Create one milestone, "Final submission 22 Nov", and four labels: `mdp`, `mediator`, `penalty`, `sim`. Two more labels help: `report` and `blocked`.

### 7.2 Issues

Every task in section 2 becomes an issue, assigned to its owner, with the module label and the milestone. Keep them small enough to close in a day. The issue list is how we see progress, so close an issue only when its pull request is merged.

If you are stuck for more than two hours, add the `blocked` label and write what you need. Do not wait for the evening call.

### 7.3 Branches

- `main` always runs. Tests pass on it at all times.
- Work happens on short branches named `<module>/<what>`, for example `mdp/solve-q`, `penalty/shapley-split`, `sim/baselines`, `report/results-section`.
- A branch lives one or two days. Long branches are where merge conflicts come from.
- Before starting work each day: `git checkout main`, `git pull`, then create or rebase your branch.

### 7.4 Commits

- Small commits with a message that says what changed: `mdp: add terminal penalty to solve_q`. Start with the module name.
- Do not commit `__pycache__`, virtual environments, or LaTeX build files. The `.gitignore` handles these.
- Only Sravan commits files under `results/` and `report/figures/`. They are generated files, and two people regenerating them causes conflicts that cannot be merged by hand.

### 7.5 Pull requests

- One pull request per issue. Write "Closes #12" in the description so the issue closes on merge.
- The description says what was added, how it was tested, and whether any shared name or signature changed.
- Reviewers: Bhava Ram and Harshith review each other. Koushik and Sravan review each other. A change to `common.py` needs approval from all three others.
- The reviewer pulls the branch and runs `pytest`. Reading the diff alone is not a review.
- Review within the same day. A pull request that waits blocks the person who depends on it.
- The author merges after approval, using "Squash and merge", and deletes the branch.

### 7.6 Conflicts

Each module has one owner and one file, so conflicts should be rare. When they happen, the person whose branch is behind rebases on `main` and fixes them. Never resolve a conflict in someone else's file without asking them.

### 7.7 Report on Overleaf

The LaTeX source lives in `report/`. If we edit on Overleaf, one person (Sravan) copies the Overleaf version into `report/` and commits it at the end of each day, so the repository always has the report that matches the code.

## 8. Final submission checklist

The evaluators will check that the numbers in the paper come from our code. So:

- Every figure and table in the report is written by `experiments/make_figures.py` from a CSV in `results/`. No number is typed into the report by hand. No plot is edited after it is generated.
- Every experiment script sets its seeds from `BASE_SEED`. Running it twice gives the same CSV.
- `python -m experiments.run_all` regenerates all results and figures from nothing. Note the total run time in the README.
- The README has: what the project does in a few lines, install steps, the one command to reproduce everything, a table mapping each figure and table in the paper to its script and CSV, screenshots of the main results, the team and who did what.
- Screenshots go in `docs/screenshots/` and are taken from the real output: the terminal after `pytest`, the terminal after `run_all`, and the main plots.
- Fresh-clone check on 21 November, done by Bhava Ram on a machine that has not run the experiments: clone, create a new environment, install from `requirements.txt`, run `pytest`, run `run_all`, and compare every number in the PDF with the regenerated output. Any mismatch is fixed in the code or the report before submission.
- After the final commit, tag it: `git tag v1.0-submission` and push the tag. Put the tag's commit hash in the report next to the repository link.
- The repository must be visible to the evaluators. If it is private, make it public or add them before submitting.
- The report is 8 pages. Check after the final figures are in, since figures change the page count.
- Submit the form on the 22nd with a few hours to spare.
