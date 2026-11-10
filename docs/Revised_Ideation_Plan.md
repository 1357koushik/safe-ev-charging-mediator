# Revised Ideation Plan: Neighbourhood EV Charging

Group 25: Busa Bhava Ram (24110083), Muppala Harshith (24110213), Sura Sravan Kumar (24110362), Moram Raghavendra Sri Koushik (24110210)

Status: working draft for the team. The formulation in section 3 has to be shown to the course instructor before we build on it, because sir asked us to clear it up with him.

Final submission is Sunday 22 November. The final report is this document completed to 8 pages in the same LaTeX template, with a Methodology section, a Results section and a Contribution section, plus a repository link. The evaluators will check that the results in the paper come from our code. The LaTeX version is in `report/main.tex` in the repository.

## 1. What sir pointed out

There were two remarks.

1. In our setup each car's utility depends only on its own action. So the equilibrium constraints of the mediator often have no solution, and this happens exactly when cars have to share the transformer limit.
2. Some lines read like unedited AI text. He wants the report in our own words and may ask us to explain the formulation.

Both are correct. Section 2 shows why the first one is a real bug and not a wording problem.

## 2. Why the old formulation breaks

Take two EVs, each with a 7 kW charger, on a 10 kW transformer. Both want to charge now, so for both Q(on) > Q(off).

Our old constraint (15) for EV 1 when it is told to wait was

    P(EV 1 is told OFF) * [Q1(off) - Q1(on)] >= 0

The bracket is negative, so the only way to satisfy it is P(EV 1 is told OFF) = 0. The same holds for EV 2. So the mediator must tell both to charge, which is 14 kW, and constraint (14) forbids that profile. The LP is infeasible.

The reason is simple. If my payoff does not depend on what others do, then knowing my recommendation tells me nothing useful, and I just play my better action. The "equilibrium" is every car playing its own best action, and the mediator has no room to coordinate anything. Nothing in the old model gave a car a reason to wait.

We found two smaller errors while checking this:

- The reward in equation (8) is only the electricity cost. The terminal penalty M is mentioned in the text but never enters any equation. With that reward, waiting is always at least as good as charging, so no EV would ever charge.
- The page header still says "PRIME AI paper" from the template.

## 3. Proposed fix

### 3.1 Idea

The overload penalty already exists in our design, but only as something computed after the slot. We move it inside the game. An EV that is told to wait and charges anyway pays for the overload it causes. Now its payoff depends on how much load the others are putting on the transformer, which is the dependence that was missing.

### 3.2 Local MDP (corrected)

State and action stay the same: s = (SoC, K), action is 0 or the rated power P_i, slot length 0.25 h.

Reward for a normal slot: minus the electricity cost, as before.

New terminal reward when K reaches 0:

    r_terminal = -M * (unmet energy in kWh)
    unmet energy = max(0, (SoC_target - SoC) * E_cap)

M must be larger than the highest price divided by the efficiency, otherwise skipping the charge is cheaper than charging. We use gamma = 1 since the horizon is one night.

The transitions are deterministic, so this is really a small dynamic program. We still call it the local MDP because the course uses that term, but we should say this plainly if asked.

The EV sends (Q_off, Q_on) to the mediator as before. Define the gain from charging now as g_i = Q_on - Q_off.

### 3.3 Payoff with the penalty

Let a* be the recommended profile and a the executed one. Let L(a) be the total load in kW.

    overload(a) = max(0, L(a) - C_max)
    total penalty = lambda * overload^2        (lambda in Rs per kW^2)

Only defectors pay, where a defector is an EV told OFF that charges. If several defect in the same slot, the penalty is split by the Shapley value exactly as in the old section 4.2. If one EV defects alone from a safe recommendation, it pays the full amount:

    pen_i(a_-i) = lambda * max(0, L(a_-i) + P_i - C_max)^2

So EV i's payoff in a slot is its Q-value for the action it takes, minus its penalty share.

### 3.4 Mediator LP

Variables: pi(a) for every joint profile a, 2^N of them. Unsafe profiles (load above C_max) are fixed to zero as before.

Obedience constraints, two per EV:

- Told ON, considers waiting. No penalty is involved, so this is the old constraint:

      P(i told ON) * g_i >= -eps

- Told OFF, considers charging:

      sum over a_-i of  pi(OFF, a_-i) * [ pen_i(a_-i) - g_i ]  >=  -eps

The second one is where the others now matter. An EV that is told to wait when the transformer is nearly full knows that charging would cause an overload it has to pay for.

eps >= 0 is a slack variable. We solve in two steps: first minimise eps, then fix it and maximise the expected total utility. If eps comes out as 0 we have an exact correlated equilibrium on safe profiles. If eps > 0, lambda was too small to make waiting worthwhile, and we report eps as the incentive gap. With the slack the LP always has a solution, because "everyone waits" is safe.

### 3.5 Check on the two-EV example

Recommend (ON, OFF) and (OFF, ON) with probability 1/2 each. An EV told OFF knows the other is charging 7 kW. If it charges too, the overload is 4 kW and it pays 16 * lambda. It obeys whenever 16 * lambda >= g. The LP that had no solution now has one.

### 3.6 When an exact solution exists

Claim to verify and include: let W be the EVs with g_i > 0. Pick any safe subset of W to which no other member of W can be added without overload. Recommend ON to that subset and OFF to the rest. Each left-out EV j would cause an overload d_j > 0, and it obeys if lambda * d_j^2 >= g_j. So an exact safe equilibrium exists when

    lambda >= max over left-out j of  g_j / d_j^2

One weak spot: if d_j is very small (say 0.3 kW), the needed lambda gets very large. We should look at this in experiments. A fixed charge for any overload on top of the quadratic term would fix it, but we add that only if the results need it.

### 3.7 Points to ask the instructor

1. Our payoff depends on the recommendation as well as the action, since only defectors pay. Is it acceptable to call the result a correlated equilibrium of the game with this penalty rule, or does he want a standard game where payoffs depend on actions only?
2. If he wants the standard form, the alternative is a congestion game: everyone who charges in an overloaded slot shares the penalty. That is textbook CE, but it also charges cars that obeyed, which we wanted to avoid.
3. Is the eps slack fine as a fallback, or should infeasible slots be reported as failures?

## 4. Methodology (what the final report must describe in detail)

1. Local model: backward induction for (Q_off, Q_on), with the terminal penalty.
2. Mediator: enumerate the 2^N profiles, drop the unsafe ones, solve the LP twice (minimise eps, then maximise total utility), sample one profile.
3. Penalty: find the defectors, compute the overload, split the penalty by the Shapley value.
4. Simulator: the slot loop that connects the three, and the defection model.
5. Baselines: immediate, price-only, urgency, independent local MDP.

Section 3 above is the formulation. The Methodology section explains how each part is computed, with the algorithm for one slot.

## 5. Experiments and results

Same as before, with these changes.

- N in {4, 8, 12}, ratings 3.3 / 7.0 / 11.0 kW, 52 slots from 18:00 to 07:00.
- Policies: charge immediately, price-only, urgency heuristic, independent local MDP, CE mediator.
- Penalty rules: Shapley, equal split, split proportional to charger rating.
- New things to measure: share of slots with eps = 0, the size of eps otherwise, and the smallest lambda at which defecting stops paying off. The last one is also how we compare the three penalty rules, since "fairness" on its own was too vague in the first draft. We also check that an obedient EV never pays and that shares add up to the total.
- Old metrics stay: overloaded slots, overload magnitude, electricity cost, share of EVs reaching target, unmet energy, mediator runtime.

Planned plots and tables for the Results section:

- Table comparing the five policies for N = 4, 8, 12.
- Total load over the night against C_max, immediate charging and mediator side by side.
- Overload against C_max for each policy.
- eps against lambda, to show where the exact equilibrium appears.
- Mediator runtime against N.
- The three penalty rules on the same defection events.
- Overload and penalties against the defection rate.

Every one of these is generated by a script in the repository from a fixed seed. Nothing is typed in by hand, because the results will be verified against the code. Each plot gets two or three sentences of discussion that state the number we read off it.

## 6. Contribution section

The final report lists the task assigned to each member and whether it was fulfilled. The table is in section 6.2 of the work division plan.

## 7. Limitations

- EVs report Q-values truthfully. Misreporting is not modelled.
- The Q-values come from a local plan that assumes the EV can charge whenever it wants. They do not account for future OFF recommendations or future penalties.
- The mediator enumerates 2^N profiles, so N = 12 is close to the practical limit.
- No variable-rate charging, voltage or three-phase effects, battery ageing, or real tariff rules.

## 8. References

Same three as before (Aumann 1974, Shapley 1953, Shojaeighadikolaei and Hashemi 2023).
