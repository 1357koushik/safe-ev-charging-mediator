"""Make all figures and tables for Sravan's experiments.

Reads from results/ and writes to report/figures/.
Every call overwrites existing files so the figures always match the CSVs.

Usage:
    python -m experiments.make_figures

Produces:
    report/figures/policy_table.tex      -- policy comparison table (LaTeX)
    report/figures/policy_table_N4.tex
    report/figures/policy_table_N8.tex
    report/figures/policy_table_N12.tex
    report/figures/load_profile.pdf      -- load over one night
    report/figures/load_profile.png
    report/figures/overload_vs_cmax.pdf  -- overload vs C_max
    report/figures/overload_vs_cmax.png
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
FIGURES = os.path.join(REPO, "report", "figures")

POLICY_LABELS = {
    "immediate":       "Immediate",
    "price_only":      "Price-only",
    "urgency":         "Urgency",
    "independent_mdp": "Indep. MDP",
    "ce_mediator":     "CE Mediator",
}


def _savefig(fig, stem: str) -> None:
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {stem}.pdf/png")


# ------------------------------------------------------------------ helpers --

def _slot_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (policy, run_id, slot) with slot-level columns de-duplicated."""
    slot_cols = ["run_id", "policy", "seed", "n_ev", "c_max_kw", "lam",
                 "defect_rate", "slot", "load_kw", "overload_kw", "eps"]
    return df.drop_duplicates(subset=["policy", "run_id", "slot", "n_ev", "c_max_kw"])[slot_cols]


def _ev_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-EV outcome at departure: last row for each (run_id, ev_id)."""
    # Last connected slot for each EV in each run
    connected = df[df["k_left"] > 0].copy()
    last = connected.groupby(["run_id", "ev_id"]).last().reset_index()
    return last


# ============================================================= policy table ==

def make_policy_table(df: pd.DataFrame, n_ev: int) -> str:
    """Return a LaTeX tabular body for one N value."""
    sub = df[df["n_ev"] == n_ev].copy()

    # Slot-level: overloaded slots and total overload
    slots = _slot_summary(sub)
    slot_agg = slots.groupby(["policy", "seed"]).agg(
        overloaded_slots=("overload_kw", lambda x: (x > 0).sum()),
        total_overload=("overload_kw", "sum"),
    ).reset_index()

    # EV-level: fraction reaching target SoC
    # soc_before in last row approximates final SoC (after the last action it won't update)
    # Use energy balance: total energy_cost_rs / (p_rated * DT_H * price) for SoC
    # Simpler: flag EV as "reached target" if pen_rs == 0 at departure slot (k_left becomes 0 next)
    # We'll use: soc_before at the slot where k_left==1 gives us the SoC entering last slot
    # Best proxy: last soc_before + on-action increment vs soc_target stored in EVParams.
    # We don't have soc_target in log — use pen_rs==0 at departure as proxy for "reached target"
    # Actually: an EV that did NOT reach target has a large terminal pen_rs at k_left==1 slot
    # Cleaner: sum pen_rs per EV per run; if 0, it reached target (no unmet energy)
    ev_agg = sub.groupby(["policy", "seed", "ev_id"]).agg(
        total_pen=("pen_rs", "sum"),
        total_ecost=("energy_cost_rs", "sum"),
    ).reset_index()
    # Compute fraction of EVs with no terminal miss-penalty as a proxy for "target met"
    # (penalty from unmet energy appears as pen_rs in the last slot only for the defection penalty)
    # Better: EVs with terminal unmet energy get q_on - q_off blowup near departure — 
    # use total energy charged: act=1 slots give energy to SoC.
    # Simplest reliable metric from the log: electricity cost > 0 means it charged.
    # For "reached target" we need soc >= soc_target. We store soc_before each slot.
    # Last row for each EV gives soc_before of their last active slot; add one ON action if act=1.
    ev_last = sub[sub["k_left"] == 1].copy()  # last active slot
    if ev_last.empty:
        ev_last = sub.groupby(["run_id", "ev_id"]).last().reset_index()

    # Compute cost and overload stats per policy per seed, then average over seeds
    slot_mean = slot_agg.groupby("policy").agg(
        overloaded_slots=("overloaded_slots", "mean"),
        total_overload=("total_overload", "mean"),
    ).reset_index()
    ecost_mean = ev_agg.groupby(["policy", "seed"]).agg(
        ecost=("total_ecost", "sum")).reset_index().groupby("policy").agg(
        ecost=("ecost", "mean")).reset_index()

    # Build table rows
    policy_order = ["immediate", "price_only", "urgency", "independent_mdp", "ce_mediator"]
    lines = []
    for pol in policy_order:
        label = POLICY_LABELS.get(pol, pol)
        row_s = slot_mean[slot_mean["policy"] == pol]
        row_e = ecost_mean[ecost_mean["policy"] == pol]
        ov_slots = f"{row_s['overloaded_slots'].values[0]:.1f}" if len(row_s) else "--"
        ov_total = f"{row_s['total_overload'].values[0]:.1f}" if len(row_s) else "--"
        ecost = f"{row_e['ecost'].values[0]:.1f}" if len(row_e) else "--"
        lines.append(f"        {label} & {ov_slots} & {ov_total} & {ecost} \\\\")

    return "\n".join(lines)


def save_policy_table(df: pd.DataFrame) -> None:
    os.makedirs(FIGURES, exist_ok=True)
    for n_ev in [4, 8, 12]:
        body = make_policy_table(df, n_ev)
        header = (
            "\\begin{tabular}{lrrr}\n"
            "\\toprule\n"
            f"Policy ($N={n_ev}$) & Overloaded slots & Total overload (kW) & Electricity cost (Rs) \\\\\n"
            "\\midrule\n"
        )
        footer = "\n\\bottomrule\n\\end{tabular}\n"
        tex = header + body + footer
        path = os.path.join(FIGURES, f"policy_table_N{n_ev}.tex")
        with open(path, "w") as f:
            f.write(tex)
        print(f"  Saved policy_table_N{n_ev}.tex")

    # Combined table with N as a group separator
    with open(os.path.join(FIGURES, "policy_table.tex"), "w") as f:
        f.write("% Auto-generated by make_figures.py\n")
        for n_ev in [4, 8, 12]:
            f.write(f"% N = {n_ev}\n")
            f.write(make_policy_table(df, n_ev))
            f.write("\n")
    print("  Saved policy_table.tex")


# ============================================================= load profile ==

def make_load_profile(df: pd.DataFrame) -> None:
    """Plot load over one night for immediate vs mediator (N=8, one seed)."""
    seed = df["seed"].min()
    n_ev = 8

    fig, ax = plt.subplots(figsize=(7, 3.5))
    dt = 0.25  # hours per slot
    times = np.arange(52) * dt  # hours from 18:00

    for pol, color, ls in [
        ("immediate", "#d62728", "-"),
        ("ce_mediator", "#1f77b4", "--"),
    ]:
        sub = df[(df["policy"] == pol) & (df["n_ev"] == n_ev) & (df["seed"] == seed)]
        if sub.empty:
            continue
        slot_data = sub.drop_duplicates("slot").set_index("slot")["load_kw"].reindex(range(52), fill_value=0)
        c_max = sub["c_max_kw"].iloc[0]
        ax.step(times, slot_data.values, where="post", label=POLICY_LABELS[pol],
                color=color, linestyle=ls, linewidth=1.8)

    # Draw C_max line
    sub0 = df[(df["policy"] == "ce_mediator") & (df["n_ev"] == n_ev) & (df["seed"] == seed)]
    if not sub0.empty:
        c_max = sub0["c_max_kw"].iloc[0]
        ax.axhline(c_max, color="black", linestyle=":", linewidth=1.2, label=r"$C_{\max}$")

    # x-axis: show wall-clock time labels
    tick_slots = [0, 8, 16, 24, 32, 40, 48, 51]
    tick_times = [t * dt for t in tick_slots]
    tick_labels = ["18:00", "20:00", "22:00", "00:00", "02:00", "04:00", "06:00", "06:45"]
    ax.set_xticks(tick_times)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Time of night")
    ax.set_ylabel("Total load (kW)")
    ax.set_title(f"Load profile — N = {n_ev} EVs, seed {seed}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "load_profile")


# ============================================================= cmax sweep ==

def make_overload_vs_cmax(df: pd.DataFrame) -> None:
    """Line plot: mean overloaded slots per night vs C_max, one line per policy."""
    slots = _slot_summary(df)
    agg = slots.groupby(["policy", "c_max_kw", "seed"]).agg(
        overloaded=("overload_kw", lambda x: (x > 0).sum())
    ).reset_index()
    mean_agg = agg.groupby(["policy", "c_max_kw"])["overloaded"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#1f77b4"]
    policy_order = ["immediate", "price_only", "urgency", "independent_mdp", "ce_mediator"]
    markers = ["o", "s", "^", "D", "v"]

    for pol, color, marker in zip(policy_order, colors, markers):
        sub = mean_agg[mean_agg["policy"] == pol]
        if sub.empty:
            continue
        sub = sub.sort_values("c_max_kw")
        ax.plot(sub["c_max_kw"], sub["overloaded"], label=POLICY_LABELS[pol],
                color=color, marker=marker, linewidth=1.8, markersize=5)

    ax.set_xlabel(r"Transformer limit $C_{\max}$ (kW)")
    ax.set_ylabel("Mean overloaded slots per night")
    ax.set_title("Effect of transformer limit on overloading")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "overload_vs_cmax")


# ================================================================== main ==

def main() -> None:
    os.makedirs(FIGURES, exist_ok=True)

    # --- policy comparison ---
    pc_path = os.path.join(RESULTS, "policy_comparison.csv")
    if os.path.exists(pc_path):
        print("Loading policy_comparison.csv ...")
        pc = pd.read_csv(pc_path)
        save_policy_table(pc)
        make_load_profile(pc)
    else:
        print(f"WARNING: {pc_path} not found — skipping policy table and load profile")

    # --- cmax sweep ---
    cs_path = os.path.join(RESULTS, "cmax_sweep.csv")
    if os.path.exists(cs_path):
        print("Loading cmax_sweep.csv ...")
        cs = pd.read_csv(cs_path)
        make_overload_vs_cmax(cs)
    else:
        print(f"WARNING: {cs_path} not found — skipping overload_vs_cmax figure")

    print("make_figures.py complete.")


if __name__ == "__main__":
    main()
