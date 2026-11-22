"""Figures and the table for the penalty part of the report.

Owner: Moram Raghavendra Sri Koushik.

Reads only the CSV files in results/ and writes to report/figures/. The
functions are meant to be called from experiments/make_figures.py; running
this file directly does the same thing for the penalty plots alone:

    python -m experiments.figures_penalty
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path("results")
FIGURES = Path("report/figures")

# Categorical slots 1-3 of the shared palette, checked for colour-blind
# separation. Aqua is below 3:1 contrast on white, so every plot also has a
# legend and the same numbers are in the table.
RULE_COLOR = {"shapley": "#2a78d6", "equal": "#eb6834", "proportional": "#1baf7a"}
RULE_LABEL = {"shapley": "Shapley", "equal": "Equal split", "proportional": "Proportional"}
# One blue ramp for N, light to dark, because N is ordered.
N_COLOR = {4: "#6da7ec", 8: "#2a78d6", 12: "#104281"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8, width=0.8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _save(fig, name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


MODE_TITLE = {"full": "Full recommendation", "spare": "Recommendation with spare capacity"}


def plot_rs_per_kw(summary: pd.DataFrame, out_dir=FIGURES):
    """Average penalty per kW of charger rating, by rating and rule, with one
    panel per recommendation mode. Both panels share the y axis."""
    modes = [m for m in MODE_TITLE if m in set(summary.rec_mode)]
    cols = sorted((c for c in summary.columns if c.startswith("rs_per_kw_")),
                  key=lambda c: float(c.removeprefix("rs_per_kw_")))
    fig, axes = plt.subplots(1, len(modes), figsize=(6.4, 2.6), sharey=True,
                             squeeze=False)
    width = 0.26
    for ax, mode in zip(axes[0], modes):
        s = summary[summary.rec_mode == mode].set_index("rule")
        for k, rule in enumerate(["shapley", "proportional", "equal"]):
            xs = [i + (k - 1) * (width + 0.02) for i in range(len(cols))]
            # A rating class with no defectors in this mode gets no bar. A
            # zero-height bar would read as "pays nothing", which is wrong.
            ys = [s.loc[rule, c] for c in cols]
            ax.bar(xs, ys, width=width, color=RULE_COLOR[rule],
                   label=RULE_LABEL[rule], edgecolor="white", linewidth=1.0)
        ax.set_xticks(range(len(cols)),
                      [f"{c.removeprefix('rs_per_kw_')} kW" for c in cols])
        ax.set_title(MODE_TITLE[mode], fontsize=9, color=INK)
        ax.set_xlabel("Charger rating of the defector", fontsize=9)
        _style(ax)
    axes[0][0].set_ylabel("Penalty per kW (Rs)", fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.88))
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 1.0), labelcolor=INK)
    _save(fig, "penalty_rs_per_kw", out_dir)


def plot_defection_rate(df: pd.DataFrame, out_dir=FIGURES):
    """Two panels: how often the limit is broken, and the mean penalty."""
    d = df[df.rule == "shapley"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    panels = [("pct_slots_overloaded", "Slots overloaded (%)"),
              ("mean_penalty_rs", "Mean penalty per slot (Rs)")]
    for p_idx, (ax, (col, ylabel)) in enumerate(zip(axes, panels)):
        for n_ev, g in d.groupby("n_ev"):
            g = g.sort_values("defect_rate")
            ax.plot(g.defect_rate, g[col], color=N_COLOR[n_ev], linewidth=1.5,
                    marker="o", markersize=3.5, label=f"N = {n_ev}")
            if p_idx == 0:
                continue      # left panel has the legend; its line ends crowd
            ax.annotate(f"N = {n_ev}", (g.defect_rate.iloc[-1], g[col].iloc[-1]),
                        xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=7, color=INK_2)
        ax.set_xlabel("Defection rate", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(0, 0.58)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left", labelcolor=INK)
    fig.tight_layout(w_pad=2.0)
    _save(fig, "defection_rate", out_dir)


def write_rules_table(summary: pd.DataFrame, out_dir=FIGURES):
    """LaTeX table body for the report, so no number is typed by hand."""
    order = ["shapley", "proportional", "equal"]
    mode_label = {"full": "Full", "spare": "Spare capacity"}

    def cell(v):
        return "--" if pd.isna(v) else f"{v:.1f}"

    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r" & & Gap to & Events with a & Events with a & \multicolumn{3}{c}{Rs per kW} \\",
        r"\cmidrule(l){6-8}",
        r"Recommendation & Rule & Shapley (\%) & defector below $c(\{i\})$ (\%)"
        r" & group subsidised (\%) & 3.3 kW & 7 kW & 11 kW \\",
    ]
    for mode in [m for m in MODE_TITLE if m in set(summary.rec_mode)]:
        s = summary[summary.rec_mode == mode].set_index("rule").loc[order]
        lines.append(r"\midrule")
        for k, (rule, r) in enumerate(s.iterrows()):
            first = f"{mode_label[mode]} ($n={int(r.events)}$)" if k == 0 else ""
            lines.append(
                f"{first} & {RULE_LABEL[rule]} & {cell(r.mean_gap_vs_shapley_pct)} & "
                f"{cell(r.pct_events_someone_below_lone)} & "
                f"{cell(r.pct_events_group_subsidised)} & "
                f"{cell(r.get('rs_per_kw_3.3'))} & {cell(r.get('rs_per_kw_7'))} & "
                f"{cell(r.get('rs_per_kw_11'))} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "penalty_rules_table.tex").write_text("\n".join(lines) + "\n")


def make_all(results=RESULTS, out_dir=FIGURES):
    summary = pd.read_csv(results / "penalty_rules_summary.csv")
    plot_rs_per_kw(summary, out_dir)
    write_rules_table(summary, out_dir)
    plot_defection_rate(pd.read_csv(results / "defection_rate.csv"), out_dir)


def main():
    """Entry point used by experiments/run_all.py."""
    make_all()


if __name__ == "__main__":
    main()
