"""Figures for the mediator part of the report.

Owner: Muppala Harshith.

Reads the CSV files in results/ and writes to report/figures/:
- mediator_eps_lambda.pdf / .png: Incentive gap (epsilon) and exact CE rate vs lambda.
- mediator_runtime.pdf / .png: Mediator solve time scaling vs N.

Can be run directly from the repository root:
    python -m experiments.figures_mediator
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path("results")
FIGURES = Path("report/figures")

N_COLOR = {
    2: "#a8cbf5",
    4: "#6da7ec",
    6: "#4b8fe0",
    8: "#2a78d6",
    10: "#1d5db0",
    12: "#104281",
}
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


def plot_eps_vs_lambda(df_lambda: pd.DataFrame, out_dir=FIGURES):
    """Plot mean epsilon and fraction of exact CE slots against lambda."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.7))

    for n_ev in sorted(df_lambda["n_ev"].unique()):
        sub = df_lambda[df_lambda["n_ev"] == n_ev].sort_values("lam")
        color = N_COLOR.get(n_ev, "#2a78d6")

        # Left panel: Mean epsilon vs lambda
        ax1.plot(
            sub["lam"],
            sub["mean_eps"],
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            color=color,
            label=f"N = {n_ev}",
        )

        # Right panel: Percentage of exact CE slots vs lambda
        ax2.plot(
            sub["lam"],
            sub["pct_exact_ce"],
            marker="s",
            markersize=3.5,
            linewidth=1.2,
            color=color,
            label=f"N = {n_ev}",
        )

    _style(ax1)
    _style(ax2)

    ax1.set_xlabel(r"Penalty weight $\lambda$ (Rs / kW$^2$)", fontsize=9)
    ax1.set_ylabel(r"Mean incentive gap $\varepsilon^\star$ (Rs)", fontsize=9)
    ax1.set_title(r"(a) Slack $\varepsilon^\star$ vs $\lambda$", fontsize=9, color=INK)
    ax1.legend(frameon=False, fontsize=8)

    ax2.set_xlabel(r"Penalty weight $\lambda$ (Rs / kW$^2$)", fontsize=9)
    ax2.set_ylabel(r"Slots with exact CE $\varepsilon^\star=0$ (%)", fontsize=9)
    ax2.set_title(r"(b) Exact equilibrium frequency", fontsize=9, color=INK)
    ax2.set_ylim(-5, 105)
    ax2.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    _save(fig, "mediator_eps_lambda", out_dir)


def plot_runtime_vs_n(df_runtime: pd.DataFrame, out_dir=FIGURES):
    """Plot solver runtime and profile count against number of EVs N."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.7))

    summary = (
        df_runtime.groupby("n_ev")
        .agg(
            n_profiles=("n_profiles", "first"),
            mean_safe=("n_safe_profiles", "mean"),
            median_ms=("runtime_ms", "median"),
            min_ms=("runtime_ms", "min"),
            max_ms=("runtime_ms", "max"),
        )
        .reset_index()
    )

    ns = summary["n_ev"].to_numpy()
    meds = summary["median_ms"].to_numpy()
    mins = summary["min_ms"].to_numpy()
    maxs = summary["max_ms"].to_numpy()

    # Left panel: runtime (log-scale y)
    ax1.semilogy(
        ns,
        meds,
        marker="o",
        markersize=4.0,
        linewidth=1.4,
        color="#2a78d6",
        label="Observed median",
    )
    ax1.fill_between(ns, mins, maxs, color="#2a78d6", alpha=0.15, label="Min–Max range")

    _style(ax1)
    ax1.set_xlabel("Number of EVs ($N$)", fontsize=9)
    ax1.set_ylabel("Solver runtime (ms, log scale)", fontsize=9)
    ax1.set_title(r"(a) Mediator solve time vs $N$", fontsize=9, color=INK)
    ax1.set_xticks(ns)
    ax1.legend(frameon=False, fontsize=8)

    # Right panel: profile explosion and safe profile pruning
    profiles = summary["n_profiles"].to_numpy()
    safe_profiles = summary["mean_safe"].to_numpy()

    ax2.semilogy(
        ns,
        profiles,
        marker="^",
        markersize=4.0,
        linewidth=1.2,
        color="#eb6834",
        linestyle="--",
        label=r"Total profiles ($2^N$)",
    )
    ax2.semilogy(
        ns,
        safe_profiles,
        marker="s",
        markersize=4.0,
        linewidth=1.2,
        color="#1baf7a",
        label="Mean safe profiles",
    )

    _style(ax2)
    ax2.set_xlabel("Number of EVs ($N$)", fontsize=9)
    ax2.set_ylabel("Profile count (log scale)", fontsize=9)
    ax2.set_title(r"(b) Profile space reduction", fontsize=9, color=INK)
    ax2.set_xticks(ns)
    ax2.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    _save(fig, "mediator_runtime", out_dir)


def make_all(results=RESULTS, out_dir=FIGURES):
    """Generate all mediator figures from result CSVs."""
    lambda_csv = results / "mediator_lambda.csv"
    if lambda_csv.exists():
        df_lambda = pd.read_csv(lambda_csv)
        plot_eps_vs_lambda(df_lambda, out_dir=out_dir)

    runtime_csv = results / "mediator_runtime.csv"
    if runtime_csv.exists():
        df_runtime = pd.read_csv(runtime_csv)
        plot_runtime_vs_n(df_runtime, out_dir=out_dir)


def main():
    make_all()
    print("Mediator figures written to report/figures/")


if __name__ == "__main__":
    main()
