"""Two charts that make a Monte Carlo run readable.

The fan chart shows the spread of balance paths over time as shaded percentile
bands, which is far more honest than a single average line. The histogram shows
the distribution of final balances, with the target and break-even marked.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_fan_chart(
    paths: np.ndarray,
    steps_per_year: int,
    title: str = "Range of outcomes over time",
):
    """Shaded percentile bands of the balance paths through time."""
    n_steps = paths.shape[1] - 1
    years = np.arange(n_steps + 1) / steps_per_year

    p5, p25, p50, p75, p95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.fill_between(years, p5, p95, alpha=0.20, color="tab:blue", label="5 to 95%")
    ax.fill_between(years, p25, p75, alpha=0.35, color="tab:blue", label="25 to 75%")
    ax.plot(years, p50, color="tab:blue", linewidth=2, label="Median")
    ax.set_title(title)
    ax.set_xlabel("Years")
    ax.set_ylabel("Balance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_terminal_histogram(
    terminal_values: np.ndarray,
    target: float | None = None,
    break_even: float | None = None,
    title: str = "Distribution of final balances",
):
    """Histogram of final balances, with the target and break-even lines."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(terminal_values, bins=60, color="tab:blue", alpha=0.7)
    if target is not None:
        ax.axvline(target, color="tab:green", linestyle="--", linewidth=2, label="Target")
    if break_even is not None:
        ax.axvline(break_even, color="tab:red", linestyle=":", linewidth=2, label="Total paid in")
    ax.set_title(title)
    ax.set_xlabel("Final balance")
    ax.set_ylabel("Number of simulated futures")
    if target is not None or break_even is not None:
        ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def save_figure(fig, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    return path
