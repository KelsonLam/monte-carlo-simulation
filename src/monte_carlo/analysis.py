"""Turning a cloud of simulated outcomes into answers.

The point of a Monte Carlo run is not the average. The average hides the spread,
and the spread is the whole story. These helpers pull out the percentiles, the
chance of hitting the target, the chance of falling short of what was paid in,
and the downside tail (value at risk and conditional value at risk).
"""

from __future__ import annotations

import numpy as np


def percentiles(
    terminal_values: np.ndarray, levels=(5, 25, 50, 75, 95)
) -> dict[int, float]:
    """Percentiles of the final balance, keyed by percentile level."""
    values = np.percentile(terminal_values, levels)
    return {int(level): float(v) for level, v in zip(levels, values)}


def probability_of_target(
    terminal_values: np.ndarray, target: float
) -> float:
    """Share of simulated futures that finish at or above the target."""
    return float(np.mean(terminal_values >= target))


def probability_below(
    terminal_values: np.ndarray, threshold: float
) -> float:
    """Share of simulated futures that finish below a threshold.

    Passing the total amount contributed answers "how often do I end up with
    less than I put in?"
    """
    return float(np.mean(terminal_values < threshold))


def value_at_risk(terminal_values: np.ndarray, alpha: float = 0.05) -> float:
    """The terminal balance at the alpha quantile (a low-end outcome).

    With alpha = 0.05 this is the 5th percentile: 5% of futures end below it.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    return float(np.percentile(terminal_values, alpha * 100.0))


def conditional_value_at_risk(
    terminal_values: np.ndarray, alpha: float = 0.05
) -> float:
    """Average terminal balance across the worst alpha share of futures."""
    threshold = value_at_risk(terminal_values, alpha)
    tail = terminal_values[terminal_values <= threshold]
    if tail.size == 0:
        return threshold
    return float(np.mean(tail))


def summarize(
    terminal_values: np.ndarray, target: float, total_contributed: float
) -> dict[str, float]:
    """Bundle the headline answers into one dictionary."""
    pcts = percentiles(terminal_values)
    return {
        "Median outcome": pcts[50],
        "5th percentile": pcts[5],
        "95th percentile": pcts[95],
        "Mean outcome": float(np.mean(terminal_values)),
        "Probability of hitting target": probability_of_target(
            terminal_values, target
        ),
        "Probability of ending below contributions": probability_below(
            terminal_values, total_contributed
        ),
        "Value at risk (5%)": value_at_risk(terminal_values, 0.05),
        "Conditional VaR (5%)": conditional_value_at_risk(terminal_values, 0.05),
    }


def format_summary(stats: dict[str, float]) -> str:
    """Render the summary as aligned text. Money is shown with thousands commas."""
    prob_keys = {
        "Probability of hitting target",
        "Probability of ending below contributions",
    }
    width = max(len(k) for k in stats)
    lines = []
    for key, value in stats.items():
        if key in prob_keys:
            shown = f"{value * 100:,.1f}%"
        else:
            shown = f"{value:,.0f}"
        lines.append(f"{key:<{width}}  {shown}")
    return "\n".join(lines)
