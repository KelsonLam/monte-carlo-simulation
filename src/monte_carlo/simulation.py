"""Roll generated returns forward into balance paths.

Each path starts at the initial investment. At every step the balance grows by
that step's return and then the periodic contribution is added. Contributions
are spread evenly across the year, so an annual contribution of 12,000 with
monthly steps adds 1,000 each month.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlanConfig:
    initial_investment: float = 50_000.0
    annual_contribution: float = 12_000.0
    horizon_years: int = 20
    steps_per_year: int = 12


@dataclass
class SimulationResult:
    paths: np.ndarray            # shape (n_paths, n_steps + 1), includes start
    total_contributed: float     # initial plus every contribution added
    plan: PlanConfig

    @property
    def terminal_values(self) -> np.ndarray:
        return self.paths[:, -1]


def simulate(
    model,
    plan: PlanConfig,
    n_paths: int,
    rng: np.random.Generator,
) -> SimulationResult:
    """Generate balance paths for the plan using the given return model."""
    n_steps = plan.horizon_years * plan.steps_per_year
    if n_steps <= 0:
        raise ValueError("horizon_years and steps_per_year must be positive.")

    returns = model.generate(n_paths, n_steps, rng)
    contribution_per_step = plan.annual_contribution / plan.steps_per_year

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = plan.initial_investment
    for t in range(n_steps):
        paths[:, t + 1] = paths[:, t] * (1.0 + returns[:, t]) + contribution_per_step

    total_contributed = (
        plan.initial_investment + contribution_per_step * n_steps
    )
    return SimulationResult(
        paths=paths, total_contributed=total_contributed, plan=plan
    )
