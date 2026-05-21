"""Tests for the return models, the simulator, and the analysis helpers.

These are deterministic where possible (fixed seeds, analytic checks) and use no
network, since the GBM model needs no data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from monte_carlo.models import GBMModel, BootstrapModel
from monte_carlo.simulation import PlanConfig, simulate
from monte_carlo import analysis


def test_gbm_zero_vol_is_deterministic():
    model = GBMModel(annual_return=0.12, annual_volatility=0.0, steps_per_year=12)
    rng = np.random.default_rng(0)
    returns = model.generate(5, 12, rng)
    expected = (1.12) ** (1 / 12) - 1
    assert np.allclose(returns, expected)


def test_gbm_mean_growth_matches_target():
    # With no contributions, the average terminal value of a unit invested for
    # ten years should land near (1 + annual_return) ** 10.
    model = GBMModel(annual_return=0.07, annual_volatility=0.15, steps_per_year=12)
    plan = PlanConfig(initial_investment=1.0, annual_contribution=0.0,
                      horizon_years=10, steps_per_year=12)
    rng = np.random.default_rng(1)
    result = simulate(model, plan, n_paths=200_000, rng=rng)
    analytic = 1.07 ** 10
    assert result.terminal_values.mean() == pytest.approx(analytic, rel=0.02)


def test_bootstrap_only_draws_from_history():
    history = np.array([-0.02, 0.0, 0.01, 0.03])
    model = BootstrapModel(history)
    rng = np.random.default_rng(2)
    draws = model.generate(50, 20, rng)
    assert np.isin(draws, history).all()


def test_contributions_compound_with_zero_return():
    # Zero return means the balance is just the sum of everything paid in.
    model = GBMModel(annual_return=0.0, annual_volatility=0.0, steps_per_year=12)
    plan = PlanConfig(initial_investment=1000.0, annual_contribution=1200.0,
                      horizon_years=5, steps_per_year=12)
    rng = np.random.default_rng(3)
    result = simulate(model, plan, n_paths=10, rng=rng)
    assert np.allclose(result.terminal_values, result.total_contributed)
    assert result.total_contributed == pytest.approx(1000.0 + 1200.0 * 5)


def test_probability_bounds_and_extremes():
    values = np.array([100.0, 200.0, 300.0, 400.0])
    assert analysis.probability_of_target(values, 0.0) == 1.0
    assert analysis.probability_of_target(values, 1_000.0) == 0.0
    p = analysis.probability_of_target(values, 250.0)
    assert 0.0 <= p <= 1.0


def test_percentiles_are_ordered():
    rng = np.random.default_rng(4)
    values = rng.lognormal(mean=12, sigma=0.5, size=50_000)
    pcts = analysis.percentiles(values)
    assert pcts[5] < pcts[25] < pcts[50] < pcts[75] < pcts[95]


def test_var_and_cvar_are_in_the_left_tail():
    rng = np.random.default_rng(5)
    values = rng.normal(1000, 100, size=100_000)
    var = analysis.value_at_risk(values, 0.05)
    cvar = analysis.conditional_value_at_risk(values, 0.05)
    median = float(np.median(values))
    assert cvar <= var <= median
