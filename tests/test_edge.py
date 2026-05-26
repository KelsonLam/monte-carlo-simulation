"""Edge-case and validation tests for the Monte Carlo engine."""

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


def test_gbm_rejects_bad_inputs():
    with pytest.raises(ValueError):
        GBMModel(annual_return=0.07, annual_volatility=-0.1)
    with pytest.raises(ValueError):
        GBMModel(annual_return=0.07, annual_volatility=0.2, steps_per_year=0)


def test_bootstrap_rejects_empty_history():
    with pytest.raises(ValueError):
        BootstrapModel(np.array([]))


def test_bootstrap_drops_nans():
    model = BootstrapModel(np.array([0.01, np.nan, -0.02]))
    draws = model.generate(20, 10, np.random.default_rng(0))
    assert not np.isnan(draws).any()


def test_simulate_rejects_zero_horizon():
    model = GBMModel(0.07, 0.15)
    with pytest.raises(ValueError):
        simulate(model, PlanConfig(horizon_years=0), 100, np.random.default_rng(0))


def test_contributions_increase_terminal_value():
    rng = np.random.default_rng(1)
    no_contrib = simulate(GBMModel(0.05, 0.10), PlanConfig(50000, 0, 10, 12), 5000, rng)
    rng = np.random.default_rng(1)
    with_contrib = simulate(GBMModel(0.05, 0.10), PlanConfig(50000, 6000, 10, 12), 5000, rng)
    assert with_contrib.terminal_values.mean() > no_contrib.terminal_values.mean()
