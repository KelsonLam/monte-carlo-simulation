"""Ways to generate future returns.

Two models are offered, and the choice is itself a modelling decision worth
thinking about:

    GBM         draws returns from a normal distribution (geometric Brownian
                motion). Clean and standard, but it assumes returns are normal,
                which understates how often large moves really happen.

    Bootstrap   resamples actual historical returns with replacement. It makes
                no assumption about the shape of the distribution, so it keeps
                the fat tails that were in the data. It does assume the future
                is drawn from the same pot as the past.

Both return an array of simple per-step returns with shape (n_paths, n_steps).
"""

from __future__ import annotations

import numpy as np


class GBMModel:
    """Geometric Brownian motion return generator.

    The drift is set so the expected annual growth factor matches
    ``annual_return``. Per step of length dt = 1 / steps_per_year, the log
    return is normal with mean (mu - 0.5 * sigma**2) * dt and standard
    deviation sigma * sqrt(dt), where mu = ln(1 + annual_return).
    """

    def __init__(
        self,
        annual_return: float,
        annual_volatility: float,
        steps_per_year: int = 12,
    ) -> None:
        if annual_volatility < 0:
            raise ValueError("annual_volatility cannot be negative.")
        if steps_per_year <= 0:
            raise ValueError("steps_per_year must be positive.")
        self.annual_return = annual_return
        self.annual_volatility = annual_volatility
        self.steps_per_year = steps_per_year

    def generate(
        self, n_paths: int, n_steps: int, rng: np.random.Generator
    ) -> np.ndarray:
        dt = 1.0 / self.steps_per_year
        mu = np.log(1.0 + self.annual_return)
        sigma = self.annual_volatility
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt)
        shocks = rng.standard_normal((n_paths, n_steps))
        log_returns = drift + diffusion * shocks
        return np.expm1(log_returns)   # simple returns: exp(log) - 1


class BootstrapModel:
    """Resample historical per-step returns with replacement.

    ``historical_returns`` should already be at the simulation step frequency
    (for example monthly returns if the simulation uses monthly steps).
    """

    def __init__(self, historical_returns: np.ndarray) -> None:
        history = np.asarray(historical_returns, dtype=float)
        history = history[~np.isnan(history)]
        if history.size == 0:
            raise ValueError("historical_returns is empty after dropping NaNs.")
        self.history = history

    def generate(
        self, n_paths: int, n_steps: int, rng: np.random.Generator
    ) -> np.ndarray:
        idx = rng.integers(0, self.history.size, size=(n_paths, n_steps))
        return self.history[idx]
