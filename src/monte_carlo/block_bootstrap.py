"""Block bootstrap: resampling that keeps short-run patterns intact.

The plain bootstrap in models.py draws one return at a time, which throws away
any serial structure: momentum, mean reversion, volatility clustering. The block
bootstrap instead resamples contiguous blocks of consecutive returns, so a
streak of bad days can travel together into a simulated path. It is the more
honest resampler when returns are not independent, which they rarely are.

Set ``block_size`` to 1 and it reduces to the ordinary bootstrap.
"""

from __future__ import annotations

import numpy as np


class BlockBootstrapModel:
    """Circular block bootstrap over a history of per-step returns."""

    def __init__(self, historical_returns: np.ndarray, block_size: int = 5) -> None:
        history = np.asarray(historical_returns, dtype=float)
        history = history[~np.isnan(history)]
        if history.size == 0:
            raise ValueError("historical_returns is empty after dropping NaNs.")
        if block_size < 1:
            raise ValueError("block_size must be at least 1.")
        self.history = history
        self.block_size = block_size

    def generate(
        self, n_paths: int, n_steps: int, rng: np.random.Generator
    ) -> np.ndarray:
        L = self.block_size
        n = self.history.size
        n_blocks = int(np.ceil(n_steps / L))

        # Random starting index for every block of every path; wrap around the
        # end of the history so late starts are still allowed.
        starts = rng.integers(0, n, size=(n_paths, n_blocks))
        offsets = np.arange(L)
        # Shape (n_paths, n_blocks, L) of indices, then flatten the block axis.
        idx = (starts[:, :, None] + offsets[None, None, :]) % n
        drawn = self.history[idx].reshape(n_paths, n_blocks * L)
        return drawn[:, :n_steps]
