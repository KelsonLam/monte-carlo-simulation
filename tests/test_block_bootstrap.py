"""Tests for the block bootstrap model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from monte_carlo.block_bootstrap import BlockBootstrapModel


def test_shape_and_values_from_history():
    history = np.array([-0.02, 0.0, 0.01, 0.03, 0.015, -0.01])
    model = BlockBootstrapModel(history, block_size=5)
    draws = model.generate(40, 23, np.random.default_rng(0))
    assert draws.shape == (40, 23)
    assert np.isin(draws, history).all()


def test_blocks_preserve_consecutive_pairs():
    # A clean ramp: every value is followed by a specific next value (wrapping).
    history = np.arange(10, dtype=float)
    model = BlockBootstrapModel(history, block_size=4)
    draws = model.generate(200, 8, np.random.default_rng(1))
    # Within a block, consecutive draws differ by exactly 1 (mod 10).
    diffs = np.diff(draws, axis=1) % 10
    # Most adjacent pairs should be a step of 1; block joins are the exception.
    assert np.mean(diffs == 1) > 0.6


def test_block_size_one_is_plain_bootstrap_shape():
    history = np.array([0.01, -0.01, 0.02])
    model = BlockBootstrapModel(history, block_size=1)
    draws = model.generate(10, 7, np.random.default_rng(2))
    assert draws.shape == (10, 7)
    assert np.isin(draws, history).all()


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        BlockBootstrapModel(np.array([]), block_size=3)
    with pytest.raises(ValueError):
        BlockBootstrapModel(np.array([0.01]), block_size=0)
