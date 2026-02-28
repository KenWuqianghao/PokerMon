"""Tests for Deep CFR on Kuhn poker."""

import numpy as np
import pytest

from pokermon.cfr.deep_cfr import DeepCFR
from pokermon.eval.exploitability import compute_exploitability_kuhn


@pytest.mark.slow
def test_deep_cfr_kuhn_convergence():
    """Deep CFR on Kuhn should reach exploitability < 0.15 after 200 iterations.

    This is a relaxed threshold compared to tabular CFR since neural nets approximate.
    """
    dcfr = DeepCFR(
        num_players=2,
        num_actions=2,
        hidden_dim=64,
        num_layers=2,
        buffer_capacity=200_000,
        lr=1e-3,
        sgd_steps=2000,
        batch_size=256,
    )

    dcfr.train_kuhn(num_iterations=200, traversals_per_iter=500, verbose=False)

    # Test with advantage-based strategy
    strategy = dcfr.get_kuhn_advantage_strategy()
    exploit = compute_exploitability_kuhn(strategy)

    # Neural approximation — threshold is relaxed
    assert exploit < 0.15, f"Deep CFR exploitability {exploit:.4f} > 0.15"


def test_deep_cfr_kuhn_runs():
    """Smoke test: Deep CFR runs without errors for a few iterations."""
    dcfr = DeepCFR(
        num_players=2,
        num_actions=2,
        hidden_dim=32,
        num_layers=1,
        buffer_capacity=10_000,
        lr=1e-3,
        sgd_steps=100,
        batch_size=64,
    )

    metrics = dcfr.train_kuhn(num_iterations=5, traversals_per_iter=50, verbose=False)
    assert len(metrics["iteration"]) == 5


def test_deep_cfr_strategy_extraction():
    """Test that strategy extraction works and produces valid probabilities."""
    dcfr = DeepCFR(
        num_players=2,
        num_actions=2,
        hidden_dim=32,
        num_layers=1,
        buffer_capacity=10_000,
        lr=1e-3,
        sgd_steps=100,
        batch_size=64,
    )

    dcfr.train_kuhn(num_iterations=10, traversals_per_iter=100, verbose=False)

    strategy = dcfr.get_kuhn_advantage_strategy()

    # Should have entries for all Kuhn info sets
    assert len(strategy) == 12

    # All probabilities should be valid
    for info_set, probs in strategy.items():
        assert len(probs) == 2
        assert abs(probs.sum() - 1.0) < 1e-5, f"{info_set}: probs sum to {probs.sum()}"
        assert (probs >= 0).all(), f"{info_set}: negative probs {probs}"
