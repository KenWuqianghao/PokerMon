"""Tests for regret matching."""

import numpy as np
from numpy.testing import assert_allclose

from pokermon.cfr.regret_matching import regret_match, regret_match_masked


def test_uniform_from_zeros():
    """Zero regrets → uniform distribution."""
    strat = regret_match(np.zeros(3))
    assert_allclose(strat, [1 / 3, 1 / 3, 1 / 3])


def test_positive_regrets():
    """Positive regrets → proportional strategy."""
    strat = regret_match(np.array([1.0, 3.0, 0.0]))
    assert_allclose(strat, [0.25, 0.75, 0.0])


def test_negative_regrets_ignored():
    """Negative regrets clipped to zero."""
    strat = regret_match(np.array([-5.0, 2.0, 8.0]))
    assert_allclose(strat, [0.0, 0.2, 0.8])


def test_all_negative():
    """All negative → uniform."""
    strat = regret_match(np.array([-1.0, -2.0, -3.0]))
    assert_allclose(strat, [1 / 3, 1 / 3, 1 / 3])


def test_masked_basic():
    """Masked regret matching."""
    mask = np.array([True, False, True])
    strat = regret_match_masked(np.array([3.0, 5.0, 1.0]), mask)
    assert strat[1] == 0.0  # Illegal action
    assert_allclose(strat, [0.75, 0.0, 0.25])


def test_masked_all_zero():
    """Uniform over legal actions when all regrets zero."""
    mask = np.array([True, False, True, True])
    strat = regret_match_masked(np.zeros(4), mask)
    assert strat[1] == 0.0
    assert_allclose(strat[0], 1 / 3)
    assert_allclose(strat[2], 1 / 3)
    assert_allclose(strat[3], 1 / 3)


def test_probability_sums_to_one():
    for _ in range(100):
        regrets = np.random.randn(7)
        strat = regret_match(regrets)
        assert_allclose(strat.sum(), 1.0)
        assert np.all(strat >= 0)
