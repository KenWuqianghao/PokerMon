"""Tests for reservoir sampling buffer."""

import numpy as np
from numpy.testing import assert_allclose

from pokermon.cfr.reservoir import ReservoirBuffer


def test_basic_add_and_sample():
    buf = ReservoirBuffer(capacity=100, feature_dim=5, target_dim=3)
    assert len(buf) == 0

    buf.add(np.ones(5), np.ones(3), weight=1.0)
    assert len(buf) == 1

    features, targets, weights = buf.sample(1)
    assert features.shape == (1, 5)
    assert targets.shape == (1, 3)
    assert_allclose(features[0], 1.0)


def test_reservoir_sampling():
    """Buffer should not exceed capacity."""
    buf = ReservoirBuffer(capacity=100, feature_dim=2, target_dim=1)
    for i in range(1000):
        buf.add(np.array([i, i], dtype=np.float32), np.array([i], dtype=np.float32))
    assert len(buf) == 100


def test_batch_add():
    buf = ReservoirBuffer(capacity=50, feature_dim=3, target_dim=2)
    features = np.random.randn(20, 3).astype(np.float32)
    targets = np.random.randn(20, 2).astype(np.float32)
    buf.add_batch(features, targets)
    assert len(buf) == 20


def test_sample_batch_size():
    buf = ReservoirBuffer(capacity=100, feature_dim=4, target_dim=2)
    for i in range(50):
        buf.add(np.random.randn(4).astype(np.float32), np.random.randn(2).astype(np.float32))

    features, targets, weights = buf.sample(10)
    assert features.shape == (10, 4)
    assert targets.shape == (10, 2)
    assert weights.shape == (10,)


def test_clear():
    buf = ReservoirBuffer(capacity=100, feature_dim=3, target_dim=1)
    for i in range(10):
        buf.add(np.zeros(3), np.zeros(1))
    assert len(buf) == 10
    buf.clear()
    assert len(buf) == 0


def test_weights():
    buf = ReservoirBuffer(capacity=100, feature_dim=2, target_dim=1)
    buf.add(np.array([1.0, 2.0]), np.array([3.0]), weight=5.0)
    _, _, weights = buf.sample(1)
    assert weights[0] == 5.0
