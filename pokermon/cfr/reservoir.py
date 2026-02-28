"""Reservoir sampling buffer (Vitter's Algorithm R).

Used to maintain a fixed-size buffer of training samples with
uniform probability of including any past sample.
"""

from __future__ import annotations

import numpy as np


class ReservoirBuffer:
    """Fixed-size reservoir sampling buffer for Deep CFR training data.

    Stores (features, targets, weights) tuples.
    """

    def __init__(self, capacity: int, feature_dim: int, target_dim: int) -> None:
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.target_dim = target_dim

        self.features = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.targets = np.zeros((capacity, target_dim), dtype=np.float32)
        self.weights = np.zeros(capacity, dtype=np.float32)

        self._size = 0
        self._count = 0  # Total samples seen (for reservoir sampling)
        self._rng = np.random.RandomState(42)

    @property
    def size(self) -> int:
        """Number of samples currently in the buffer."""
        return self._size

    def add(self, features: np.ndarray, targets: np.ndarray, weight: float = 1.0) -> None:
        """Add a sample to the buffer using reservoir sampling.

        Args:
            features: Feature vector of shape (feature_dim,).
            targets: Target vector of shape (target_dim,).
            weight: Sample weight (e.g., iteration number for linear weighting).
        """
        if self._size < self.capacity:
            idx = self._size
            self._size += 1
        else:
            # Reservoir sampling: replace a random existing sample with prob capacity/count
            j = self._rng.randint(0, self._count + 1)
            if j < self.capacity:
                idx = j
            else:
                self._count += 1
                return

        self.features[idx] = features
        self.targets[idx] = targets
        self.weights[idx] = weight
        self._count += 1

    def add_batch(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        """Add multiple samples."""
        n = features.shape[0]
        if weights is None:
            weights = np.ones(n, dtype=np.float32)
        for i in range(n):
            self.add(features[i], targets[i], weights[i])

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch from the buffer.

        Returns:
            (features, targets, weights) — each as numpy arrays.
        """
        if self._size == 0:
            raise ValueError("Buffer is empty")
        indices = self._rng.randint(0, self._size, size=min(batch_size, self._size))
        return self.features[indices], self.targets[indices], self.weights[indices]

    def clear(self) -> None:
        """Clear the buffer."""
        self._size = 0
        self._count = 0

    def __len__(self) -> int:
        return self._size
