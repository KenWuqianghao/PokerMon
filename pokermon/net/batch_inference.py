"""Batched GPU inference during traversals.

Collects multiple inference requests and processes them in a single batch
for better GPU utilization during MCCFR traversals.
"""

from __future__ import annotations

import numpy as np
import torch


class BatchInference:
    """Batched inference wrapper for advantage/strategy networks."""

    def __init__(self, model: torch.nn.Module, device: torch.device, batch_size: int = 256) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.model.eval()

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run inference on a single feature vector.

        Args:
            features: (feature_dim,) numpy array.

        Returns:
            (output_dim,) numpy array of network outputs.
        """
        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        out = self.model(x).squeeze(0).cpu().numpy()
        return out

    @torch.no_grad()
    def predict_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """Run inference on a batch of features.

        Args:
            features_batch: (batch, feature_dim) numpy array.

        Returns:
            (batch, output_dim) numpy array.
        """
        x = torch.tensor(features_batch, dtype=torch.float32, device=self.device)
        out = self.model(x).cpu().numpy()
        return out
