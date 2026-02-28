"""Fixed-length betting history encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from pokermon.cfr.infoset import HISTORY_DIM, RAW_HISTORY_DIM


class HistoryEncoder(nn.Module):
    """Encode betting history from raw features to fixed-size vector.

    Input: 4 streets × 12 slots × 14 features = 672 → Linear → 128.
    """

    def __init__(self, input_dim: int = RAW_HISTORY_DIM, output_dim: int = HISTORY_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Args: history (batch, input_dim). Returns: (batch, output_dim)."""
        return self.net(history)
