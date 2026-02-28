"""Strategy network for Deep CFR.

Single shared network that learns the average policy across all players.
Uses masked softmax output and cross-entropy loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pokermon.cfr.infoset import TOTAL_DIM
from pokermon.game.action import NUM_ACTIONS


class StrategyNet(nn.Module):
    """MLP that predicts a strategy (action probabilities) from info set features."""

    def __init__(
        self,
        input_dim: int = TOTAL_DIM,
        hidden_dim: int = 512,
        num_actions: int = NUM_ACTIONS,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        """Predict action probabilities with masked softmax.

        Args:
            x: (batch, input_dim) info set features.
            legal_mask: (batch, num_actions) boolean mask of legal actions.

        Returns:
            (batch, num_actions) action probabilities (sums to 1 over legal actions).
        """
        logits = self.net(x)
        # Mask illegal actions with -inf before softmax
        logits = logits.masked_fill(~legal_mask.bool(), float("-inf"))
        return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        """Alias for forward."""
        return self.forward(x, legal_mask)
