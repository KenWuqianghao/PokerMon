"""Advantage (regret) network for Deep CFR.

One per seat (6 total for 6-max). Predicts counterfactual regret values
for each action given an information set encoding.

Architecture: 211 → 512 → 512 → 512 → 512 → 7 (raw advantages).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pokermon.cfr.infoset import TOTAL_DIM
from pokermon.game.action import NUM_ACTIONS


class AdvantageNet(nn.Module):
    """MLP that predicts advantage (regret) values for each action."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict raw advantage values.

        Args:
            x: (batch, input_dim) info set features.

        Returns:
            (batch, num_actions) raw advantage values (not masked).
        """
        return self.net(x)

    def predict_advantages(
        self, x: torch.Tensor, legal_mask: torch.Tensor
    ) -> torch.Tensor:
        """Predict advantages masked to legal actions.

        Illegal actions get advantage = -inf (effectively zero after regret matching).
        """
        adv = self.forward(x)
        adv = adv.masked_fill(~legal_mask.bool(), float("-inf"))
        return adv
