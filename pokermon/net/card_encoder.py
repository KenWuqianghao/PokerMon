"""Card encoding with learned embeddings for rank and suit."""

from __future__ import annotations

import torch
import torch.nn as nn

from pokermon.game.card import NUM_RANKS, NUM_SUITS


class CardEncoder(nn.Module):
    """Encode cards using learned rank and suit embeddings.

    Each card → rank_embedding(8) + suit_embedding(4) = 12d → projected to embed_dim.
    """

    def __init__(self, embed_dim: int = 20, rank_dim: int = 8, suit_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.rank_embed = nn.Embedding(NUM_RANKS, rank_dim)
        self.suit_embed = nn.Embedding(NUM_SUITS, suit_dim)
        self.proj = nn.Linear(rank_dim + suit_dim, embed_dim)

    def forward(self, cards: torch.Tensor) -> torch.Tensor:
        """Encode cards.

        Args:
            cards: (batch, num_cards, 2) — last dim is [rank, suit].
                   Use rank=-1/suit=-1 for padding (will be zeroed out).

        Returns:
            (batch, num_cards, embed_dim)
        """
        # Mask padded cards
        valid = (cards[:, :, 0] >= 0).unsqueeze(-1).float()  # (batch, num_cards, 1)

        ranks = cards[:, :, 0].clamp(min=0)
        suits = cards[:, :, 1].clamp(min=0)

        r = self.rank_embed(ranks)  # (batch, num_cards, rank_dim)
        s = self.suit_embed(suits)  # (batch, num_cards, suit_dim)

        combined = torch.cat([r, s], dim=-1)
        encoded = self.proj(combined)  # (batch, num_cards, embed_dim)

        return encoded * valid  # Zero out padding


class PrivateCardEncoder(nn.Module):
    """Encode 2 hole cards → flat vector."""

    def __init__(self, embed_dim: int = 20) -> None:
        super().__init__()
        self.encoder = CardEncoder(embed_dim=embed_dim)

    def forward(self, cards: torch.Tensor) -> torch.Tensor:
        """Args: cards (batch, 2, 2). Returns: (batch, 2*embed_dim)."""
        encoded = self.encoder(cards)  # (batch, 2, embed_dim)
        return encoded.flatten(start_dim=1)  # (batch, 2*embed_dim)


class CommunityCardEncoder(nn.Module):
    """Encode 0-5 community cards → fixed-size vector via sum pooling."""

    def __init__(self, embed_dim: int = 20) -> None:
        super().__init__()
        self.encoder = CardEncoder(embed_dim=embed_dim)

    def forward(self, cards: torch.Tensor) -> torch.Tensor:
        """Args: cards (batch, 5, 2). Returns: (batch, embed_dim)."""
        encoded = self.encoder(cards)  # (batch, 5, embed_dim)
        return encoded.sum(dim=1)  # (batch, embed_dim) — sum pool
