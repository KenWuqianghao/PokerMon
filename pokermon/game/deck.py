"""Seeded deck for reproducible shuffling."""

from __future__ import annotations

import numpy as np

from pokermon.game.card import NUM_CARDS


class Deck:
    """A standard 52-card deck with optional seeded shuffling."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.RandomState(seed)
        self._cards = np.arange(NUM_CARDS, dtype=np.int32)
        self._pos = 0

    def shuffle(self) -> None:
        """Shuffle the deck and reset position."""
        self._rng.shuffle(self._cards)
        self._pos = 0

    def deal(self, n: int = 1) -> list[int]:
        """Deal n cards from the top."""
        if self._pos + n > NUM_CARDS:
            raise ValueError(f"Cannot deal {n} cards, only {NUM_CARDS - self._pos} remaining")
        cards = self._cards[self._pos : self._pos + n].tolist()
        self._pos += n
        return cards

    def remaining(self) -> int:
        """Number of cards remaining in deck."""
        return NUM_CARDS - self._pos

    def reseed(self, seed: int) -> None:
        """Reset RNG with a new seed."""
        self._rng = np.random.RandomState(seed)
