"""Card constants, rank/suit enums, and treys index mapping."""

from __future__ import annotations

from enum import IntEnum

NUM_RANKS = 13
NUM_SUITS = 4
NUM_CARDS = 52


class Rank(IntEnum):
    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"

# Treys uses its own integer encoding for cards.
# We map our (rank, suit) → treys int and back.
_TREYS_RANK_CHARS = "23456789TJQKA"
_TREYS_SUIT_CHARS = "shdc"  # treys ordering


def card_index(rank: int, suit: int) -> int:
    """Return a unique index 0..51 for (rank, suit)."""
    return rank * NUM_SUITS + suit


def card_from_index(idx: int) -> tuple[int, int]:
    """Return (rank, suit) from card index 0..51."""
    return idx // NUM_SUITS, idx % NUM_SUITS


def card_to_str(idx: int) -> str:
    """Human-readable string like 'As', '2c'."""
    rank, suit = card_from_index(idx)
    return RANK_CHARS[rank] + SUIT_CHARS[suit]


def card_from_str(s: str) -> int:
    """Parse 'As' → card index."""
    rank = RANK_CHARS.index(s[0])
    suit = SUIT_CHARS.index(s[1])
    return card_index(rank, suit)


def card_to_treys(idx: int) -> int:
    """Convert our card index to treys integer representation."""
    from treys import Card

    rank, suit = card_from_index(idx)
    treys_str = _TREYS_RANK_CHARS[rank] + _TREYS_SUIT_CHARS[suit]
    return Card.new(treys_str)


def card_from_treys(treys_int: int) -> int:
    """Convert treys integer representation to our card index."""
    from treys import Card

    s = Card.int_to_str(treys_int)
    rank = _TREYS_RANK_CHARS.index(s[0])
    suit = _TREYS_SUIT_CHARS.index(s[1])
    return card_index(rank, suit)
