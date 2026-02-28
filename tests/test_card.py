"""Tests for card.py."""

from pokermon.game.card import (
    NUM_CARDS,
    RANK_CHARS,
    SUIT_CHARS,
    Rank,
    Suit,
    card_from_index,
    card_from_str,
    card_from_treys,
    card_index,
    card_to_str,
    card_to_treys,
)


def test_card_index_roundtrip():
    for idx in range(NUM_CARDS):
        rank, suit = card_from_index(idx)
        assert card_index(rank, suit) == idx


def test_card_str_roundtrip():
    for idx in range(NUM_CARDS):
        s = card_to_str(idx)
        assert card_from_str(s) == idx


def test_specific_cards():
    # Ace of spades
    ace_spades = card_from_str("As")
    rank, suit = card_from_index(ace_spades)
    assert rank == Rank.ACE
    assert suit == Suit.SPADES

    # Two of clubs
    two_clubs = card_from_str("2c")
    rank, suit = card_from_index(two_clubs)
    assert rank == Rank.TWO
    assert suit == Suit.CLUBS


def test_treys_roundtrip():
    for idx in range(NUM_CARDS):
        treys_int = card_to_treys(idx)
        assert card_from_treys(treys_int) == idx


def test_unique_indices():
    indices = set()
    for r in range(13):
        for s in range(4):
            idx = card_index(r, s)
            assert idx not in indices
            indices.add(idx)
    assert len(indices) == 52
