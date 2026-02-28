"""Tests for deck.py."""

from pokermon.game.deck import Deck


def test_deal_all_cards():
    deck = Deck(seed=42)
    deck.shuffle()
    cards = deck.deal(52)
    assert len(cards) == 52
    assert len(set(cards)) == 52  # All unique


def test_remaining():
    deck = Deck(seed=42)
    deck.shuffle()
    assert deck.remaining() == 52
    deck.deal(5)
    assert deck.remaining() == 47


def test_seeded_reproducibility():
    d1 = Deck(seed=123)
    d1.shuffle()
    cards1 = d1.deal(10)

    d2 = Deck(seed=123)
    d2.shuffle()
    cards2 = d2.deal(10)

    assert cards1 == cards2


def test_different_seeds():
    d1 = Deck(seed=1)
    d1.shuffle()
    cards1 = d1.deal(52)

    d2 = Deck(seed=2)
    d2.shuffle()
    cards2 = d2.deal(52)

    assert cards1 != cards2


def test_deal_overflow():
    deck = Deck(seed=42)
    deck.shuffle()
    deck.deal(50)
    try:
        deck.deal(5)
        assert False, "Should have raised"
    except ValueError:
        pass
