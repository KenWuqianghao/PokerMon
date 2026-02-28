"""Thin wrapper around treys for hand evaluation."""

from __future__ import annotations

from treys import Evaluator

from pokermon.game.card import card_to_treys

_evaluator = Evaluator()


def evaluate(hole_cards: list[int], community_cards: list[int]) -> int:
    """Evaluate a hand. Lower score = better hand (treys convention).

    Args:
        hole_cards: List of 2 card indices (our encoding).
        community_cards: List of 3-5 card indices.

    Returns:
        Hand rank (1 = Royal Flush, 7462 = worst).
    """
    treys_hole = [card_to_treys(c) for c in hole_cards]
    treys_board = [card_to_treys(c) for c in community_cards]
    return _evaluator.evaluate(treys_board, treys_hole)


def hand_rank_class(score: int) -> int:
    """Map a hand score to a class (1=Straight Flush .. 9=High Card)."""
    return _evaluator.get_rank_class(score)


def hand_rank_string(score: int) -> str:
    """Human-readable hand class string."""
    return _evaluator.class_to_string(hand_rank_class(score))
