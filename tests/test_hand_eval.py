"""Tests for hand_eval.py."""

from pokermon.game.card import card_from_str
from pokermon.game.hand_eval import evaluate, hand_rank_string


def test_royal_flush():
    hole = [card_from_str("As"), card_from_str("Ks")]
    board = [card_from_str("Qs"), card_from_str("Js"), card_from_str("Ts")]
    score = evaluate(hole, board)
    assert score == 1  # Best possible hand


def test_full_house_beats_flush():
    # Full house
    fh_hole = [card_from_str("Ah"), card_from_str("Ad")]
    fh_board = [card_from_str("Kh"), card_from_str("Kd"), card_from_str("Ks"), card_from_str("2c"), card_from_str("3c")]
    fh_score = evaluate(fh_hole, fh_board)

    # Flush
    fl_hole = [card_from_str("9h"), card_from_str("2h")]
    fl_board = [card_from_str("Kh"), card_from_str("Qh"), card_from_str("Jh"), card_from_str("3c"), card_from_str("4d")]
    fl_score = evaluate(fl_hole, fl_board)

    assert fh_score < fl_score  # Lower score = better hand


def test_high_card():
    hole = [card_from_str("2c"), card_from_str("7d")]
    board = [card_from_str("9h"), card_from_str("Js"), card_from_str("Ks"), card_from_str("4c"), card_from_str("3h")]
    score = evaluate(hole, board)
    rank_str = hand_rank_string(score)
    assert "High Card" in rank_str
