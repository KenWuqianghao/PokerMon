"""Tests for action.py."""

from pokermon.game.action import Action, compute_bet_size, legal_actions_mask


def test_action_values():
    assert Action.FOLD == 0
    assert Action.CHECK_CALL == 1
    assert Action.ALL_IN == 6


def test_compute_bet_size_fold():
    assert compute_bet_size(Action.FOLD, pot=200, to_call=100, stack=1000, min_raise=100) == 0


def test_compute_bet_size_check_call():
    assert compute_bet_size(Action.CHECK_CALL, pot=200, to_call=100, stack=1000, min_raise=100) == 100


def test_compute_bet_size_all_in():
    assert compute_bet_size(Action.ALL_IN, pot=200, to_call=100, stack=1000, min_raise=100) == 1000


def test_compute_bet_size_pot():
    # 100% pot: pot_after_call = 200 + 100 = 300, raise = 300, total = 100 + 300 = 400
    size = compute_bet_size(Action.BET_100X, pot=200, to_call=100, stack=1000, min_raise=100)
    assert size == 400


def test_compute_bet_size_clamp_to_stack():
    size = compute_bet_size(Action.BET_100X, pot=2000, to_call=500, stack=600, min_raise=100)
    assert size == 600  # Clamped to stack


def test_legal_actions_facing_bet():
    mask = legal_actions_mask(stack=1000, to_call=100, pot=300, min_raise=100, can_check=False)
    assert mask[Action.FOLD] is True
    assert mask[Action.CHECK_CALL] is True
    assert mask[Action.ALL_IN] is True


def test_legal_actions_no_bet():
    mask = legal_actions_mask(stack=1000, to_call=0, pot=200, min_raise=100, can_check=True)
    assert mask[Action.FOLD] is False  # Can't fold when no bet to face
    assert mask[Action.CHECK_CALL] is True


def test_legal_actions_short_stack():
    # Stack of 50 — can only call or go all-in (bets require more chips)
    mask = legal_actions_mask(stack=50, to_call=100, pot=200, min_raise=100, can_check=False)
    assert mask[Action.FOLD] is True
    assert mask[Action.CHECK_CALL] is True
    assert mask[Action.ALL_IN] is True
    # Bet sizes should be False (total would exceed stack)
    assert mask[Action.BET_033X] is False
