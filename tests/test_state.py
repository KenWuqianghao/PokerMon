"""Tests for state.py."""

from pokermon.game.state import GameState, PlayerState, Street


def test_player_state_frozen():
    ps = PlayerState(stack=1000, bet=0, total_bet=0, folded=False, all_in=False)
    try:
        ps.stack = 500
        assert False, "Should be frozen"
    except AttributeError:
        pass


def test_active_players():
    players = (
        PlayerState(stack=1000, bet=0, total_bet=0, folded=False, all_in=False),
        PlayerState(stack=0, bet=0, total_bet=100, folded=True, all_in=False),
        PlayerState(stack=500, bet=0, total_bet=0, folded=False, all_in=False),
    )
    state = GameState(
        num_players=3,
        players=players,
        street=Street.PREFLOP,
        community_cards=(),
        pot=100,
        current_player=0,
        button=0,
        small_blind=50,
        big_blind=100,
        min_raise=100,
    )
    assert state.active_players == [0, 2]


def test_players_with_chips():
    players = (
        PlayerState(stack=1000, bet=0, total_bet=0, folded=False, all_in=False),
        PlayerState(stack=0, bet=0, total_bet=1000, folded=False, all_in=True),
        PlayerState(stack=500, bet=0, total_bet=0, folded=False, all_in=False),
    )
    state = GameState(
        num_players=3,
        players=players,
        street=Street.PREFLOP,
        community_cards=(),
        pot=1000,
        current_player=0,
        button=0,
        small_blind=50,
        big_blind=100,
        min_raise=100,
    )
    assert state.players_with_chips == [0, 2]


def test_total_chips():
    players = (
        PlayerState(stack=900, bet=100, total_bet=100, folded=False, all_in=False),
        PlayerState(stack=800, bet=200, total_bet=200, folded=False, all_in=False),
    )
    state = GameState(
        num_players=2,
        players=players,
        street=Street.PREFLOP,
        community_cards=(),
        pot=500,
        current_player=0,
        button=0,
        small_blind=50,
        big_blind=100,
        min_raise=100,
    )
    assert state.total_chips == 500 + 900 + 800 + 100 + 200
