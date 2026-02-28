"""Tests for the NLHE engine."""

from pokermon.game.action import Action
from pokermon.game.deck import Deck
from pokermon.game.engine import apply_action, get_legal_actions, new_hand
from pokermon.game.state import Street


def test_new_hand_basic():
    state = new_hand(num_players=6, small_blind=50, big_blind=100, button=0)
    assert state.num_players == 6
    assert state.street == Street.PREFLOP
    assert not state.is_terminal
    # Check blinds posted
    sb_pos = 1  # (button + 1) % 6
    bb_pos = 2  # (button + 2) % 6
    assert state.players[sb_pos].bet == 50
    assert state.players[bb_pos].bet == 100
    # UTG should be first to act
    assert state.current_player == 3


def test_new_hand_heads_up():
    state = new_hand(num_players=2, small_blind=50, big_blind=100, button=0)
    # In heads-up, button IS the SB
    assert state.players[0].bet == 50   # Button = SB
    assert state.players[1].bet == 100  # Other = BB
    # Button/SB acts first preflop in heads-up
    assert state.current_player == 0


def test_chip_conservation():
    deck = Deck(seed=42)
    state = new_hand(num_players=6, small_blind=50, big_blind=100, button=0, deck=deck)
    initial_chips = state.total_chips

    # Play out a hand
    while not state.is_terminal:
        actions = get_legal_actions(state)
        # Just call/check everything
        if Action.CHECK_CALL in actions:
            state = apply_action(state, Action.CHECK_CALL)
        else:
            state = apply_action(state, actions[0])

    # Chips should be conserved
    final_chips = sum(p.stack for p in state.players)
    assert final_chips == initial_chips, f"Chips not conserved: {initial_chips} -> {final_chips}"


def test_fold_to_win():
    state = new_hand(num_players=2, small_blind=50, big_blind=100, button=0)
    # Button (SB) folds
    state = apply_action(state, Action.FOLD)
    assert state.is_terminal
    # BB wins SB's blind
    assert sum(state.payoffs) == 0  # Zero-sum


def test_everyone_folds_to_bb():
    state = new_hand(num_players=6, small_blind=50, big_blind=100, button=0)
    # Everyone folds to BB
    for _ in range(4):  # UTG through button
        state = apply_action(state, Action.FOLD)
    # SB folds
    state = apply_action(state, Action.FOLD)
    assert state.is_terminal
    # BB wins
    bb_pos = 2
    assert state.payoffs[bb_pos] > 0


def test_call_down_to_showdown():
    deck = Deck(seed=42)
    state = new_hand(num_players=2, small_blind=50, big_blind=100, button=0, deck=deck)

    while not state.is_terminal:
        actions = get_legal_actions(state)
        state = apply_action(state, Action.CHECK_CALL)

    assert state.is_terminal
    assert state.street == Street.SHOWDOWN
    assert len(state.community_cards) == 5
    assert sum(state.payoffs) == 0


def test_raise_and_call():
    deck = Deck(seed=42)
    state = new_hand(num_players=2, small_blind=50, big_blind=100, button=0, deck=deck)

    # Button raises 50% pot
    state = apply_action(state, Action.BET_050X)
    # BB calls
    state = apply_action(state, Action.CHECK_CALL)

    # Should be on flop
    assert state.street == Street.FLOP
    assert len(state.community_cards) == 3


def test_all_in_heads_up():
    deck = Deck(seed=42)
    state = new_hand(
        num_players=2, stacks=[1000, 1000], small_blind=50, big_blind=100, button=0, deck=deck
    )

    # Button shoves
    state = apply_action(state, Action.ALL_IN)
    # BB calls
    state = apply_action(state, Action.CHECK_CALL)

    assert state.is_terminal
    assert len(state.community_cards) == 5  # Dealt to river
    # One player should have ~2000, other ~0
    stacks = [p.stack for p in state.players]
    assert sum(stacks) == 2000
    assert 0 in stacks or stacks[0] == stacks[1]  # Someone won or split


def test_payoffs_zero_sum():
    """Payoffs should always sum to zero."""
    for seed in range(20):
        deck = Deck(seed=seed)
        state = new_hand(num_players=6, small_blind=50, big_blind=100, button=0, deck=deck)

        import random
        rng = random.Random(seed)
        while not state.is_terminal:
            actions = get_legal_actions(state)
            action = rng.choice(actions)
            state = apply_action(state, action)

        assert sum(state.payoffs) == 0, f"Seed {seed}: payoffs sum to {sum(state.payoffs)}"


def test_side_pots():
    """Test with unequal stacks creating side pots."""
    deck = Deck(seed=42)
    state = new_hand(
        num_players=3,
        stacks=[500, 1000, 1500],
        small_blind=50,
        big_blind=100,
        button=0,
        deck=deck,
    )

    # Player 2 (UTG, since 3-handed with button=0, sb=1, bb=2, UTG=0)
    # Wait, 3-handed: button=0, sb=1, bb=2, UTG=0 (wraps)
    # Actually for 3 players: sb=(0+1)%3=1, bb=(0+2)%3=2, first_to_act=(2+1)%3=0
    # Player 0 shoves (500 chips)
    state = apply_action(state, Action.ALL_IN)
    # Player 1 (SB) shoves (1000 chips - 50 sb = 950 remaining)
    state = apply_action(state, Action.ALL_IN)
    # Player 2 (BB) calls
    state = apply_action(state, Action.CHECK_CALL)

    assert state.is_terminal
    total_stacks = sum(p.stack for p in state.players)
    assert total_stacks == 3000  # Chip conservation
