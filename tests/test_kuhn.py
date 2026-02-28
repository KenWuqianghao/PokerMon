"""Tests for Kuhn poker."""

from pokermon.game.kuhn import KuhnAction, KuhnState, all_kuhn_deals


def test_all_deals():
    deals = all_kuhn_deals()
    assert len(deals) == 6
    # All unique card pairs
    pairs = [(d.cards[0], d.cards[1]) for d in deals]
    assert len(set(pairs)) == 6


def test_terminal_states():
    state = KuhnState(cards=(0, 1))

    # Check-check: terminal
    s = state.apply(KuhnAction.CHECK).apply(KuhnAction.CHECK)
    assert s.is_terminal
    assert s.history == "cc"
    # Q > J, player 1 wins
    assert s.payoff(0) == -1.0
    assert s.payoff(1) == 1.0


def test_bet_fold():
    state = KuhnState(cards=(2, 0))  # K vs J
    # Player 0 bets, player 1 folds
    s = state.apply(KuhnAction.BET).apply(KuhnAction.CHECK)  # CHECK after bet = fold
    assert s.is_terminal
    assert s.history == "bf"
    assert s.payoff(0) == 1.0
    assert s.payoff(1) == -1.0


def test_bet_call():
    state = KuhnState(cards=(2, 0))  # K vs J
    # Player 0 bets, player 1 calls
    s = state.apply(KuhnAction.BET).apply(KuhnAction.BET)  # BET after bet = call
    assert s.is_terminal
    assert s.history == "bc"
    # K > J, player 0 wins 2
    assert s.payoff(0) == 2.0
    assert s.payoff(1) == -2.0


def test_check_bet_call():
    state = KuhnState(cards=(0, 2))  # J vs K
    # Check, bet, call
    s = state.apply(KuhnAction.CHECK).apply(KuhnAction.BET).apply(KuhnAction.BET)
    assert s.is_terminal
    assert s.history == "cbc"
    # K > J, player 1 wins 2
    assert s.payoff(0) == -2.0
    assert s.payoff(1) == 2.0


def test_check_bet_fold():
    state = KuhnState(cards=(0, 2))
    s = state.apply(KuhnAction.CHECK).apply(KuhnAction.BET).apply(KuhnAction.CHECK)
    assert s.is_terminal
    assert s.history == "cbf"
    assert s.payoff(0) == -1.0  # Player 0 folds, loses ante
    assert s.payoff(1) == 1.0


def test_info_sets():
    state = KuhnState(cards=(0, 2))
    assert state.info_set == "J"  # Player 0's view: J card, no history

    s1 = state.apply(KuhnAction.CHECK)
    assert s1.info_set == "Kc"  # Player 1's view: K card, check history


def test_zero_sum():
    """All terminal payoffs should be zero-sum."""
    for deal in all_kuhn_deals():
        for h in ["cc", "bc", "bf", "cbc", "cbf"]:
            state = KuhnState(cards=deal.cards, history=h)
            if state.is_terminal:
                assert state.payoff(0) + state.payoff(1) == 0
