"""Tests for Leduc Hold'em."""

from pokermon.game.leduc import LeducAction, LeducState, all_leduc_deals


def test_all_deals():
    deals = all_leduc_deals()
    # 6 cards, choose 3 (ordered): 6*5*4 = 120
    assert len(deals) == 120


def test_terminal_fold():
    # P0 has J(0), P1 has K(4), community Q(2)
    state = LeducState(cards=(0, 4, 2))
    # P0 checks, P1 raises, P0 folds
    s = state.apply(LeducAction.CHECK_CALL)
    s = s.apply(LeducAction.RAISE)
    s = s.apply(LeducAction.FOLD)
    assert s.is_terminal
    # P1 wins P0's bet
    assert s.payoff(1) > 0
    assert s.payoff(0) < 0
    assert s.payoff(0) + s.payoff(1) == 0  # Zero-sum


def test_showdown_pair_wins():
    # P0 has J(0), P1 has K(4), community J(1) — P0 has pair of Js
    state = LeducState(cards=(0, 4, 1))
    # Both check both rounds
    s = state.apply(LeducAction.CHECK_CALL)  # P0 checks
    s = s.apply(LeducAction.CHECK_CALL)  # P1 checks -> round 1
    s = s.apply(LeducAction.CHECK_CALL)  # P0 checks
    s = s.apply(LeducAction.CHECK_CALL)  # P1 checks -> showdown
    assert s.is_terminal
    # P0 has pair of Js, P1 has K — pair beats high card
    assert s.payoff(0) > 0


def test_showdown_high_card():
    # P0 has J(0), P1 has K(4), community Q(2) — no pairs, K wins
    state = LeducState(cards=(0, 4, 2))
    s = state.apply(LeducAction.CHECK_CALL)
    s = s.apply(LeducAction.CHECK_CALL)
    s = s.apply(LeducAction.CHECK_CALL)
    s = s.apply(LeducAction.CHECK_CALL)
    assert s.is_terminal
    assert s.payoff(1) > 0  # K wins


def test_raises_capped():
    state = LeducState(cards=(0, 4, 2))
    # P0 raises, P1 raises (2 raises = capped)
    s = state.apply(LeducAction.RAISE)
    s = s.apply(LeducAction.RAISE)
    # Now only fold/call are legal
    actions = s.legal_actions()
    assert LeducAction.RAISE not in actions
    assert LeducAction.CHECK_CALL in actions
    assert LeducAction.FOLD in actions


def test_info_sets():
    state = LeducState(cards=(0, 4, 2))
    info = state.info_set
    # Player 0 has J, community Q, empty history
    assert info.startswith("JQ:")

    s1 = state.apply(LeducAction.CHECK_CALL)
    info1 = s1.info_set
    # Player 1 has K, community Q, one check
    assert info1.startswith("KQ:")


def test_zero_sum():
    """All terminal payoffs should be zero-sum."""
    deals = all_leduc_deals()
    for deal in deals[:20]:  # Test a subset
        for actions_seq in [
            [1, 1, 1, 1],  # check-check both rounds
            [2, 1, 1, 1],  # raise-call, check-check
            [1, 2, 0],  # check-raise-fold
        ]:
            state = deal
            terminal = False
            for a in actions_seq:
                if state.is_terminal:
                    terminal = True
                    break
                legal = state.legal_actions()
                if LeducAction(a) in legal:
                    state = state.apply(LeducAction(a))
                else:
                    break
            if state.is_terminal:
                assert state.payoff(0) + state.payoff(1) == 0, (
                    f"Not zero-sum: {state.payoff(0)} + {state.payoff(1)}"
                )
