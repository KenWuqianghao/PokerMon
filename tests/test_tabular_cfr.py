"""Tests for tabular CFR on Kuhn poker."""

import numpy as np

from pokermon.cfr.tabular_cfr import TabularCFR
from pokermon.eval.exploitability import compute_exploitability_kuhn
from pokermon.game.kuhn import all_kuhn_deals


def test_cfr_converges_kuhn():
    """Tabular CFR on Kuhn should reach exploitability < 0.05 in 10K iterations."""
    cfr = TabularCFR(num_actions=2)
    cfr.train(all_kuhn_deals, num_iterations=10000)

    # Extract average strategy
    strategy = {}
    for info_set in cfr.strategy_sum:
        strategy[info_set] = cfr.get_average_strategy(info_set)

    exploit = compute_exploitability_kuhn(strategy)
    assert exploit < 0.05, f"Exploitability {exploit:.4f} > 0.05"


def test_cfr_game_value():
    """Kuhn game value is -1/18 ≈ -0.0556 for player 0."""
    cfr = TabularCFR(num_actions=2)
    values = cfr.train(all_kuhn_deals, num_iterations=10000)
    # Average game value should be close to -1/18
    assert abs(values[-1] - (-1 / 18)) < 0.05, f"Game value {values[-1]:.4f} != -1/18"


def test_cfr_known_strategy():
    """Verify some known Nash equilibrium properties of Kuhn poker.

    At Nash equilibrium:
    - J should never bet as first action (bet frequency = 0 or α ≤ 1/3)
    - K should always bet when facing a bet (call/bet frequency = 1)
    """
    cfr = TabularCFR(num_actions=2)
    cfr.train(all_kuhn_deals, num_iterations=20000)

    strategy = {}
    for info_set in cfr.strategy_sum:
        strategy[info_set] = cfr.get_average_strategy(info_set)

    # K facing a bet should always call (bet = call after opponent bet)
    if "Kb" in strategy:
        assert strategy["Kb"][1] > 0.95, f"K should call bet: {strategy['Kb']}"

    # J opening should rarely bet (optimal is bet with prob α ≤ 1/3)
    if "J" in strategy:
        assert strategy["J"][1] < 0.4, f"J should rarely bet first: {strategy['J']}"
