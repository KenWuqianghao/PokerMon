"""Tabular CFR for Kuhn and Leduc poker (exact solution).

Implements CFR+ (regret flooring) by default for faster convergence.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from pokermon.cfr.regret_matching import regret_match


class TabularCFR:
    """CFR+ for small games (floors negative regrets, O(1/T) convergence).

    Works with any game that has:
    - state.is_terminal
    - state.current_player
    - state.info_set
    - state.payoff(player)
    - state.legal_actions()
    - state.apply(action)
    """

    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions
        self.regret_sum: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float64)
        )
        self.strategy_sum: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float64)
        )
        self.iterations = 0

    def get_strategy(self, info_set: str) -> np.ndarray:
        """Current strategy from regret matching."""
        return regret_match(self.regret_sum[info_set])

    def get_average_strategy(self, info_set: str) -> np.ndarray:
        """Average strategy over all iterations."""
        total = self.strategy_sum[info_set].sum()
        if total > 0:
            return self.strategy_sum[info_set] / total
        return np.ones(self.num_actions, dtype=np.float64) / self.num_actions

    def cfr(self, state, reach_probs: np.ndarray) -> np.ndarray:
        """Recursive CFR traversal.

        Args:
            state: Current game state.
            reach_probs: Reach probabilities for each player.

        Returns:
            Expected utility for each player.
        """
        if state.is_terminal:
            return np.array([state.payoff(p) for p in range(2)], dtype=np.float64)

        player = state.current_player
        info_set = state.info_set
        actions = state.legal_actions()
        n_actions = len(actions)
        action_indices = [int(a) for a in actions]

        # Get strategy using actual action indices (not positional slicing)
        raw_strategy = self.get_strategy(info_set)
        strategy = raw_strategy[action_indices]
        total = strategy.sum()
        if total > 0:
            strategy = strategy / total
        else:
            strategy = np.ones(n_actions, dtype=np.float64) / n_actions

        # Accumulate weighted strategy for average computation
        for i, a_idx in enumerate(action_indices):
            self.strategy_sum[info_set][a_idx] += reach_probs[player] * strategy[i]

        # Compute utility for each action
        action_utils = np.zeros((n_actions, 2), dtype=np.float64)
        node_util = np.zeros(2, dtype=np.float64)

        for i, action in enumerate(actions):
            next_state = state.apply(action)
            new_reach = reach_probs.copy()
            new_reach[player] *= strategy[i]
            action_utils[i] = self.cfr(next_state, new_reach)
            node_util += strategy[i] * action_utils[i]

        # Compute and accumulate regrets using actual action indices
        opponent = 1 - player
        for i, a_idx in enumerate(action_indices):
            regret = action_utils[i][player] - node_util[player]
            self.regret_sum[info_set][a_idx] += reach_probs[opponent] * regret

        return node_util

    def train(self, deal_fn, num_iterations: int) -> list[float]:
        """Run CFR+ for given iterations.

        Args:
            deal_fn: Function that returns all possible initial states.
            num_iterations: Number of iterations.

        Returns:
            List of average game values per iteration.
        """
        game_values = []
        all_deals = deal_fn()

        for _ in range(num_iterations):
            total_value = 0.0
            for state in all_deals:
                reach = np.ones(2, dtype=np.float64)
                util = self.cfr(state, reach)
                total_value += util[0]
            # CFR+: floor negative regrets to zero (once per iteration)
            for info_set in self.regret_sum:
                self.regret_sum[info_set] = np.maximum(self.regret_sum[info_set], 0)
            game_values.append(total_value / len(all_deals))
            self.iterations += 1

        return game_values
