"""Multi-agent match runner."""

from __future__ import annotations

import numpy as np

from pokermon.eval.baselines import BaseAgent
from pokermon.game.action import Action
from pokermon.game.deck import Deck
from pokermon.game.engine import apply_action, get_legal_actions, new_hand


def run_match(
    agents: list[BaseAgent],
    num_hands: int = 1000,
    small_blind: int = 50,
    big_blind: int = 100,
    starting_stack: int = 10_000,
    seed: int = 42,
) -> dict:
    """Run a multi-agent match and collect statistics.

    Args:
        agents: List of agents (one per seat).
        num_hands: Number of hands to play.
        small_blind: Small blind amount.
        big_blind: Big blind amount.
        starting_stack: Starting stack for each player.
        seed: Random seed.

    Returns:
        Dict with results: total_profit, hands_played, per-player stats.
    """
    num_players = len(agents)
    total_profit = np.zeros(num_players, dtype=np.float64)
    hands_played = 0

    for hand_idx in range(num_hands):
        deck = Deck(seed=seed + hand_idx)
        button = hand_idx % num_players
        stacks = [starting_stack] * num_players

        state = new_hand(
            num_players=num_players,
            stacks=stacks,
            small_blind=small_blind,
            big_blind=big_blind,
            button=button,
            deck=deck,
        )

        # Play the hand
        max_actions = 200  # Safety limit
        action_count = 0
        while not state.is_terminal and action_count < max_actions:
            player = state.current_player
            action = agents[player].act(state, player)

            # Validate action is legal
            legal = get_legal_actions(state)
            if action not in legal:
                # Fallback to check/call
                action = Action.CHECK_CALL if Action.CHECK_CALL in legal else legal[0]

            state = apply_action(state, action)
            action_count += 1

        if state.is_terminal:
            for i in range(num_players):
                total_profit[i] += state.payoffs[i]
            hands_played += 1

    return {
        "num_players": num_players,
        "hands_played": hands_played,
        "total_profit": total_profit.tolist(),
        "avg_profit_per_hand": (total_profit / max(hands_played, 1)).tolist(),
        "bb_per_100": (total_profit / max(hands_played, 1) / big_blind * 100).tolist(),
    }
