"""Baseline agents: random, call-station, fold-bot."""

from __future__ import annotations

import numpy as np

from pokermon.game.action import Action
from pokermon.game.engine import get_legal_actions
from pokermon.game.state import GameState


class BaseAgent:
    """Base class for poker agents."""

    def act(self, state: GameState, player: int) -> Action:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """Plays a random legal action."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.RandomState(seed)

    def act(self, state: GameState, player: int) -> Action:
        actions = get_legal_actions(state)
        return self.rng.choice(actions)


class CallStation(BaseAgent):
    """Always checks or calls. Never folds, never raises."""

    def act(self, state: GameState, player: int) -> Action:
        return Action.CHECK_CALL


class FoldBot(BaseAgent):
    """Folds whenever possible, otherwise checks/calls."""

    def act(self, state: GameState, player: int) -> Action:
        actions = get_legal_actions(state)
        if Action.FOLD in actions:
            return Action.FOLD
        return Action.CHECK_CALL


class AggressiveBot(BaseAgent):
    """Always raises or goes all-in when possible."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.RandomState(seed)

    def act(self, state: GameState, player: int) -> Action:
        actions = get_legal_actions(state)
        # Prefer biggest bet size
        for action in [Action.ALL_IN, Action.BET_100X, Action.BET_075X, Action.BET_050X, Action.BET_033X]:
            if action in actions:
                return action
        return Action.CHECK_CALL
