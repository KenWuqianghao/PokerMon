"""Immutable GameState dataclass for NLHE."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4


@dataclass(frozen=True)
class PlayerState:
    """Per-player state."""

    stack: int
    bet: int  # Amount bet on current street
    total_bet: int  # Total amount put in pot this hand
    folded: bool
    all_in: bool
    hole_cards: tuple[int, ...] = ()  # 2 cards, empty if mucked


@dataclass(frozen=True)
class GameState:
    """Immutable snapshot of a NLHE hand.

    All mutations produce a new GameState.
    """

    num_players: int
    players: tuple[PlayerState, ...]
    street: Street
    community_cards: tuple[int, ...]
    pot: int  # Main pot (chips from previous streets + collected bets)
    current_player: int  # Index of player to act (-1 if terminal)
    button: int  # Dealer button position
    small_blind: int
    big_blind: int
    min_raise: int  # Minimum raise size for current street
    history: tuple[tuple[int, ...], ...] = ()  # Per-street action history
    is_terminal: bool = False
    payoffs: tuple[int, ...] = ()  # Net profit/loss for each player (only if terminal)
    last_raiser: int = -1  # Last player who raised on this street
    num_actions_this_street: int = 0
    initial_stacks: tuple[int, ...] = ()  # Starting stacks for chip conservation check
    deck_cards: tuple[int, ...] = ()  # Remaining deck cards for community dealing

    @property
    def active_players(self) -> list[int]:
        """Players still in the hand (not folded)."""
        return [i for i in range(self.num_players) if not self.players[i].folded]

    @property
    def players_with_chips(self) -> list[int]:
        """Active players who still have chips to act."""
        return [
            i
            for i in range(self.num_players)
            if not self.players[i].folded and not self.players[i].all_in
        ]

    @property
    def total_chips(self) -> int:
        """Total chips in play (stacks + pot + bets). Should be constant."""
        return (
            self.pot
            + sum(p.stack for p in self.players)
            + sum(p.bet for p in self.players)
        )
