"""Kuhn poker — minimal validation game.

3 cards (J, Q, K), 2 players, ante of 1 chip each.
Actions: CHECK, BET (1 chip).
One betting round. Higher card wins at showdown.

This is a perfect testbed for CFR convergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class KuhnAction(IntEnum):
    CHECK = 0
    BET = 1


NUM_KUHN_ACTIONS = 2


@dataclass(frozen=True)
class KuhnState:
    """Kuhn poker state.

    cards: (player0_card, player1_card) — values 0(J), 1(Q), 2(K)
    history: string of actions ('c'=check, 'b'=bet)
    """

    cards: tuple[int, int]
    history: str = ""

    @property
    def is_terminal(self) -> bool:
        h = self.history
        # Terminal states: cc (check-check), cbc (check-bet-call),
        # cbf (check-bet-fold), bc (bet-call), bf (bet-fold), bb->bc
        return h in ("cc", "cbc", "cbf", "bc", "bf")

    @property
    def current_player(self) -> int:
        return len(self.history) % 2

    @property
    def info_set(self) -> str:
        """Information set: player's card + action history."""
        card = self.cards[self.current_player]
        card_str = "JQK"[card]
        return card_str + self.history

    def payoff(self, player: int) -> float:
        """Terminal payoff for player (ante = 1 each)."""
        assert self.is_terminal
        h = self.history

        if h == "bf":
            # Player 0 bet, player 1 fold → player 0 wins ante (1)
            return 1.0 if player == 0 else -1.0

        if h == "cbf":
            # Player 0 check, player 1 bet, player 0 fold → player 1 wins ante
            return -1.0 if player == 0 else 1.0

        # Showdown cases: cc, bc, cbc
        winner = 0 if self.cards[0] > self.cards[1] else 1

        if h == "cc":
            # Check-check showdown: winner gets 1
            return 1.0 if player == winner else -1.0

        if h in ("bc", "cbc"):
            # Bet-call showdown: winner gets 2
            return 2.0 if player == winner else -2.0

        raise ValueError(f"Unknown terminal history: {h}")

    def legal_actions(self) -> list[KuhnAction]:
        """Return legal actions."""
        if self.is_terminal:
            return []

        h = self.history
        if h == "" or h == "c":
            # First action or after check: can check or bet
            return [KuhnAction.CHECK, KuhnAction.BET]
        if h == "b" or h == "cb":
            # Facing a bet: fold (check=fold) or call (bet=call)
            # In Kuhn, after a bet, the other player can fold or call
            # We represent fold as CHECK and call as BET
            return [KuhnAction.CHECK, KuhnAction.BET]

        return []

    def apply(self, action: KuhnAction) -> KuhnState:
        """Apply action and return new state."""
        a_str = "c" if action == KuhnAction.CHECK else "b"
        # In Kuhn: after a bet, CHECK means FOLD and BET means CALL
        if self.history in ("b", "cb"):
            a_str = "f" if action == KuhnAction.CHECK else "c"
        return KuhnState(cards=self.cards, history=self.history + a_str)


def deal_kuhn(seed: int | None = None) -> KuhnState:
    """Deal a Kuhn poker hand."""
    import numpy as np

    rng = np.random.RandomState(seed)
    cards = rng.choice(3, size=2, replace=False)
    return KuhnState(cards=(int(cards[0]), int(cards[1])))


def all_kuhn_deals() -> list[KuhnState]:
    """Generate all 6 possible Kuhn deals."""
    deals = []
    for c0 in range(3):
        for c1 in range(3):
            if c0 != c1:
                deals.append(KuhnState(cards=(c0, c1)))
    return deals
