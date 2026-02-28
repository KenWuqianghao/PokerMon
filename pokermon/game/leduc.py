"""Leduc Hold'em — validation game with ~936 info sets.

6-card deck: {J, Q, K} × {suit0, suit1}
2 players, ante of 1 chip each
2 betting rounds: preflop (1 private card), flop (1 community card)
Actions: FOLD, CHECK/CALL, RAISE (fixed size: 2 in round 1, 4 in round 2)
Max 2 raises per round.
If cards match community, that's a pair (best hand).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class LeducAction(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE = 2


NUM_LEDUC_ACTIONS = 3

# Card values: 0=J, 1=Q, 2=K (rank only; suits just for deck diversity)
LEDUC_RANKS = 3
LEDUC_SUITS = 2
LEDUC_DECK_SIZE = LEDUC_RANKS * LEDUC_SUITS  # 6 cards


@dataclass(frozen=True)
class LeducState:
    """Leduc Hold'em state.

    cards: (player0_card, player1_card, community_card)
        Card is 0-5; rank = card // 2, suit = card % 2
        community_card = -1 if not yet dealt
    history: tuple of tuples of actions per round
    bets: (player0_total_bet, player1_total_bet)
    round: 0 = preflop, 1 = flop
    raises_this_round: number of raises in current round
    """

    cards: tuple[int, int, int]  # (p0, p1, community)
    history: tuple[tuple[int, ...], ...] = ((),)
    bets: tuple[int, int] = (1, 1)  # Ante of 1 each
    current_round: int = 0
    raises_this_round: int = 0

    @staticmethod
    def card_rank(card: int) -> int:
        return card // LEDUC_SUITS

    @property
    def is_terminal(self) -> bool:
        h = self.history
        if len(h) == 0:
            return False
        for round_h in h:
            for a in round_h:
                if a == LeducAction.FOLD:
                    return True
        # Terminal if we've completed 2 rounds of betting
        if self.current_round >= 2:
            return True
        return False

    @property
    def current_player(self) -> int:
        """Determine whose turn it is."""
        actions_this_round = self.history[-1] if self.history else ()
        return len(actions_this_round) % 2

    @property
    def info_set(self) -> str:
        """Info set: player's card rank + community card rank (if dealt) + history."""
        player = self.current_player
        my_card = self.card_rank(self.cards[player])
        card_str = "JQK"[my_card]

        if self.current_round > 0 and self.cards[2] >= 0:
            comm_rank = self.card_rank(self.cards[2])
            card_str += "JQK"[comm_rank]

        # Encode history as string
        action_chars = {0: "f", 1: "c", 2: "r"}
        history_str = ""
        for round_h in self.history:
            for a in round_h:
                history_str += action_chars.get(a, "?")
            if round_h and round_h != self.history[-1]:
                history_str += "|"

        return card_str + ":" + history_str

    def payoff(self, player: int) -> float:
        """Terminal payoff for player."""
        assert self.is_terminal

        # Check for fold
        for round_h in self.history:
            for i, a in enumerate(round_h):
                if a == LeducAction.FOLD:
                    folder = i % 2
                    # The other player wins
                    winner = 1 - folder
                    return float(self.bets[1 - player]) if player == winner else -float(self.bets[player])

        # Showdown
        p0_rank = self.card_rank(self.cards[0])
        p1_rank = self.card_rank(self.cards[1])
        comm_rank = self.card_rank(self.cards[2])

        # Pair with community card is best
        p0_pair = p0_rank == comm_rank
        p1_pair = p1_rank == comm_rank

        if p0_pair and not p1_pair:
            winner = 0
        elif p1_pair and not p0_pair:
            winner = 1
        elif p0_rank > p1_rank:
            winner = 0
        elif p1_rank > p0_rank:
            winner = 1
        else:
            # Tie — split pot
            return 0.0

        if player == winner:
            return float(self.bets[1 - player])  # Win opponent's bet
        else:
            return -float(self.bets[player])

    def legal_actions(self) -> list[LeducAction]:
        """Legal actions for current player."""
        if self.is_terminal:
            return []

        actions = [LeducAction.CHECK_CALL]

        # Can fold if facing a raise
        round_h = self.history[-1] if self.history else ()
        facing_raise = any(a == LeducAction.RAISE for a in round_h) and (
            len(round_h) == 0 or round_h[-1] == LeducAction.RAISE
        )

        if facing_raise:
            actions.insert(0, LeducAction.FOLD)

        # Can raise if < 2 raises this round
        if self.raises_this_round < 2:
            actions.append(LeducAction.RAISE)

        return actions

    def apply(self, action: LeducAction) -> LeducState:
        """Apply action and return new state."""
        round_h = list(self.history[-1]) if self.history else []
        round_h.append(int(action))

        new_bets = list(self.bets)
        new_raises = self.raises_this_round
        new_round = self.current_round

        if action == LeducAction.RAISE:
            # Raise amount: 2 in round 0, 4 in round 1
            raise_amount = 2 if self.current_round == 0 else 4
            player = self.current_player
            # First match the opponent's bet, then raise
            opponent_bet = self.bets[1 - player]
            call_amount = max(0, opponent_bet - self.bets[player])
            new_bets[player] = self.bets[player] + call_amount + raise_amount
            new_raises += 1

        elif action == LeducAction.CHECK_CALL:
            player = self.current_player
            opponent_bet = self.bets[1 - player]
            call_amount = max(0, opponent_bet - self.bets[player])
            new_bets[player] = self.bets[player] + call_amount

        # Check if round is complete
        round_complete = False
        if action == LeducAction.FOLD:
            round_complete = True  # Hand over
        elif len(round_h) >= 2:
            # At least 2 actions this round
            if action == LeducAction.CHECK_CALL:
                round_complete = True

        new_history = list(self.history)
        new_history[-1] = tuple(round_h)

        if round_complete and action != LeducAction.FOLD:
            if self.current_round == 0:
                # Move to round 1 (flop)
                new_round = 1
                new_raises = 0
                new_history.append(())
            else:
                # Showdown
                new_round = 2

        return LeducState(
            cards=self.cards,
            history=tuple(new_history),
            bets=tuple(new_bets),
            current_round=new_round,
            raises_this_round=new_raises,
        )


def all_leduc_deals() -> list[LeducState]:
    """Generate all possible Leduc deals (30 total: 6×5×4/? ... actually permutations)."""
    deals = []
    for p0 in range(LEDUC_DECK_SIZE):
        for p1 in range(LEDUC_DECK_SIZE):
            if p1 == p0:
                continue
            for comm in range(LEDUC_DECK_SIZE):
                if comm == p0 or comm == p1:
                    continue
                deals.append(LeducState(cards=(p0, p1, comm)))
    return deals


def all_leduc_deals_preflop() -> list[LeducState]:
    """Generate all preflop Leduc deals (community card not yet dealt)."""
    deals = []
    for p0 in range(LEDUC_DECK_SIZE):
        for p1 in range(LEDUC_DECK_SIZE):
            if p1 == p0:
                continue
            deals.append(LeducState(cards=(p0, p1, -1)))
    return deals
