"""7-action discrete action space for NLHE."""

from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    BET_033X = 2  # 33% pot
    BET_050X = 3  # 50% pot
    BET_075X = 4  # 75% pot
    BET_100X = 5  # 100% pot (pot-size)
    ALL_IN = 6

    def __repr__(self) -> str:
        return self.name


NUM_ACTIONS = 7

# Pot fractions for bet sizing actions
BET_FRACTIONS = {
    Action.BET_033X: 0.33,
    Action.BET_050X: 0.50,
    Action.BET_075X: 0.75,
    Action.BET_100X: 1.00,
}


def compute_bet_size(action: Action, pot: int, to_call: int, stack: int, min_raise: int) -> int:
    """Compute the actual chip amount for a bet/raise action.

    Args:
        action: The discrete action.
        pot: Current pot size (including pending bets on the street).
        to_call: Amount the player needs to call.
        stack: Player's remaining stack.
        min_raise: Minimum legal raise size (typically BB or last raise increment).

    Returns:
        Total chips the player puts in this action (including calling amount).
        Returns 0 for FOLD/CHECK_CALL (handled separately).
    """
    if action == Action.FOLD:
        return 0
    if action == Action.CHECK_CALL:
        return min(to_call, stack)
    if action == Action.ALL_IN:
        return stack

    # Bet sizing: fraction of (pot + to_call) as the raise portion
    frac = BET_FRACTIONS[action]
    pot_after_call = pot + to_call
    raise_amount = max(int(pot_after_call * frac), min_raise)
    total = to_call + raise_amount
    # Clamp to stack
    return min(total, stack)


def legal_actions_mask(
    stack: int, to_call: int, pot: int, min_raise: int, can_check: bool
) -> list[bool]:
    """Return a boolean mask of length NUM_ACTIONS for legal actions.

    Args:
        stack: Player's current stack.
        to_call: Amount needed to call.
        pot: Current pot (including bets on this street).
        min_raise: Minimum raise increment.
        can_check: Whether checking is allowed (no pending bet).
    """
    mask = [False] * NUM_ACTIONS

    if stack == 0:
        # Already all-in; no actions possible (shouldn't be called)
        return mask

    # Fold is always legal when facing a bet
    if to_call > 0:
        mask[Action.FOLD] = True

    # Check/Call is always legal
    mask[Action.CHECK_CALL] = True

    # Bet/raise actions: legal if player has enough chips beyond calling
    for action, frac in BET_FRACTIONS.items():
        pot_after_call = pot + to_call
        raise_amount = max(int(pot_after_call * frac), min_raise)
        total = to_call + raise_amount
        if total < stack:  # Must have more chips than the bet (otherwise it's all-in)
            mask[action] = True

    # All-in is always legal if player has chips
    if stack > 0:
        mask[Action.ALL_IN] = True

    return mask
