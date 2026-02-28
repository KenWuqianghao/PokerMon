"""NLHE engine: state transitions, side pots, showdown."""

from __future__ import annotations

from dataclasses import replace

from pokermon.game.action import (
    Action,
    NUM_ACTIONS,
    compute_bet_size,
    legal_actions_mask,
)
from pokermon.game.deck import Deck
from pokermon.game.hand_eval import evaluate
from pokermon.game.state import GameState, PlayerState, Street


def new_hand(
    num_players: int = 6,
    stacks: list[int] | None = None,
    small_blind: int = 50,
    big_blind: int = 100,
    button: int = 0,
    deck: Deck | None = None,
) -> GameState:
    """Deal a new hand and post blinds.

    Args:
        num_players: Number of players (2-6).
        stacks: Starting stacks. Defaults to 10000 (100bb) each.
        small_blind: Small blind amount.
        big_blind: Big blind amount.
        button: Dealer button position.
        deck: Pre-seeded deck (will be shuffled). Creates new one if None.

    Returns:
        Initial GameState after blinds posted and hole cards dealt.
    """
    if num_players < 2 or num_players > 6:
        raise ValueError(f"num_players must be 2-6, got {num_players}")

    if stacks is None:
        stacks = [100 * big_blind] * num_players

    if len(stacks) != num_players:
        raise ValueError(f"stacks length {len(stacks)} != num_players {num_players}")

    if deck is None:
        deck = Deck()
    deck.shuffle()

    # In heads-up, button IS the small blind
    if num_players == 2:
        sb_pos = button
        bb_pos = (button + 1) % num_players
    else:
        sb_pos = (button + 1) % num_players
        bb_pos = (button + 2) % num_players

    # Deal hole cards
    all_hole_cards = []
    for _ in range(num_players):
        all_hole_cards.append(tuple(deck.deal(2)))

    # Store remaining deck for community cards
    deck_cards = tuple(deck.deal(deck.remaining()))

    # Create player states with blinds posted
    players = []
    for i in range(num_players):
        stack = stacks[i]
        bet = 0
        if i == sb_pos:
            bet = min(small_blind, stack)
        elif i == bb_pos:
            bet = min(big_blind, stack)
        players.append(
            PlayerState(
                stack=stack - bet,
                bet=bet,
                total_bet=bet,
                folded=False,
                all_in=(stack - bet == 0),
                hole_cards=all_hole_cards[i],
            )
        )

    # First to act preflop: UTG (left of BB)
    # For heads-up: button/SB acts first preflop
    if num_players == 2:
        first_to_act = button  # SB = button in heads-up
    else:
        first_to_act = (bb_pos + 1) % num_players

    first_to_act = _next_active_player(first_to_act, num_players, players)

    state = GameState(
        num_players=num_players,
        players=tuple(players),
        street=Street.PREFLOP,
        community_cards=(),
        pot=0,
        current_player=first_to_act,
        button=button,
        small_blind=small_blind,
        big_blind=big_blind,
        min_raise=big_blind,
        history=((),),
        is_terminal=False,
        last_raiser=bb_pos,
        num_actions_this_street=0,
        initial_stacks=tuple(stacks),
        deck_cards=deck_cards,
    )

    # If everyone is all-in from blinds, run to showdown
    if len(state.players_with_chips) <= 1 and len(state.active_players) >= 2:
        state = _run_to_showdown(state)

    return state


def apply_action(state: GameState, action: Action) -> GameState:
    """Apply an action and return the new state."""
    if state.is_terminal:
        raise ValueError("Cannot apply action to terminal state")

    player_idx = state.current_player
    player = state.players[player_idx]

    max_bet = max(p.bet for p in state.players)
    to_call = max_bet - player.bet

    # Validate
    can_check = to_call == 0
    mask = legal_actions_mask(
        player.stack, to_call, _street_pot(state), state.min_raise, can_check
    )
    if not mask[action]:
        raise ValueError(
            f"Illegal action {action.name} for player {player_idx}. "
            f"Legal: {[Action(i).name for i, v in enumerate(mask) if v]}"
        )

    new_players = list(state.players)
    new_min_raise = state.min_raise
    new_last_raiser = state.last_raiser

    if action == Action.FOLD:
        new_players[player_idx] = replace(player, folded=True)

    elif action == Action.CHECK_CALL:
        call_amount = min(to_call, player.stack)
        new_players[player_idx] = replace(
            player,
            stack=player.stack - call_amount,
            bet=player.bet + call_amount,
            total_bet=player.total_bet + call_amount,
            all_in=(player.stack - call_amount == 0),
        )

    elif action == Action.ALL_IN:
        all_in_amount = player.stack
        raise_amount = (player.bet + all_in_amount) - max_bet
        if raise_amount > 0:
            new_min_raise = max(state.min_raise, raise_amount)
            new_last_raiser = player_idx
        new_players[player_idx] = replace(
            player,
            stack=0,
            bet=player.bet + all_in_amount,
            total_bet=player.total_bet + all_in_amount,
            all_in=True,
        )

    else:
        # Bet/raise
        bet_amount = compute_bet_size(
            action, _street_pot(state), to_call, player.stack, state.min_raise
        )
        raise_portion = bet_amount - to_call
        new_min_raise = max(state.min_raise, raise_portion)
        new_last_raiser = player_idx
        new_players[player_idx] = replace(
            player,
            stack=player.stack - bet_amount,
            bet=player.bet + bet_amount,
            total_bet=player.total_bet + bet_amount,
            all_in=(bet_amount == player.stack),
        )

    # Update history
    new_history = list(state.history)
    street_history = list(new_history[-1])
    street_history.append(int(action))
    new_history[-1] = tuple(street_history)

    new_state = replace(
        state,
        players=tuple(new_players),
        min_raise=new_min_raise,
        history=tuple(new_history),
        last_raiser=new_last_raiser,
        num_actions_this_street=state.num_actions_this_street + 1,
    )

    # Only one player left → they win
    active = new_state.active_players
    if len(active) == 1:
        return _finish_hand_fold(new_state, active[0])

    # Street complete?
    if _is_street_complete(new_state, player_idx):
        return _advance_street(new_state)

    # Next player
    next_player = _next_active_player(
        (player_idx + 1) % state.num_players, state.num_players, list(new_state.players)
    )
    return replace(new_state, current_player=next_player)


def get_legal_actions(state: GameState) -> list[Action]:
    """Return list of legal actions for the current player."""
    if state.is_terminal:
        return []
    return [Action(i) for i, v in enumerate(get_legal_actions_mask(state)) if v]


def get_legal_actions_mask(state: GameState) -> list[bool]:
    """Return boolean mask of legal actions for the current player."""
    if state.is_terminal:
        return [False] * NUM_ACTIONS
    player = state.players[state.current_player]
    max_bet = max(p.bet for p in state.players)
    to_call = max_bet - player.bet
    can_check = to_call == 0
    return legal_actions_mask(player.stack, to_call, _street_pot(state), state.min_raise, can_check)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _street_pot(state: GameState) -> int:
    """Total pot including current street bets."""
    return state.pot + sum(p.bet for p in state.players)


def _next_active_player(start: int, num_players: int, players: list[PlayerState]) -> int:
    """Find next player who is not folded and not all-in."""
    for offset in range(num_players):
        idx = (start + offset) % num_players
        if not players[idx].folded and not players[idx].all_in:
            return idx
    return start


def _is_street_complete(state: GameState, last_actor: int) -> bool:
    """Check if the current betting street is complete."""
    active_with_chips = state.players_with_chips

    # No one can act
    if len(active_with_chips) == 0:
        return True

    # All active players must have equal bets (or be all-in)
    active = state.active_players
    max_bet = max(state.players[i].bet for i in active)

    for i in active_with_chips:
        if state.players[i].bet < max_bet:
            return False

    # Must have enough actions taken
    if state.num_actions_this_street < len(active_with_chips):
        return False

    # If the next-to-act would be the last raiser (or no raiser), round is done
    next_player = _next_active_player(
        (last_actor + 1) % state.num_players,
        state.num_players,
        list(state.players),
    )

    # Everyone has matched and we've gone around
    if next_player == state.last_raiser or state.last_raiser == -1:
        return True

    # All bets equal and enough actions
    return all(state.players[i].bet == max_bet for i in active_with_chips)


def _advance_street(state: GameState) -> GameState:
    """Collect bets and deal community cards for next street."""
    collected = sum(p.bet for p in state.players)
    new_pot = state.pot + collected
    new_players = tuple(replace(p, bet=0) for p in state.players)

    new_street = Street(state.street + 1)

    if new_street == Street.SHOWDOWN:
        return _showdown(replace(state, players=new_players, pot=new_pot, street=Street.SHOWDOWN))

    # Deal community cards
    new_community = list(state.community_cards)
    if new_street == Street.FLOP:
        new_community.extend(state.deck_cards[0:3])
    elif new_street == Street.TURN:
        new_community.append(state.deck_cards[3])
    elif new_street == Street.RIVER:
        new_community.append(state.deck_cards[4])

    # First to act post-flop: first active left of dealer
    first_to_act = _next_active_player(
        (state.button + 1) % state.num_players,
        state.num_players,
        list(new_players),
    )

    new_state = replace(
        state,
        players=new_players,
        street=new_street,
        community_cards=tuple(new_community),
        pot=new_pot,
        current_player=first_to_act,
        min_raise=state.big_blind,
        history=tuple(list(state.history) + [()]),
        last_raiser=-1,
        num_actions_this_street=0,
    )

    # If <= 1 player can act, run to showdown
    if len(new_state.players_with_chips) <= 1 and len(new_state.active_players) >= 2:
        return _run_to_showdown(new_state)

    return new_state


def _run_to_showdown(state: GameState) -> GameState:
    """Deal remaining community cards and run showdown."""
    new_community = list(state.community_cards)
    if len(new_community) < 3:
        new_community.extend(state.deck_cards[0:3])
    if len(new_community) < 4:
        new_community.append(state.deck_cards[3])
    if len(new_community) < 5:
        new_community.append(state.deck_cards[4])

    collected = sum(p.bet for p in state.players)
    new_pot = state.pot + collected
    new_players = tuple(replace(p, bet=0) for p in state.players)

    return _showdown(replace(
        state,
        players=new_players,
        street=Street.SHOWDOWN,
        community_cards=tuple(new_community),
        pot=new_pot,
    ))


def _showdown(state: GameState) -> GameState:
    """Evaluate hands and distribute pot(s)."""
    active = state.active_players

    if len(active) == 1:
        return _finish_hand_fold(state, active[0])

    # Evaluate hands
    hand_scores = {}
    for i in active:
        hand_scores[i] = evaluate(list(state.players[i].hole_cards), list(state.community_cards))

    # Compute side pots and distribute
    side_pots = _compute_side_pots(state)
    winnings = [0] * state.num_players

    for pot_amount, eligible in side_pots:
        eligible_active = [i for i in eligible if i in hand_scores]
        if not eligible_active:
            continue
        best_score = min(hand_scores[i] for i in eligible_active)
        winners = [i for i in eligible_active if hand_scores[i] == best_score]
        share = pot_amount // len(winners)
        remainder = pot_amount % len(winners)
        for j, w in enumerate(winners):
            winnings[w] += share + (1 if j < remainder else 0)

    payoffs = tuple(winnings[i] - state.players[i].total_bet for i in range(state.num_players))
    new_players = tuple(
        replace(state.players[i], stack=state.players[i].stack + winnings[i])
        for i in range(state.num_players)
    )

    return replace(
        state,
        players=new_players,
        is_terminal=True,
        payoffs=payoffs,
        current_player=-1,
        pot=0,
    )


def _finish_hand_fold(state: GameState, winner: int) -> GameState:
    """End hand when all but one player has folded."""
    total_pot = state.pot + sum(p.bet for p in state.players)

    winnings = [0] * state.num_players
    winnings[winner] = total_pot

    payoffs = tuple(winnings[i] - state.players[i].total_bet for i in range(state.num_players))
    new_players = tuple(
        replace(state.players[i], stack=state.players[i].stack + (total_pot if i == winner else 0), bet=0)
        for i in range(state.num_players)
    )

    return replace(
        state,
        players=new_players,
        is_terminal=True,
        payoffs=payoffs,
        current_player=-1,
        pot=0,
    )


def _compute_side_pots(state: GameState) -> list[tuple[int, list[int]]]:
    """Compute side pots for showdown.

    Returns list of (pot_amount, eligible_player_indices) from smallest to largest.
    """
    active = [i for i in range(state.num_players) if not state.players[i].folded]
    contributions = {i: state.players[i].total_bet for i in range(state.num_players)}

    # All contributions including folded players
    all_contribs = [contributions[i] for i in range(state.num_players)]

    # Unique contribution levels from active players (sorted)
    levels = sorted(set(contributions[i] for i in active))

    side_pots: list[tuple[int, list[int]]] = []
    prev_level = 0

    for level in levels:
        if level <= prev_level:
            continue
        pot_amount = 0
        eligible = []
        for i in range(state.num_players):
            contrib_in_range = min(contributions[i], level) - min(contributions[i], prev_level)
            pot_amount += max(0, contrib_in_range)
        # Eligible = active players who contributed at least this level
        eligible = [i for i in active if contributions[i] >= level]
        if pot_amount > 0:
            side_pots.append((pot_amount, eligible))
        prev_level = level

    # Add any main pot not covered by contributions (shouldn't happen, but safety)
    side_pot_total = sum(amount for amount, _ in side_pots)
    remaining = state.pot + sum(p.bet for p in state.players) - side_pot_total
    # The pot field should be 0 at showdown since we collected, but handle edge cases
    if not side_pots:
        side_pots = [(state.pot, active)]

    return side_pots
