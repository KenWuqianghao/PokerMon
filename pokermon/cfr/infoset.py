"""Information set encoding for Deep CFR.

Converts game states into fixed-size feature vectors suitable for neural networks.
Includes suit canonicalization for reduced state space.
"""

from __future__ import annotations

import numpy as np

from pokermon.game.action import NUM_ACTIONS
from pokermon.game.card import NUM_RANKS, NUM_SUITS, card_from_index
from pokermon.game.state import GameState, Street
from pokermon.utils.cards import canonicalize_suits


# Feature dimensions
CARD_EMBED_DIM = 20  # rank(13) + suit(4) + padding → projected to 20
PRIVATE_DIM = 40  # 2 cards × 20d
COMMUNITY_DIM = 20  # sum-pooled 20d
HISTORY_DIM = 128  # compressed betting history
META_DIM = 23  # pot(1) + stacks(6) + position(6) + street(4) + player(6)
TOTAL_DIM = PRIVATE_DIM + COMMUNITY_DIM + HISTORY_DIM + META_DIM  # 211

# Betting history: 4 streets × 12 slots × 14 features = 672 → Linear → 128
MAX_STREETS = 4
MAX_ACTIONS_PER_STREET = 12
HISTORY_FEATURES_PER_ACTION = 14  # one-hot action(7) + player(6) + amount(1)
RAW_HISTORY_DIM = MAX_STREETS * MAX_ACTIONS_PER_STREET * HISTORY_FEATURES_PER_ACTION  # 672


def encode_infoset(state: GameState, player: int) -> dict[str, np.ndarray]:
    """Encode a game state from a player's perspective into feature vectors.

    Returns a dict of named feature arrays for the neural network:
    - 'private_cards': (2, 2) — rank and suit indices for hole cards
    - 'community_cards': (5, 2) — rank and suit indices (padded with -1)
    - 'history': (672,) — flattened betting history
    - 'meta': (23,) — pot, stacks, position, street, player one-hot
    - 'legal_mask': (7,) — boolean mask of legal actions
    """
    p = state.players[player]

    # Private cards (canonicalized suits)
    hole = list(p.hole_cards)
    community = list(state.community_cards)
    hole_canon, community_canon = canonicalize_suits(hole, community)

    private_cards = np.full((2, 2), -1, dtype=np.int32)
    for i, c in enumerate(hole_canon):
        rank, suit = card_from_index(c)
        private_cards[i] = [rank, suit]

    # Community cards (padded)
    community_cards_arr = np.full((5, 2), -1, dtype=np.int32)
    for i, c in enumerate(community_canon):
        rank, suit = card_from_index(c)
        community_cards_arr[i] = [rank, suit]

    # Betting history
    history = _encode_history(state)

    # Meta features
    meta = _encode_meta(state, player)

    # Legal actions mask
    from pokermon.game.engine import get_legal_actions_mask
    legal_mask = np.array(get_legal_actions_mask(state), dtype=np.float32)

    return {
        "private_cards": private_cards,
        "community_cards": community_cards_arr,
        "history": history,
        "meta": meta,
        "legal_mask": legal_mask,
    }


def encode_infoset_flat(state: GameState, player: int) -> np.ndarray:
    """Encode info set as a single flat feature vector of dimension TOTAL_DIM.

    For simpler networks (MLP input). Uses raw numeric features.
    """
    p = state.players[player]
    features = []

    # Private cards: one-hot rank (13) + one-hot suit (4) × 2 cards = 34
    hole = list(p.hole_cards)
    community = list(state.community_cards)
    hole_canon, community_canon = canonicalize_suits(hole, community)

    for c in hole_canon:
        rank, suit = card_from_index(c)
        r_oh = np.zeros(NUM_RANKS, dtype=np.float32)
        r_oh[rank] = 1.0
        s_oh = np.zeros(NUM_SUITS, dtype=np.float32)
        s_oh[suit] = 1.0
        features.extend([r_oh, s_oh])
    # Pad to 40d (2 × 20, but we have 2 × 17 = 34; add 6 zeros)
    features.append(np.zeros(PRIVATE_DIM - 2 * (NUM_RANKS + NUM_SUITS), dtype=np.float32))

    # Community cards: sum of one-hot encodings
    comm_rank = np.zeros(NUM_RANKS, dtype=np.float32)
    comm_suit = np.zeros(NUM_SUITS, dtype=np.float32)
    for c in community_canon:
        rank, suit = card_from_index(c)
        comm_rank[rank] += 1.0
        comm_suit[suit] += 1.0
    features.extend([comm_rank, comm_suit])
    # Pad to 20d
    features.append(np.zeros(COMMUNITY_DIM - NUM_RANKS - NUM_SUITS, dtype=np.float32))

    # History (compressed to HISTORY_DIM via raw features)
    history = _encode_history(state)
    features.append(history[:HISTORY_DIM])

    # Meta
    meta = _encode_meta(state, player)
    features.append(meta)

    flat = np.concatenate(features)
    # Ensure exact dimension
    if len(flat) < TOTAL_DIM:
        flat = np.concatenate([flat, np.zeros(TOTAL_DIM - len(flat), dtype=np.float32)])
    return flat[:TOTAL_DIM]


def _encode_history(state: GameState) -> np.ndarray:
    """Encode betting history as a flat vector."""
    history = np.zeros(RAW_HISTORY_DIM, dtype=np.float32)

    for street_idx, street_actions in enumerate(state.history):
        if street_idx >= MAX_STREETS:
            break
        for action_idx, action in enumerate(street_actions):
            if action_idx >= MAX_ACTIONS_PER_STREET:
                break
            offset = (street_idx * MAX_ACTIONS_PER_STREET + action_idx) * HISTORY_FEATURES_PER_ACTION
            # One-hot action (7 dims)
            if action < NUM_ACTIONS:
                history[offset + action] = 1.0
            # Player indicator (6 dims) — approximate from action sequence
            player_in_seq = action_idx % state.num_players
            if player_in_seq < 6:
                history[offset + 7 + player_in_seq] = 1.0
            # Amount placeholder (1 dim) — normalized by big blind
            history[offset + 13] = action / 6.0  # rough normalization

    return history[:HISTORY_DIM]  # Truncate to HISTORY_DIM


def _encode_meta(state: GameState, player: int) -> np.ndarray:
    """Encode meta-features: pot, stacks, position, street, player."""
    meta = np.zeros(META_DIM, dtype=np.float32)

    # Pot normalized by 100 BB
    bb = max(state.big_blind, 1)
    meta[0] = (state.pot + sum(p.bet for p in state.players)) / (100 * bb)

    # Stacks (6 slots, normalized)
    for i in range(min(state.num_players, 6)):
        meta[1 + i] = state.players[i].stack / (100 * bb)

    # Position relative to button (6 slots one-hot)
    pos = (player - state.button) % state.num_players
    if pos < 6:
        meta[7 + pos] = 1.0

    # Street one-hot (4 dims)
    if state.street < 4:
        meta[13 + state.street] = 1.0

    # Current player one-hot (6 dims)
    if player < 6:
        meta[17 + player] = 1.0

    return meta
