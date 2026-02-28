"""Suit isomorphism canonicalization.

In poker, suits are interchangeable (no suit is inherently better).
We canonicalize suits to reduce the state space: the first suit seen
gets mapped to suit 0, the second to suit 1, etc.
"""

from __future__ import annotations

from pokermon.game.card import NUM_SUITS, card_from_index, card_index


def canonicalize_suits(
    hole_cards: list[int], community_cards: list[int]
) -> tuple[list[int], list[int]]:
    """Canonicalize suits so the first suit encountered is always 0.

    This reduces the number of distinct info sets by a factor of up to 4!.

    Args:
        hole_cards: Player's hole card indices.
        community_cards: Community card indices.

    Returns:
        (canonicalized_hole, canonicalized_community)
    """
    suit_map: dict[int, int] = {}
    next_suit = 0

    def map_card(card_idx: int) -> int:
        nonlocal next_suit
        rank, suit = card_from_index(card_idx)
        if suit not in suit_map:
            if next_suit < NUM_SUITS:
                suit_map[suit] = next_suit
                next_suit += 1
            else:
                suit_map[suit] = suit  # Shouldn't happen with 4 suits
        return card_index(rank, suit_map[suit])

    # Process hole cards first (player's private cards define the canonical mapping)
    canon_hole = [map_card(c) for c in hole_cards]
    canon_community = [map_card(c) for c in community_cards]

    return canon_hole, canon_community
