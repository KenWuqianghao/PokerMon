"""External sampling MCCFR tree walk for Deep CFR.

The traversing player explores ALL actions while opponents sample ONE action each.
This makes multiplayer tractable.
"""

from __future__ import annotations

import numpy as np
import torch

from pokermon.cfr.infoset import TOTAL_DIM, encode_infoset_flat
from pokermon.cfr.regret_matching import regret_match_masked
from pokermon.game.action import NUM_ACTIONS, Action
from pokermon.game.engine import apply_action, get_legal_actions, get_legal_actions_mask
from pokermon.game.state import GameState


def external_sampling_mccfr(
    state: GameState,
    traverser: int,
    advantage_nets: list,  # List of advantage networks (one per player)
    iteration: int,
    advantage_memory: list,  # List of ReservoirBuffers (one per player)
    strategy_memory,  # Single ReservoirBuffer for strategy network
    device: torch.device = torch.device("cpu"),
    rng: np.random.RandomState | None = None,
    prune_threshold: float = -3e8,
    prune_after: int = 100,
) -> np.ndarray:
    """External sampling MCCFR traversal.

    Args:
        state: Current game state.
        traverser: Index of the traversing player (explores all actions).
        advantage_nets: Advantage networks for each player.
        iteration: Current training iteration (for weighting).
        advantage_memory: Reservoir buffers for advantage training data.
        strategy_memory: Reservoir buffer for strategy training data.
        device: Torch device for inference.
        rng: Random number generator.
        prune_threshold: Skip actions with advantage below this (after prune_after iterations).
        prune_after: Start pruning after this many iterations.

    Returns:
        Expected utility for the traversing player.
    """
    if rng is None:
        rng = np.random.RandomState()

    if state.is_terminal:
        return state.payoffs[traverser]

    current = state.current_player
    legal_actions = get_legal_actions(state)
    legal_mask = np.array(get_legal_actions_mask(state), dtype=np.float32)

    if len(legal_actions) == 0:
        return 0.0

    # Get features for current info set
    features = encode_infoset_flat(state, current)

    # Get strategy from advantage network
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        advantages = advantage_nets[current](x).squeeze(0).cpu().numpy()

    strategy = regret_match_masked(advantages, legal_mask)

    if current == traverser:
        # Traversing player: explore ALL actions
        action_utils = np.zeros(NUM_ACTIONS, dtype=np.float64)
        explored = np.zeros(NUM_ACTIONS, dtype=bool)

        for action in legal_actions:
            a = int(action)
            # Pruning: skip very bad actions after some iterations
            if iteration > prune_after and advantages[a] < prune_threshold:
                continue

            next_state = apply_action(state, action)
            action_utils[a] = external_sampling_mccfr(
                next_state, traverser, advantage_nets, iteration,
                advantage_memory, strategy_memory, device, rng,
                prune_threshold, prune_after,
            )
            explored[a] = True

        # Compute node value under current strategy
        node_value = np.sum(strategy * action_utils)

        # Compute advantages (regrets)
        regrets = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for action in legal_actions:
            a = int(action)
            if explored[a]:
                regrets[a] = action_utils[a] - node_value

        # Store in advantage memory with iteration weighting
        weight = iteration ** 1.5  # Linear CFR weighting
        advantage_memory[traverser].add(features, regrets, weight)

        return node_value

    else:
        # Opponent: sample ONE action according to their strategy
        action_probs = strategy[np.array([int(a) for a in legal_actions])]
        action_probs = action_probs / action_probs.sum() if action_probs.sum() > 0 else np.ones(len(legal_actions)) / len(legal_actions)
        chosen_idx = rng.choice(len(legal_actions), p=action_probs)
        chosen_action = legal_actions[chosen_idx]

        # Store in strategy memory
        weight = iteration ** 1.5
        strategy_memory.add(features, strategy, weight)

        next_state = apply_action(state, chosen_action)
        return external_sampling_mccfr(
            next_state, traverser, advantage_nets, iteration,
            advantage_memory, strategy_memory, device, rng,
            prune_threshold, prune_after,
        )


def traverse_kuhn(
    state,
    traverser: int,
    advantage_nets: list,
    iteration: int,
    advantage_memory: list,
    strategy_memory,
    device: torch.device = torch.device("cpu"),
    rng: np.random.RandomState | None = None,
) -> float:
    """Simplified traversal for Kuhn poker (used for Deep CFR validation).

    Uses the Kuhn game interface directly.
    """
    if rng is None:
        rng = np.random.RandomState()

    if state.is_terminal:
        return state.payoff(traverser)

    current = state.current_player
    actions = state.legal_actions()
    num_actions = len(actions)

    # Get strategy from advantage network (uses info_set string as feature)
    info_set = state.info_set
    features = _kuhn_infoset_to_features(info_set)

    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        advantages = advantage_nets[current](x).squeeze(0).cpu().numpy()[:num_actions]

    strategy = regret_match_masked(
        advantages, np.ones(num_actions, dtype=np.float32)
    )

    if current == traverser:
        action_utils = np.zeros(num_actions, dtype=np.float64)
        for i, action in enumerate(actions):
            next_state = state.apply(action)
            action_utils[i] = traverse_kuhn(
                next_state, traverser, advantage_nets, iteration,
                advantage_memory, strategy_memory, device, rng,
            )

        node_value = np.sum(strategy * action_utils)
        regrets = np.zeros(num_actions, dtype=np.float32)
        for i in range(num_actions):
            regrets[i] = action_utils[i] - node_value

        weight = max(iteration, 1) ** 1.5
        # Pad regrets to target_dim if needed
        padded_regrets = np.zeros(advantage_memory[traverser].target_dim, dtype=np.float32)
        padded_regrets[:num_actions] = regrets
        advantage_memory[traverser].add(features, padded_regrets, weight)

        return node_value
    else:
        # Sample action
        chosen_idx = rng.choice(num_actions, p=strategy)
        chosen_action = actions[chosen_idx]

        weight = max(iteration, 1) ** 1.5
        padded_strategy = np.zeros(strategy_memory.target_dim, dtype=np.float32)
        padded_strategy[:num_actions] = strategy
        strategy_memory.add(features, padded_strategy, weight)

        next_state = state.apply(chosen_action)
        return traverse_kuhn(
            next_state, traverser, advantage_nets, iteration,
            advantage_memory, strategy_memory, device, rng,
        )


# Kuhn info set feature encoding (simple one-hot for validation)
_KUHN_INFOSETS = [
    "J", "Jb", "Jc", "Jcb",
    "Q", "Qb", "Qc", "Qcb",
    "K", "Kb", "Kc", "Kcb",
]
KUHN_FEATURE_DIM = len(_KUHN_INFOSETS)


def traverse_leduc(
    state,
    traverser: int,
    advantage_nets: list,
    iteration: int,
    advantage_memory: list,
    strategy_memory,
    device: torch.device = torch.device("cpu"),
    rng: np.random.RandomState | None = None,
) -> float:
    """Simplified traversal for Leduc Hold'em (Deep CFR validation).

    Uses the Leduc game interface directly.
    """
    if rng is None:
        rng = np.random.RandomState()

    if state.is_terminal:
        return state.payoff(traverser)

    current = state.current_player
    actions = state.legal_actions()
    num_actions = len(actions)
    action_indices = [int(a) for a in actions]

    info_set = state.info_set
    features = _leduc_infoset_to_features(info_set)

    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        all_advantages = advantage_nets[current](x).squeeze(0).cpu().numpy()

    # Extract advantages at actual action positions (not positional slicing)
    advantages = all_advantages[action_indices]
    strategy = regret_match_masked(
        advantages, np.ones(num_actions, dtype=np.float32)
    )

    if current == traverser:
        action_utils = np.zeros(num_actions, dtype=np.float64)
        for i, action in enumerate(actions):
            next_state = state.apply(action)
            action_utils[i] = traverse_leduc(
                next_state, traverser, advantage_nets, iteration,
                advantage_memory, strategy_memory, device, rng,
            )

        node_value = np.sum(strategy * action_utils)
        regrets = np.zeros(num_actions, dtype=np.float32)
        for i in range(num_actions):
            regrets[i] = action_utils[i] - node_value

        weight = max(iteration, 1) ** 1.5
        padded_regrets = np.zeros(advantage_memory[traverser].target_dim, dtype=np.float32)
        for i, a_idx in enumerate(action_indices):
            padded_regrets[a_idx] = regrets[i]
        advantage_memory[traverser].add(features, padded_regrets, weight)

        return node_value
    else:
        chosen_idx = rng.choice(num_actions, p=strategy)
        chosen_action = actions[chosen_idx]

        weight = max(iteration, 1) ** 1.5
        padded_strategy = np.zeros(strategy_memory.target_dim, dtype=np.float32)
        for i, a_idx in enumerate(action_indices):
            padded_strategy[a_idx] = strategy[i]
        strategy_memory.add(features, padded_strategy, weight)

        next_state = state.apply(chosen_action)
        return traverse_leduc(
            next_state, traverser, advantage_nets, iteration,
            advantage_memory, strategy_memory, device, rng,
        )


# Leduc info set feature encoding
# Layout: private_rank(3) + community_rank(3) + pair(1) + round(2) + round0_actions(12) + round1_actions(12) = 33
_LEDUC_RANK_MAP = {"J": 0, "Q": 1, "K": 2}
_LEDUC_ACTION_MAP = {"f": 0, "c": 1, "r": 2}
LEDUC_FEATURE_DIM = 33


def _leduc_infoset_to_features(info_set: str) -> np.ndarray:
    """Convert Leduc info set string to feature vector."""
    features = np.zeros(LEDUC_FEATURE_DIM, dtype=np.float32)

    card_part, history_part = info_set.split(":")

    # Private card rank (3 dims one-hot)
    private_rank = _LEDUC_RANK_MAP[card_part[0]]
    features[private_rank] = 1.0

    # Community card rank (3 dims one-hot) — only present after preflop
    if len(card_part) > 1:
        comm_rank = _LEDUC_RANK_MAP[card_part[1]]
        features[3 + comm_rank] = 1.0
        # Pair indicator
        if private_rank == comm_rank:
            features[6] = 1.0

    # Round indicator (2 dims)
    rounds = history_part.split("|") if history_part else [""]
    round_idx = len(rounds) - 1
    if round_idx < 2:
        features[7 + round_idx] = 1.0

    # Action history: separate encoding per round (4 slots × 3 actions × 2 rounds = 24 dims)
    for r, round_h in enumerate(rounds):
        if r >= 2:
            break
        for j, ch in enumerate(round_h):
            if j < 4 and ch in _LEDUC_ACTION_MAP:
                features[9 + r * 12 + j * 3 + _LEDUC_ACTION_MAP[ch]] = 1.0

    return features


def _kuhn_infoset_to_features(info_set: str) -> np.ndarray:
    """Convert Kuhn info set to one-hot feature vector."""
    features = np.zeros(KUHN_FEATURE_DIM, dtype=np.float32)
    if info_set in _KUHN_INFOSETS:
        features[_KUHN_INFOSETS.index(info_set)] = 1.0
    return features
