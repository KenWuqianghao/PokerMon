"""Regret matching: convert regrets to a probability distribution."""

from __future__ import annotations

import numpy as np


def regret_match(regrets: np.ndarray) -> np.ndarray:
    """Convert cumulative regrets to a strategy via regret matching.

    Args:
        regrets: Array of cumulative regrets for each action.

    Returns:
        Probability distribution over actions.
    """
    positive = np.maximum(regrets, 0.0)
    total = positive.sum()
    if total > 0:
        return positive / total
    # Uniform over all actions
    return np.ones_like(regrets) / len(regrets)


def regret_match_masked(regrets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Regret matching with legal action masking.

    Args:
        regrets: Array of regrets for each action.
        mask: Boolean array of legal actions.

    Returns:
        Strategy with zero probability on illegal actions.
    """
    positive = np.maximum(regrets, 0.0) * mask
    total = positive.sum()
    if total > 0:
        return positive / total
    # Uniform over legal actions
    n_legal = mask.sum()
    if n_legal > 0:
        return mask.astype(np.float64) / n_legal
    return np.zeros_like(regrets)
