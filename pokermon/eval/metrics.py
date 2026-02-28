"""Evaluation metrics: bb/100, variance, confidence intervals."""

from __future__ import annotations

import numpy as np


def bb_per_100(profits: list[float], big_blind: int = 100) -> float:
    """Compute win rate in big blinds per 100 hands.

    Args:
        profits: List of per-hand profits.
        big_blind: Big blind amount.

    Returns:
        Win rate in bb/100.
    """
    if not profits:
        return 0.0
    return float(np.mean(profits)) / big_blind * 100


def variance_bb(profits: list[float], big_blind: int = 100) -> float:
    """Compute variance in bb^2 per hand."""
    if len(profits) < 2:
        return 0.0
    return float(np.var(profits, ddof=1)) / (big_blind ** 2)


def confidence_interval_95(
    profits: list[float], big_blind: int = 100
) -> tuple[float, float]:
    """Compute 95% confidence interval for bb/100.

    Args:
        profits: List of per-hand profits.
        big_blind: Big blind amount.

    Returns:
        (lower, upper) bounds for bb/100 at 95% confidence.
    """
    n = len(profits)
    if n < 2:
        return (0.0, 0.0)

    bb_profits = np.array(profits) / big_blind * 100
    mean = float(np.mean(bb_profits))
    se = float(np.std(bb_profits, ddof=1)) / np.sqrt(n)
    z = 1.96  # 95% CI

    return (mean - z * se, mean + z * se)


def summarize_results(results: dict, big_blind: int = 100) -> str:
    """Format match results as a human-readable string."""
    lines = [f"Match results ({results['hands_played']} hands):"]

    for i in range(results["num_players"]):
        bb100 = results["bb_per_100"][i]
        total = results["total_profit"][i]
        lines.append(f"  Player {i}: {bb100:+.1f} bb/100 (total: {total:+.0f})")

    return "\n".join(lines)
