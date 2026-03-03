"""Save/load training checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    iteration: int,
    advantage_nets: list[torch.nn.Module],
    strategy_net: torch.nn.Module,
    metrics: dict | None = None,
) -> None:
    """Save training checkpoint.

    Args:
        path: File path for the checkpoint.
        iteration: Current iteration number.
        advantage_nets: List of advantage network modules.
        strategy_net: Strategy network module.
        metrics: Optional training metrics dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "iteration": iteration,
        "advantage_nets": [net.state_dict() for net in advantage_nets],
        "strategy_net": strategy_net.state_dict(),
        "metrics": metrics or {},
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    advantage_nets: list[torch.nn.Module],
    strategy_net: torch.nn.Module,
) -> dict:
    """Load training checkpoint.

    Args:
        path: Checkpoint file path.
        advantage_nets: List of advantage network modules to load weights into.
        strategy_net: Strategy network module to load weights into.

    Returns:
        Dict with 'iteration' and 'metrics'.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    for i, state_dict in enumerate(checkpoint["advantage_nets"]):
        if i < len(advantage_nets):
            advantage_nets[i].load_state_dict(state_dict)

    strategy_net.load_state_dict(checkpoint["strategy_net"])

    return {
        "iteration": checkpoint["iteration"],
        "metrics": checkpoint.get("metrics", {}),
    }
