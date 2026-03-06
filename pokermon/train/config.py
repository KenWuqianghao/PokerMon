"""Training configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    """Deep CFR training configuration."""

    # Game
    game: str = "nlhe6"  # kuhn, leduc, nlhe6
    num_players: int = 6
    small_blind: int = 50
    big_blind: int = 100
    starting_stack: int = 10_000

    # Deep CFR
    num_iterations: int = 1000
    traversals_per_iter: int = 10_000
    advantage_sgd_steps: int = 4000
    strategy_sgd_steps: int = 4000
    strategy_train_every: int = 10
    eval_every: int = 50

    # Neural network
    hidden_dim: int = 512
    num_layers: int = 4
    num_actions: int = 7

    # Training
    lr: float = 1e-3
    batch_size: int = 2048
    buffer_capacity: int = 2_000_000
    weight_exponent: float = 1.5  # Linear CFR weighting: t^exponent

    # Pruning
    prune_after: int = 100
    prune_threshold: float = -300.0

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 50
    log_dir: str = "runs"

    # Hardware
    device: str = "auto"  # auto, cpu, cuda, mps, xla
    num_workers: int = 1

    # Seed
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML file."""
        from dataclasses import asdict
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def resolve_device(self) -> str:
        """Resolve 'auto' device to best available."""
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        try:
            import torch_xla.core.xla_model as xm  # noqa: F401
            return "xla"
        except ImportError:
            pass
        return "cpu"
