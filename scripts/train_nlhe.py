#!/usr/bin/env python3
"""Train Deep CFR on 6-player No-Limit Texas Hold'em."""

import argparse

from pokermon.train.config import TrainConfig
from pokermon.train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Deep CFR on 6-max NLHE")
    parser.add_argument("--config", default="configs/nlhe6.yaml", help="Config file")
    parser.add_argument("--iterations", type=int, default=None, help="Override num iterations")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    n_iter = args.iterations or config.num_iterations

    print(f"Training Deep CFR on 6-max NLHE")
    print(f"  Device: {config.resolve_device()}")
    print(f"  Iterations: {n_iter}")
    print(f"  Traversals/iter: {config.traversals_per_iter}")
    print(f"  Players: {config.num_players}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Buffer capacity: {config.buffer_capacity:,}")

    trainer = Trainer(config)

    if args.resume:
        from pokermon.train.checkpoint import load_checkpoint
        info = load_checkpoint(args.resume, trainer.advantage_nets, trainer.strategy_net)
        print(f"  Resumed from iteration {info['iteration']}")

    trainer.train(num_iterations=n_iter)
    print("Training complete!")


if __name__ == "__main__":
    main()
