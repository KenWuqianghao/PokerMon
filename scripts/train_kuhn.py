#!/usr/bin/env python3
"""Train Deep CFR on Kuhn poker (validation)."""

import argparse
import sys

from pokermon.cfr.deep_cfr import DeepCFR
from pokermon.eval.exploitability import compute_exploitability_kuhn
from pokermon.train.config import TrainConfig


def main():
    parser = argparse.ArgumentParser(description="Train Deep CFR on Kuhn poker")
    parser.add_argument("--config", default="configs/kuhn.yaml", help="Config file")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)

    dcfr = DeepCFR(
        num_players=config.num_players,
        num_actions=config.num_actions,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        buffer_capacity=config.buffer_capacity,
        lr=config.lr,
        sgd_steps=config.advantage_sgd_steps,
        batch_size=config.batch_size,
    )

    n_iter = args.iterations or config.num_iterations
    n_trav = args.traversals or config.traversals_per_iter

    print(f"Training Deep CFR on Kuhn for {n_iter} iterations, {n_trav} traversals/iter")
    dcfr.train_kuhn(num_iterations=n_iter, traversals_per_iter=n_trav, verbose=True)

    # Evaluate
    strategy = dcfr.get_kuhn_advantage_strategy()
    exploit = compute_exploitability_kuhn(strategy)

    print(f"\nFinal exploitability: {exploit:.6f}")
    print("Strategy:")
    for info_set in sorted(strategy):
        print(f"  {info_set:6s}: {strategy[info_set]}")

    if exploit < 0.15:
        print("\nPASS: Exploitability < 0.15")
    else:
        print("\nFAIL: Exploitability >= 0.15")
        sys.exit(1)


if __name__ == "__main__":
    main()
