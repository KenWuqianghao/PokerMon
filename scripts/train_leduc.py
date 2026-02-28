#!/usr/bin/env python3
"""Train Deep CFR on Leduc Hold'em (validation)."""

import argparse
import sys

from pokermon.cfr.deep_cfr import DeepCFR
from pokermon.cfr.traversal import LEDUC_FEATURE_DIM
from pokermon.eval.exploitability import compute_exploitability_leduc
from pokermon.train.config import TrainConfig


def main():
    parser = argparse.ArgumentParser(description="Train Deep CFR on Leduc Hold'em")
    parser.add_argument("--config", default="configs/leduc.yaml", help="Config file")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--exploit-threshold", type=float, default=None,
                        help="Exploitability threshold (default: 0.20)")
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)

    dcfr = DeepCFR(
        num_players=config.num_players,
        feature_dim=LEDUC_FEATURE_DIM,
        num_actions=config.num_actions,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        buffer_capacity=config.buffer_capacity,
        lr=config.lr,
        sgd_steps=config.advantage_sgd_steps,
        batch_size=config.batch_size,
    )

    n_iter = args.iterations if args.iterations is not None else config.num_iterations
    n_trav = args.traversals if args.traversals is not None else config.traversals_per_iter

    print(f"Training Deep CFR on Leduc for {n_iter} iterations, {n_trav} traversals/iter")
    dcfr.train_leduc(num_iterations=n_iter, traversals_per_iter=n_trav, verbose=True)

    # Final strategy network training pass
    if dcfr.strategy_memory.size > config.batch_size:
        dcfr._train_strategy_net()

    # Evaluate using the strategy network (average strategy)
    strategy = dcfr.get_leduc_strategy()
    exploit = compute_exploitability_leduc(strategy)

    print(f"\nFinal exploitability: {exploit:.6f}")
    print(f"Info sets in strategy: {len(strategy)}")

    threshold = args.exploit_threshold if args.exploit_threshold is not None else 0.20
    if exploit < threshold:
        print(f"\nPASS: Exploitability {exploit:.6f} < {threshold}")
    else:
        print(f"\nFAIL: Exploitability {exploit:.6f} >= {threshold}")
        sys.exit(1)


if __name__ == "__main__":
    main()
