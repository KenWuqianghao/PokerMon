#!/usr/bin/env python3
"""Train Deep CFR on Leduc Hold'em (validation)."""

import argparse

from pokermon.train.config import TrainConfig


def main():
    parser = argparse.ArgumentParser(description="Train Deep CFR on Leduc Hold'em")
    parser.add_argument("--config", default="configs/leduc.yaml", help="Config file")
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)

    print(f"Training Deep CFR on Leduc for {config.num_iterations} iterations")
    print("(Leduc training uses the same Deep CFR framework as NLHE)")
    print("See train_nlhe.py for the full training pipeline.")
    # Leduc training would use the same Trainer class but with Leduc game states.
    # For now, validation is done via tabular CFR on Leduc.
    print("Run tabular CFR validation:")
    print("  python -c \"from pokermon.cfr.tabular_cfr import TabularCFR; ...\"")


if __name__ == "__main__":
    main()
