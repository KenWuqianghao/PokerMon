#!/usr/bin/env python3
"""Batch evaluation vs baselines."""

import argparse

from pokermon.eval.arena import run_match
from pokermon.eval.baselines import AggressiveBot, CallStation, FoldBot, RandomAgent
from pokermon.eval.metrics import summarize_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate agents")
    parser.add_argument("--hands", type=int, default=10000, help="Number of hands")
    parser.add_argument("--players", type=int, default=6, help="Number of players")
    parser.add_argument("--stack", type=int, default=10000, help="Starting stack")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("PokerMon Evaluation")
    print("=" * 50)

    # Baseline matchups
    matchups = {
        "All Random": [RandomAgent(seed=i) for i in range(args.players)],
        "Random vs CallStation": (
            [RandomAgent(seed=0)] + [CallStation() for _ in range(args.players - 1)]
        ),
        "CallStation vs FoldBot": (
            [CallStation() for _ in range(args.players // 2)]
            + [FoldBot() for _ in range(args.players - args.players // 2)]
        ),
        "Mixed Baselines": [
            RandomAgent(seed=0),
            CallStation(),
            FoldBot(),
            AggressiveBot(seed=0),
            RandomAgent(seed=1),
            CallStation(),
        ][:args.players],
    }

    for name, agents in matchups.items():
        print(f"\n{name}:")
        results = run_match(
            agents=agents,
            num_hands=args.hands,
            starting_stack=args.stack,
            seed=args.seed,
        )
        print(summarize_results(results))


if __name__ == "__main__":
    main()
