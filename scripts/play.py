#!/usr/bin/env python3
"""Interactive play vs trained agent."""

import argparse
import sys

import torch

from pokermon.game.action import Action
from pokermon.game.card import card_to_str
from pokermon.game.deck import Deck
from pokermon.game.engine import apply_action, get_legal_actions, new_hand
from pokermon.game.state import Street


def main():
    parser = argparse.ArgumentParser(description="Play against trained PokerMon agent")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--seat", type=int, default=0, help="Your seat (0-5)")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--stack", type=int, default=10000, help="Starting stack")
    args = parser.parse_args()

    print("PokerMon — Interactive Play")
    print("=" * 40)
    print(f"Players: {args.num_players}, Stack: {args.stack}, Your seat: {args.seat}")
    print()

    hand_num = 0
    while True:
        hand_num += 1
        print(f"\n--- Hand #{hand_num} ---")

        deck = Deck()
        state = new_hand(
            num_players=args.num_players,
            stacks=[args.stack] * args.num_players,
            small_blind=50,
            big_blind=100,
            button=hand_num % args.num_players,
            deck=deck,
        )

        # Show player's hole cards
        player = state.players[args.seat]
        hole = [card_to_str(c) for c in player.hole_cards]
        print(f"Your cards: {hole[0]} {hole[1]}")

        while not state.is_terminal:
            # Show community cards
            if state.community_cards:
                board = " ".join(card_to_str(c) for c in state.community_cards)
                print(f"Board: {board}")

            current = state.current_player
            pot = state.pot + sum(p.bet for p in state.players)
            print(f"Pot: {pot} | Player {current}'s turn | Stack: {state.players[current].stack}")

            if current == args.seat:
                # Human's turn
                legal = get_legal_actions(state)
                print("Legal actions:")
                for i, a in enumerate(legal):
                    print(f"  {i}: {a.name}")

                while True:
                    try:
                        choice = int(input("Your action: "))
                        if 0 <= choice < len(legal):
                            action = legal[choice]
                            break
                    except (ValueError, EOFError):
                        pass
                    print("Invalid choice, try again.")
            else:
                # AI's turn — for now, simple check/call bot
                action = Action.CHECK_CALL
                legal = get_legal_actions(state)
                if action not in legal:
                    action = legal[0]
                print(f"  Player {current} plays: {action.name}")

            state = apply_action(state, action)

        # Show results
        print(f"\nHand over! Payoffs: {state.payoffs}")
        print(f"Your result: {state.payoffs[args.seat]:+d}")

        try:
            cont = input("\nPlay another hand? (y/n): ").strip().lower()
            if cont != "y":
                break
        except EOFError:
            break

    print("Thanks for playing!")


if __name__ == "__main__":
    main()
