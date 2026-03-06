"""FastAPI backend for PokerMon web interface."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pokermon.cfr.infoset import encode_infoset_flat
from pokermon.eval.baselines import AggressiveBot
from pokermon.game.action import Action, compute_bet_size
from pokermon.game.card import card_to_str
from pokermon.game.engine import (
    apply_action,
    get_legal_actions,
    get_legal_actions_mask,
    new_hand,
)
from pokermon.game.hand_eval import evaluate, hand_rank_string
from pokermon.game.state import Street
from pokermon.net.strategy_net import StrategyNet

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "nlhe6" / "smoke_test.pt"
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

HUMAN = 0
AI = 1
NUM_PLAYERS = 2
STACK = 10_000
SMALL_BLIND = 50
BIG_BLIND = 100

# ---------------------------------------------------------------------------
# AI agent
# ---------------------------------------------------------------------------


class NeuralAgent:
    """Wraps StrategyNet for inference."""

    def __init__(self, net: StrategyNet) -> None:
        self.net = net
        self.net.eval()

    @torch.no_grad()
    def act(self, state, player: int) -> Action:
        features = encode_infoset_flat(state, player)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(
            get_legal_actions_mask(state), dtype=torch.bool
        ).unsqueeze(0)
        probs = self.net(x, mask).squeeze(0).numpy()
        action_idx = int(np.random.choice(len(probs), p=probs))
        return Action(action_idx)


def _load_ai_agent():
    if CHECKPOINT_PATH.exists():
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            net = StrategyNet()
            if "strategy_net" in checkpoint:
                net.load_state_dict(checkpoint["strategy_net"])
            elif "model_state_dict" in checkpoint:
                net.load_state_dict(checkpoint["model_state_dict"])
            else:
                net.load_state_dict(checkpoint)
            return NeuralAgent(net)
        except Exception:
            pass
    return AggressiveBot()


ai_agent = _load_ai_agent()

# ---------------------------------------------------------------------------
# Session storage
# ---------------------------------------------------------------------------

games: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _compute_action_amount(action: Action, state) -> int:
    """Compute chip cost for an action given the current state."""
    player = state.players[state.current_player]
    max_bet = max(p.bet for p in state.players)
    to_call = max_bet - player.bet
    pot = state.pot + sum(p.bet for p in state.players)

    if action == Action.FOLD:
        return 0
    if action == Action.CHECK_CALL:
        return min(to_call, player.stack)
    if action == Action.ALL_IN:
        return player.stack
    return compute_bet_size(action, pot, to_call, player.stack, state.min_raise)


def _action_label(action: Action, state) -> str:
    """Human-readable label with chip amount."""
    amount = _compute_action_amount(action, state)

    if action == Action.FOLD:
        return "Fold"
    if action == Action.CHECK_CALL:
        return f"Call {amount:,}" if amount > 0 else "Check"
    if action == Action.ALL_IN:
        return f"All In {amount:,}"

    # Bet or Raise
    player = state.players[state.current_player]
    max_bet = max(p.bet for p in state.players)
    to_call = max_bet - player.bet
    if to_call > 0:
        return f"Raise {amount:,}"
    return f"Bet {amount:,}"


def serialize_state(state, game_id: str, hand_num: int) -> dict:
    community = [card_to_str(c) for c in state.community_cards]
    pot = state.pot + sum(p.bet for p in state.players)

    human = state.players[HUMAN]
    ai_player = state.players[AI]

    human_cards = [card_to_str(c) for c in human.hole_cards]

    # AI cards hidden unless terminal
    ai_cards = None
    if state.is_terminal and not ai_player.folded:
        ai_cards = [card_to_str(c) for c in ai_player.hole_cards]

    legal_actions = None
    if not state.is_terminal and state.current_player == HUMAN:
        actions = get_legal_actions(state)
        legal_actions = [
            {
                "action": int(a),
                "label": _action_label(a, state),
                "amount": _compute_action_amount(a, state),
            }
            for a in actions
        ]

    result: dict[str, Any] = {
        "game_id": game_id,
        "hand_num": hand_num,
        "street": state.street.name.lower(),
        "community_cards": community,
        "pot": pot,
        "human": {
            "cards": human_cards,
            "stack": human.stack,
            "bet": human.bet,
            "folded": human.folded,
            "all_in": human.all_in,
        },
        "ai": {
            "cards": ai_cards,
            "stack": ai_player.stack,
            "bet": ai_player.bet,
            "folded": ai_player.folded,
            "all_in": ai_player.all_in,
        },
        "current_player": state.current_player,
        "is_terminal": state.is_terminal,
        "legal_actions": legal_actions,
        "button": state.button,
    }

    if state.is_terminal:
        result["payoffs"] = list(state.payoffs)
        result["result"] = state.payoffs[HUMAN]
        if len(state.community_cards) >= 5:
            hand_info = {}
            for i, label in [(HUMAN, "human"), (AI, "ai")]:
                p = state.players[i]
                if not p.folded and p.hole_cards:
                    score = evaluate(list(p.hole_cards), list(state.community_cards))
                    hand_info[label] = hand_rank_string(score)
            result["hand_info"] = hand_info

    return result


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="PokerMon")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewGameRequest(BaseModel):
    pass


class ActionRequest(BaseModel):
    game_id: str
    action: int


@app.post("/api/game/new")
def new_game(_req: NewGameRequest | None = None):
    game_id = uuid.uuid4().hex[:12]
    hand_num = 1
    button = 0

    state = new_hand(
        num_players=NUM_PLAYERS,
        stacks=[STACK] * NUM_PLAYERS,
        small_blind=SMALL_BLIND,
        big_blind=BIG_BLIND,
        button=button,
    )

    # If AI acts first, auto-play
    actions_taken = []
    _loop_limit = 20
    while not state.is_terminal and state.current_player == AI:
        _loop_limit -= 1
        if _loop_limit <= 0:
            raise HTTPException(status_code=500, detail="AI loop exceeded safety limit")
        action = ai_agent.act(state, AI)
        label = _action_label(action, state)
        amount = _compute_action_amount(action, state)
        state = apply_action(state, action)
        actions_taken.append({"player": "ai", "action": int(action), "label": label, "amount": amount})

    games[game_id] = {"state": state, "hand_num": hand_num, "button": button, "stacks": [STACK, STACK]}

    return {"game_id": game_id, "state": serialize_state(state, game_id, hand_num), "actions_taken": actions_taken}


@app.post("/api/game/action")
def game_action(req: ActionRequest):
    session = games.get(req.game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")

    state = session["state"]

    if state.is_terminal:
        raise HTTPException(status_code=400, detail="Hand is over")

    if state.current_player != HUMAN:
        raise HTTPException(status_code=400, detail="Not your turn")

    try:
        action = Action(req.action)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid action")

    legal = get_legal_actions(state)
    if action not in legal:
        raise HTTPException(status_code=400, detail=f"Illegal action: {action.name}")

    human_label = _action_label(action, state)
    human_amount = _compute_action_amount(action, state)
    state = apply_action(state, action)
    actions_taken = [{"player": "human", "action": int(action), "label": human_label, "amount": human_amount}]

    # AI auto-play loop (safety limit prevents infinite loops)
    _ai_loop_limit = 20
    while not state.is_terminal and state.current_player == AI:
        _ai_loop_limit -= 1
        if _ai_loop_limit <= 0:
            raise HTTPException(status_code=500, detail="AI loop exceeded safety limit")
        ai_action = ai_agent.act(state, AI)
        ai_label = _action_label(ai_action, state)
        ai_amount = _compute_action_amount(ai_action, state)
        state = apply_action(state, ai_action)
        actions_taken.append({"player": "ai", "action": int(ai_action), "label": ai_label, "amount": ai_amount})

    session["state"] = state

    # If terminal, update stacks for next hand
    if state.is_terminal:
        session["stacks"] = [state.players[i].stack for i in range(NUM_PLAYERS)]

    return {
        "state": serialize_state(state, req.game_id, session["hand_num"]),
        "actions_taken": actions_taken,
    }


@app.post("/api/game/deal")
def deal_again(req: ActionRequest):
    """Deal a new hand in the same session, preserving stacks."""
    session = games.get(req.game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")

    session["hand_num"] += 1
    session["button"] = (session["button"] + 1) % NUM_PLAYERS

    state = new_hand(
        num_players=NUM_PLAYERS,
        stacks=session["stacks"],
        small_blind=SMALL_BLIND,
        big_blind=BIG_BLIND,
        button=session["button"],
    )

    actions_taken = []
    _loop_limit = 20
    while not state.is_terminal and state.current_player == AI:
        _loop_limit -= 1
        if _loop_limit <= 0:
            raise HTTPException(status_code=500, detail="AI loop exceeded safety limit")
        action = ai_agent.act(state, AI)
        label = _action_label(action, state)
        amount = _compute_action_amount(action, state)
        state = apply_action(state, action)
        actions_taken.append({"player": "ai", "action": int(action), "label": label, "amount": amount})

    session["state"] = state

    return {
        "game_id": req.game_id,
        "state": serialize_state(state, req.game_id, session["hand_num"]),
        "actions_taken": actions_taken,
    }


# Serve frontend in production
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
