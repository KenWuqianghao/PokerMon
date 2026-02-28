"""Deep CFR orchestrator.

Trains advantage networks and strategy network using external sampling MCCFR.
The advantage networks replace tabular regret storage.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange

from pokermon.cfr.reservoir import ReservoirBuffer
from pokermon.cfr.traversal import (
    KUHN_FEATURE_DIM,
    LEDUC_FEATURE_DIM,
    _leduc_infoset_to_features,
    traverse_kuhn,
    traverse_leduc,
)
from pokermon.game.kuhn import all_kuhn_deals
from pokermon.game.leduc import all_leduc_deals


def _collect_all_leduc_infosets() -> list[tuple[str, int, int, list[int]]]:
    """Enumerate all Leduc info sets by traversing all deals.

    Returns list of (info_set, player, num_legal_actions, action_indices).
    """
    seen: dict[str, tuple[int, int, list[int]]] = {}

    def _traverse(state) -> None:
        if state.is_terminal:
            return
        info_set = state.info_set
        if info_set not in seen:
            actions = state.legal_actions()
            seen[info_set] = (
                state.current_player,
                len(actions),
                [int(a) for a in actions],
            )
        for action in state.legal_actions():
            _traverse(state.apply(action))

    for deal in all_leduc_deals():
        _traverse(deal)

    return [(k, v[0], v[1], v[2]) for k, v in seen.items()]


class DeepCFR:
    """Deep CFR training orchestrator."""

    def __init__(
        self,
        num_players: int = 2,
        feature_dim: int = KUHN_FEATURE_DIM,
        num_actions: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        buffer_capacity: int = 100_000,
        lr: float = 1e-3,
        sgd_steps: int = 2000,
        batch_size: int = 256,
        device: torch.device | None = None,
    ) -> None:
        self.num_players = num_players
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self._num_layers = num_layers
        self.sgd_steps = sgd_steps
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.lr = lr

        # Advantage networks (one per player)
        self.advantage_nets = [
            self._make_network(feature_dim, hidden_dim, num_actions, num_layers)
            for _ in range(num_players)
        ]

        # Strategy network (shared)
        self.strategy_net = self._make_network(
            feature_dim, hidden_dim, num_actions, num_layers
        )

        # Reservoir buffers
        self.advantage_memories = [
            ReservoirBuffer(buffer_capacity, feature_dim, num_actions)
            for _ in range(num_players)
        ]
        self.strategy_memory = ReservoirBuffer(buffer_capacity, feature_dim, num_actions)

    def _make_network(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> nn.Module:
        """Create a simple MLP."""
        layers: list[nn.Module] = []
        in_d = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_d, hidden_dim), nn.ReLU()])
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        model = nn.Sequential(*layers)
        return model.to(self.device)

    def train_kuhn(
        self,
        num_iterations: int = 200,
        traversals_per_iter: int = 1000,
        verbose: bool = True,
    ) -> dict[str, list]:
        """Train Deep CFR on Kuhn poker.

        Returns dict of training metrics.
        """
        rng = np.random.RandomState(42)
        deals = all_kuhn_deals()
        metrics: dict[str, list] = {"iteration": [], "loss": []}

        iterator = trange(1, num_iterations + 1, desc="Deep CFR") if verbose else range(1, num_iterations + 1)

        for t in iterator:
            # Phase 1: Run MCCFR traversals for each player
            for p in range(self.num_players):
                for _ in range(traversals_per_iter):
                    # Random deal
                    deal_idx = rng.randint(0, len(deals))
                    state = deals[deal_idx]

                    traverse_kuhn(
                        state, p, self.advantage_nets, t,
                        self.advantage_memories, self.strategy_memory,
                        self.device, rng,
                    )

                # Retrain advantage net FROM SCRATCH
                if self.advantage_memories[p].size > self.batch_size:
                    self.advantage_nets[p] = self._make_network(
                        self.feature_dim, self.hidden_dim, self.num_actions, self._num_layers
                    )
                    loss = self._train_advantage_net(p)
                    metrics["loss"].append(loss)

            # Periodically train strategy network
            if t % 10 == 0 and self.strategy_memory.size > self.batch_size:
                self._train_strategy_net()

            metrics["iteration"].append(t)

            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix(
                    mem=self.advantage_memories[0].size,
                    loss=metrics["loss"][-1] if metrics["loss"] else 0,
                )

        return metrics

    def _train_advantage_net(self, player: int) -> float:
        """Train advantage network from scratch on buffered data."""
        net = self.advantage_nets[player]
        optimizer = optim.Adam(net.parameters(), lr=self.lr)
        buf = self.advantage_memories[player]

        total_loss = 0.0
        net.train()

        for step in range(self.sgd_steps):
            features, targets, weights = buf.sample(self.batch_size)

            x = torch.tensor(features, dtype=torch.float32, device=self.device)
            y = torch.tensor(targets, dtype=torch.float32, device=self.device)
            w = torch.tensor(weights, dtype=torch.float32, device=self.device)

            pred = net(x)
            # Weighted MSE loss
            loss = (w.unsqueeze(-1) * (pred - y) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / self.sgd_steps

    def _train_strategy_net(self) -> float:
        """Fine-tune strategy network on buffered data."""
        net = self.strategy_net
        optimizer = optim.Adam(net.parameters(), lr=self.lr)
        buf = self.strategy_memory

        total_loss = 0.0
        net.train()

        for step in range(self.sgd_steps):
            features, targets, weights = buf.sample(self.batch_size)

            x = torch.tensor(features, dtype=torch.float32, device=self.device)
            y = torch.tensor(targets, dtype=torch.float32, device=self.device)
            w = torch.tensor(weights, dtype=torch.float32, device=self.device)

            # Predict logits → softmax → cross-entropy with target strategy
            logits = net(x)
            log_probs = torch.log_softmax(logits, dim=-1)
            # Weighted cross-entropy: -Σ w * y * log(p)
            loss = -(w.unsqueeze(-1) * y * log_probs).sum(dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / self.sgd_steps

    def train_leduc(
        self,
        num_iterations: int = 500,
        traversals_per_iter: int = 2000,
        verbose: bool = True,
    ) -> dict[str, list]:
        """Train Deep CFR on Leduc Hold'em.

        Returns dict of training metrics.
        """
        rng = np.random.RandomState(42)
        deals = all_leduc_deals()
        metrics: dict[str, list] = {"iteration": [], "loss": []}

        iterator = trange(1, num_iterations + 1, desc="Deep CFR Leduc") if verbose else range(1, num_iterations + 1)

        for t in iterator:
            for p in range(self.num_players):
                for _ in range(traversals_per_iter):
                    deal_idx = rng.randint(0, len(deals))
                    state = deals[deal_idx]

                    traverse_leduc(
                        state, p, self.advantage_nets, t,
                        self.advantage_memories, self.strategy_memory,
                        self.device, rng,
                    )

                # Retrain advantage net FROM SCRATCH
                if self.advantage_memories[p].size > self.batch_size:
                    self.advantage_nets[p] = self._make_network(
                        self.feature_dim, self.hidden_dim, self.num_actions,
                        self._num_layers,
                    )
                    loss = self._train_advantage_net(p)
                    metrics["loss"].append(loss)

            # Periodically train strategy network
            if t % 10 == 0 and self.strategy_memory.size > self.batch_size:
                self._train_strategy_net()

            metrics["iteration"].append(t)

            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix(
                    mem=self.advantage_memories[0].size,
                    loss=metrics["loss"][-1] if metrics["loss"] else 0,
                )

        return metrics

    def get_leduc_advantage_strategy(self) -> dict[str, np.ndarray]:
        """Extract the current strategy from advantage networks for Leduc poker."""
        from pokermon.cfr.regret_matching import regret_match

        infosets = _collect_all_leduc_infosets()
        strategy = {}

        for info_set, player, num_legal, action_indices in infosets:
            features = _leduc_infoset_to_features(info_set)
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
                all_advantages = self.advantage_nets[player](x).squeeze(0).cpu().numpy()

            # Extract advantages at actual action positions
            legal_advantages = all_advantages[action_indices]
            legal_probs = regret_match(legal_advantages)

            # Store as action-indexed array (size num_actions)
            full_strategy = np.zeros(self.num_actions, dtype=np.float64)
            for i, a_idx in enumerate(action_indices):
                full_strategy[a_idx] = legal_probs[i]
            strategy[info_set] = full_strategy

        return strategy

    def get_leduc_strategy(self) -> dict[str, np.ndarray]:
        """Extract the average strategy from the strategy network for Leduc poker."""
        infosets = _collect_all_leduc_infosets()
        strategy = {}
        self.strategy_net.eval()

        with torch.no_grad():
            for info_set, _player, num_legal, action_indices in infosets:
                features = _leduc_infoset_to_features(info_set)
                x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
                all_logits = self.strategy_net(x).squeeze(0).cpu().numpy()
                legal_logits = all_logits[action_indices]
                probs = np.exp(legal_logits - legal_logits.max())
                probs = probs / probs.sum()

                full_strategy = np.zeros(self.num_actions, dtype=np.float64)
                for i, a_idx in enumerate(action_indices):
                    full_strategy[a_idx] = probs[i]
                strategy[info_set] = full_strategy

        return strategy

    def get_kuhn_strategy(self) -> dict[str, np.ndarray]:
        """Extract the average strategy from the strategy network for Kuhn poker.

        Returns info_set → action probability mapping.
        """
        from pokermon.cfr.traversal import _KUHN_INFOSETS, _kuhn_infoset_to_features
        from pokermon.cfr.regret_matching import regret_match

        strategy = {}
        self.strategy_net.eval()

        with torch.no_grad():
            for info_set in _KUHN_INFOSETS:
                features = _kuhn_infoset_to_features(info_set)
                x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits = self.strategy_net(x).squeeze(0).cpu().numpy()
                # Softmax
                probs = np.exp(logits - logits.max())
                probs = probs / probs.sum()
                strategy[info_set] = probs

        return strategy

    def get_kuhn_advantage_strategy(self) -> dict[str, np.ndarray]:
        """Extract the current strategy from advantage networks for Kuhn poker."""
        from pokermon.cfr.traversal import _KUHN_INFOSETS, _kuhn_infoset_to_features
        from pokermon.cfr.regret_matching import regret_match

        strategy = {}

        for info_set in _KUHN_INFOSETS:
            features = _kuhn_infoset_to_features(info_set)
            # Determine which player this info set belongs to
            # In Kuhn, player 0's info sets have even-length history, player 1's have odd
            history = info_set[1:]  # Remove card character
            player = len(history) % 2

            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
                advantages = self.advantage_nets[player](x).squeeze(0).cpu().numpy()

            strategy[info_set] = regret_match(advantages[:2])

        return strategy
