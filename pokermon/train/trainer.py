"""Main training loop for Deep CFR on NLHE."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from pokermon.cfr.infoset import TOTAL_DIM
from pokermon.cfr.reservoir import ReservoirBuffer
from pokermon.cfr.traversal import external_sampling_mccfr
from pokermon.game.action import NUM_ACTIONS
from pokermon.game.deck import Deck
from pokermon.game.engine import new_hand
from pokermon.net.advantage_net import AdvantageNet
from pokermon.net.strategy_net import StrategyNet
from pokermon.train.checkpoint import save_checkpoint
from pokermon.train.config import TrainConfig
from pokermon.utils.logging import TBLogger


class Trainer:
    """Deep CFR trainer for NLHE."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = self._resolve_torch_device(config)

        # Networks
        self.advantage_nets = [
            AdvantageNet(
                input_dim=TOTAL_DIM,
                hidden_dim=config.hidden_dim,
                num_actions=config.num_actions,
                num_layers=config.num_layers,
            ).to(self.device)
            for _ in range(config.num_players)
        ]

        self.strategy_net = StrategyNet(
            input_dim=TOTAL_DIM,
            hidden_dim=config.hidden_dim,
            num_actions=config.num_actions,
            num_layers=config.num_layers,
        ).to(self.device)

        # Reservoir buffers
        self.advantage_memories = [
            ReservoirBuffer(config.buffer_capacity, TOTAL_DIM, config.num_actions)
            for _ in range(config.num_players)
        ]
        self.strategy_memory = ReservoirBuffer(
            config.buffer_capacity, TOTAL_DIM, config.num_actions
        )

        # Logging
        self.logger = TBLogger(config.log_dir, enabled=True)

        # RNG
        self.rng = np.random.RandomState(config.seed)

    def train(
        self,
        num_iterations: int | None = None,
        start_iteration: int = 1,
        on_checkpoint: Callable[[int], None] | None = None,
    ) -> None:
        """Run the full Deep CFR training loop."""
        n_iter = num_iterations or self.config.num_iterations

        for t in trange(start_iteration, n_iter + 1, desc="Deep CFR NLHE"):
            # For each player, run traversals and retrain advantage net
            for p in range(self.config.num_players):
                self._run_traversals(t, p)

                if self.advantage_memories[p].size > self.config.batch_size:
                    # Retrain from scratch (paper finding)
                    self.advantage_nets[p] = AdvantageNet(
                        input_dim=TOTAL_DIM,
                        hidden_dim=self.config.hidden_dim,
                        num_actions=self.config.num_actions,
                        num_layers=self.config.num_layers,
                    ).to(self.device)
                    loss = self._train_network(
                        self.advantage_nets[p],
                        self.advantage_memories[p],
                        self.config.advantage_sgd_steps,
                        loss_type="mse",
                    )
                    self.logger.scalar(f"advantage_loss/player_{p}", loss, t)

            # Train strategy network periodically
            if t % self.config.strategy_train_every == 0:
                if self.strategy_memory.size > self.config.batch_size:
                    loss = self._train_network(
                        self.strategy_net,
                        self.strategy_memory,
                        self.config.strategy_sgd_steps,
                        loss_type="cross_entropy",
                    )
                    self.logger.scalar("strategy_loss", loss, t)

            # Checkpoint
            if t % self.config.checkpoint_every == 0:
                save_checkpoint(
                    f"{self.config.checkpoint_dir}/checkpoint_{t:06d}.pt",
                    t,
                    self.advantage_nets,
                    self.strategy_net,
                )
                if on_checkpoint:
                    on_checkpoint(t)

            self.logger.flush()

        self.logger.close()

    @staticmethod
    def _resolve_torch_device(config: TrainConfig) -> torch.device:
        """Create torch.device, handling XLA/TPU specially."""
        device_str = config.resolve_device()
        if device_str == "xla":
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        return torch.device(device_str)

    def _run_traversals(self, iteration: int, traverser: int) -> None:
        for net in self.advantage_nets:
            net.eval()
        """Run MCCFR traversals for a given player."""
        for _ in range(self.config.traversals_per_iter):
            deck = Deck(seed=self.rng.randint(0, 2**31))
            stacks = [self.config.starting_stack] * self.config.num_players
            state = new_hand(
                num_players=self.config.num_players,
                stacks=stacks,
                small_blind=self.config.small_blind,
                big_blind=self.config.big_blind,
                button=self.rng.randint(0, self.config.num_players),
                deck=deck,
            )

            external_sampling_mccfr(
                state,
                traverser,
                self.advantage_nets,
                iteration,
                self.advantage_memories,
                self.strategy_memory,
                self.device,
                self.rng,
                weight_exponent=self.config.weight_exponent,
                prune_threshold=self.config.prune_threshold,
                prune_after=self.config.prune_after,
            )

    def _train_network(
        self,
        net: nn.Module,
        buffer: ReservoirBuffer,
        sgd_steps: int,
        loss_type: str = "mse",
    ) -> float:
        """Train a network on buffered data."""
        optimizer = optim.Adam(net.parameters(), lr=self.config.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sgd_steps)
        net.train()
        total_loss = 0.0

        for step in range(sgd_steps):
            features, targets, weights = buffer.sample(self.config.batch_size)

            x = torch.tensor(features, dtype=torch.float32, device=self.device)
            y = torch.tensor(targets, dtype=torch.float32, device=self.device)
            w = torch.tensor(weights, dtype=torch.float32, device=self.device)

            # AdvantageNet.forward(x) returns raw values; StrategyNet.forward
            # applies masked softmax which we don't want during training.
            # Access the underlying Sequential directly for raw logits.
            pred = net(x) if loss_type == "mse" else net.net(x)

            if loss_type == "mse":
                loss = (w.unsqueeze(-1) * (pred - y) ** 2).mean()
            else:
                # Cross-entropy for strategy net
                log_probs = torch.log_softmax(pred, dim=-1)
                loss = -(w.unsqueeze(-1) * y * log_probs).sum(dim=-1).mean()

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        return total_loss / sgd_steps
