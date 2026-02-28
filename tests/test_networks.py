"""Tests for neural networks."""

import torch

from pokermon.net.advantage_net import AdvantageNet
from pokermon.net.card_encoder import CardEncoder, CommunityCardEncoder, PrivateCardEncoder
from pokermon.net.strategy_net import StrategyNet


def test_card_encoder():
    enc = CardEncoder(embed_dim=20)
    # Batch of 2, 3 cards each, (rank, suit) pairs
    cards = torch.tensor([
        [[12, 3], [11, 2], [0, 0]],
        [[5, 1], [-1, -1], [-1, -1]],
    ], dtype=torch.long)
    out = enc(cards)
    assert out.shape == (2, 3, 20)
    # Padded cards should be zeroed
    assert torch.allclose(out[1, 1], torch.zeros(20))
    assert torch.allclose(out[1, 2], torch.zeros(20))


def test_private_card_encoder():
    enc = PrivateCardEncoder(embed_dim=20)
    cards = torch.tensor([[[12, 3], [11, 2]]], dtype=torch.long)
    out = enc(cards)
    assert out.shape == (1, 40)


def test_community_card_encoder():
    enc = CommunityCardEncoder(embed_dim=20)
    cards = torch.tensor([
        [[5, 0], [6, 1], [7, 2], [-1, -1], [-1, -1]],
    ], dtype=torch.long)
    out = enc(cards)
    assert out.shape == (1, 20)


def test_advantage_net():
    net = AdvantageNet(input_dim=211, hidden_dim=64, num_actions=7, num_layers=2)
    x = torch.randn(4, 211)
    out = net(x)
    assert out.shape == (4, 7)


def test_advantage_net_masked():
    net = AdvantageNet(input_dim=211, hidden_dim=64, num_actions=7, num_layers=2)
    x = torch.randn(4, 211)
    mask = torch.tensor([
        [False, True, True, False, False, False, True],
        [True, True, False, False, False, False, True],
        [False, True, True, True, True, True, True],
        [True, True, True, True, True, True, True],
    ])
    out = net.predict_advantages(x, mask)
    # Masked actions should be -inf
    assert out[0, 0] == float("-inf")
    assert out[0, 3] == float("-inf")
    assert out[1, 2] == float("-inf")


def test_strategy_net():
    net = StrategyNet(input_dim=211, hidden_dim=64, num_actions=7, num_layers=2)
    x = torch.randn(4, 211)
    mask = torch.ones(4, 7, dtype=torch.bool)
    probs = net(x, mask)
    assert probs.shape == (4, 7)
    # Should sum to 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)
    # All non-negative
    assert (probs >= 0).all()


def test_strategy_net_masked():
    net = StrategyNet(input_dim=211, hidden_dim=64, num_actions=7, num_layers=2)
    x = torch.randn(2, 211)
    mask = torch.tensor([
        [False, True, True, False, False, False, False],
        [True, True, False, False, False, False, True],
    ], dtype=torch.bool)
    probs = net(x, mask)
    # Masked actions should have zero probability
    assert probs[0, 0].item() < 1e-6
    assert probs[0, 3].item() < 1e-6
    # Should still sum to 1 over legal actions
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_parameter_count():
    net = AdvantageNet(input_dim=211, hidden_dim=512, num_actions=7, num_layers=4)
    params = sum(p.numel() for p in net.parameters())
    # ~900K params: 211*512 + 512*512*3 + 512*7 + biases
    assert 800_000 < params < 1_000_000, f"Expected ~900K params, got {params}"
