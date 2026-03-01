# PokerMon

Deep Counterfactual Regret Minimization (Deep CFR) for 6-player No-Limit Texas Hold'em.

PokerMon trains neural networks to approximate Nash equilibrium strategies in multiplayer poker using external sampling MCCFR with function approximation. The implementation is validated on Kuhn poker and Leduc Hold'em before scaling to full 6-max NLHE.

## Project Structure

```
pokermon/
  game/          # Game logic: cards, deck, hand eval, engine, Kuhn, Leduc
  cfr/           # CFR algorithms: Deep CFR, tabular CFR+, traversal, regret matching
  net/           # Neural networks: advantage net, strategy net, encoders
  eval/          # Evaluation: exploitability, arena, baselines, metrics
  train/         # Training: config, trainer, checkpointing
  utils/         # Logging, card utilities
scripts/         # Training and evaluation entry points
configs/         # YAML configs for Kuhn, Leduc, and 6-max NLHE
tests/           # Test suite (75+ tests)
```

## How It Works

Deep CFR replaces the regret tables in vanilla CFR with neural networks:

1. **Traverse** the game tree using external sampling MCCFR
2. **Collect** counterfactual regrets and strategy samples into reservoir buffers
3. **Train** advantage networks (one per player) from scratch each iteration to predict regrets
4. **Train** a strategy network on weighted strategy samples to produce the average policy
5. The strategy network's output converges to a Nash equilibrium

Key implementation details:
- Advantage networks are rebuilt from scratch each iteration (not fine-tuned)
- Linear CFR weighting (`iteration^1.5`) prioritizes later, more accurate samples
- Reservoir sampling keeps memory bounded regardless of iteration count
- Exploitability is computed via best-response traversal for small games

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.10+ and PyTorch 2.0+.

## Training

**Kuhn poker** (3-card toy game, ~5 seconds):
```bash
python scripts/train_kuhn.py
```

**Leduc Hold'em** (6-card game, ~40 minutes on CPU):
```bash
python scripts/train_leduc.py --iterations 200
```

**6-max No-Limit Hold'em** (full game):
```bash
python scripts/train_nlhe.py --config configs/nlhe6.yaml
```

## Evaluation

Run baseline matchups (Random, CallStation, FoldBot, AggressiveBot):
```bash
python scripts/evaluate.py --hands 10000
```

Play interactively against the agent:
```bash
python scripts/play.py --num-players 2
```

## Validation Results

The implementation is validated through a series of gates on progressively harder games:

| Game | Method | Exploitability | Threshold |
|------|--------|---------------|-----------|
| Kuhn | Deep CFR (150 iter) | 0.141 | < 0.15 |
| Kuhn | Tabular CFR+ (10K iter) | 0.002 | < 0.01 |
| Leduc | Tabular CFR+ (10K iter) | 0.011 | < 0.10 |
| Leduc | Deep CFR (200 iter) | 0.171 | < 0.20 |

For reference, Kuhn poker's Nash equilibrium game value is -1/18 (~-0.0556); tabular CFR+ converges to -0.0499.

## Tests

```bash
pytest                          # all tests
pytest -k "not deep_cfr"       # skip slow Deep CFR tests (~15s)
```

## Configuration

Training configs are in `configs/`. Key parameters:

| Parameter | Kuhn | Leduc | NLHE 6-max |
|-----------|------|-------|------------|
| `hidden_dim` | 64 | 128 | 512 |
| `num_layers` | 2 | 3 | 4 |
| `num_actions` | 2 | 3 | 7 |
| `buffer_capacity` | 100K | 500K | 2M |
| `traversals_per_iter` | 1000 | 2000 | 10K |
| `batch_size` | 256 | 512 | 2048 |

## References

- [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164) (Brown et al., 2019)
- [Regret Minimization in Games with Incomplete Information](https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf) (Zinkevich et al., 2007)
