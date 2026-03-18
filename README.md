# selfplay

Self-play reinforcement learning for autonomous cyber defense using MiniCAGE.

Trains Red (attacker) and Blue (defender) PPO agents via self-play with three opponent-sampling strategies: **vanilla**, **delta-uniform**, and **prioritized fictitious self-play (PFSP)**.

## Setup

```bash
uv sync
```

## Quick Start

```bash
# List all available commands
just

# Train a single agent pair
just train --config.selfplay.strategy=vanilla --config.selfplay.total_rounds=50

# Evaluate agents
just evaluate --red=bline --blue=react_restore --episodes=1000

# Quick smoke test (minutes)
just smoke
```

## Full Experiment Pipeline

Everything to run is defined in a TOML experiment config
(`experiments/paper.toml`).  Edit the config to change strategies, seeds,
hyperparameters, or output directories — no Python knowledge needed:

```bash
# Full paper reproduction (hours)
just pipeline

# With a different config
just config=experiments/quick.toml train-all

# Run phases individually — if one breaks, re-run just that step
just train-all
just crossplay
just exploitability
just plot
just analyze
```

## Experiment Configs

Configs are simple TOML files in `experiments/`.  Copy one to create a new
experiment:

```bash
cp experiments/paper.toml experiments/my_experiment.toml
# edit strategies, seeds, hyperparams …
just config=experiments/my_experiment.toml pipeline
```

## Project Structure

```
src/selfplay/
├── config.py          # Dataclasses, experiment config loader
├── analysis.py        # Result aggregation across seeds
├── env/               # MiniCAGE environment, agents, gym wrappers
├── training/          # Self-play loop, BC pre-training, batch training
├── eval/              # Evaluation, cross-play, exploitability
└── viz/               # Plotting utilities
scripts/               # Thin CLI wrappers (absl.app)
experiments/           # TOML experiment configs
justfile               # Task runner (just)
tests/                 # Smoke tests
docs/milestone/        # Project report
```

## License

MIT
