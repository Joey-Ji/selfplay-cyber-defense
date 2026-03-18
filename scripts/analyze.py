"""Aggregate and print strategy comparison from experiment results.

Usage::

    uv run python scripts/analyze.py
    uv run python scripts/analyze.py --config experiments/quick.toml
"""

from __future__ import annotations

import json
from pathlib import Path

from absl import app, flags

from selfplay import config
from selfplay.analysis import load_metrics, compare_strategies, aggregate_across_seeds

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "experiments/paper.toml", "Path to experiment batch_config.")
flags.DEFINE_string("output", None, "Output path (default: <results_dir>/analysis.json).")


def main(_):
    batch_config = config.load(FLAGS.config)
    output = FLAGS.output or str(Path(batch_config.results_dir) / "analysis.json")

    print("Loading metrics...")
    all_metrics = load_metrics(batch_config.save_dir, batch_config.strategies, batch_config.seeds)

    if not all_metrics:
        print("No metrics found. Run training first.")
        return

    print("\nStrategy Comparison (Final Performance):")
    print("-" * 70)
    comparison = compare_strategies(all_metrics)
    for tag, stats in comparison.items():
        print(f"\n{tag} ({stats['n_seeds']} seeds):")
        print(f"  Blue final reward: {stats['blue_final']}")
        print(f"  Red final reward:  {stats['red_final']}")
        print(f"  Red win rate:      {stats['red_wr_final']}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result = {"comparison": comparison, "aggregated": {}}
    for tag, seed_runs in all_metrics.items():
        result["aggregated"][tag] = aggregate_across_seeds(seed_runs)

    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nAnalysis saved to {output}")


if __name__ == "__main__":
    app.run(main)
