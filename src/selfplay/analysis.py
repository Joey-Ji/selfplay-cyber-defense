"""Aggregate results across seeds and compute statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from selfplay import config


def load_metrics(
    base_dir: str,
    strategies: list[config.Strategy],
    seeds: list[int],
) -> dict:
    """Load metrics.json for each strategy x seed combination."""
    results = {}
    base = Path(base_dir)
    for strategy in strategies:
        seed_data = []
        for seed in seeds:
            path = base / f"{strategy.tag}_seed{seed}" / strategy.name / "metrics.json"
            if path.exists():
                with open(path) as f:
                    seed_data.append(json.load(f))

        if seed_data:
            results[strategy.tag] = seed_data
    return results


def aggregate_across_seeds(seed_runs: list[list[dict]]) -> dict:
    """Aggregate metrics across seeds, computing mean and std."""
    max_rounds = max(len(run) for run in seed_runs)
    keys = ["red_mean_reward", "blue_mean_reward", "red_win_rate"]

    agg = {}
    for key in keys:
        all_values = []
        for run in seed_runs:
            values = [m.get(key, 0.0) for m in run]
            if len(values) < max_rounds:
                values.extend([values[-1]] * (max_rounds - len(values)))
            all_values.append(values)

        arr = np.array(all_values)
        agg[f"{key}_mean"] = arr.mean(axis=0).tolist()
        agg[f"{key}_std"] = arr.std(axis=0).tolist()
        agg[f"{key}_final_mean"] = float(arr[:, -1].mean())
        agg[f"{key}_final_std"] = float(arr[:, -1].std())

    agg["n_seeds"] = len(seed_runs)
    agg["n_rounds"] = max_rounds
    return agg


def compare_strategies(all_metrics: dict) -> dict:
    """Compare final performance across strategies."""
    comparison = {}
    for tag, seed_runs in all_metrics.items():
        agg = aggregate_across_seeds(seed_runs)
        comparison[tag] = {
            "red_final": f"{agg['red_mean_reward_final_mean']:.2f} +/- {agg['red_mean_reward_final_std']:.2f}",
            "blue_final": f"{agg['blue_mean_reward_final_mean']:.2f} +/- {agg['blue_mean_reward_final_std']:.2f}",
            "red_wr_final": f"{agg['red_win_rate_final_mean']:.3f} +/- {agg['red_win_rate_final_std']:.3f}",
            "n_seeds": agg["n_seeds"],
        }
    return comparison
