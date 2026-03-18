"""Visualization utilities for self-play experiment results."""

from __future__ import annotations

import json
from pathlib import Path

from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from selfplay import config


def plot_crossplay_heatmap(
    matrix: np.ndarray,
    title: str = "Cross-Play Reward Matrix",
    xlabel: str = "Blue Checkpoint",
    ylabel: str = "Red Checkpoint",
    output: str | None = None,
):
    """Plot a heatmap of cross-play rewards."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(
        matrix, annot=True, fmt=".1f", cmap="RdYlGn",
        xticklabels=[f"B{i}" for i in range(matrix.shape[1])],
        yticklabels=[f"R{i}" for i in range(matrix.shape[0])],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150, bbox_inches="tight")
        logging.info("Saved heatmap to %s", output)
    else:
        plt.show()
    plt.close()


def plot_exploitability_curve(
    values: np.ndarray,
    names: list[str] | None = None,
    title: str = "Exploitability Over Training",
    output: str | None = None,
):
    """Plot exploitability values over training rounds."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = range(len(values))
    ax.plot(x, values, "o-", linewidth=2, markersize=6)
    ax.set_xlabel("Checkpoint Index")
    ax.set_ylabel("Exploitability (Best-Response Reward)")
    ax.set_title(title)
    if names:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150, bbox_inches="tight")
        logging.info("Saved exploitability curve to %s", output)
    else:
        plt.show()
    plt.close()


def plot_reward_curves(
    data: dict[str, list[dict]],
    role: str = "red",
    title: str = "Training Reward Curves",
    output: str | None = None,
):
    """Plot reward curves across strategies.

    Args:
        data: ``{strategy_name: [list of per-round metric dicts]}``.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    key = f"{role}_mean_reward"
    for name, metrics in data.items():
        rounds = [m["round"] for m in metrics]
        rewards = [m[key] for m in metrics]
        window = max(1, len(rewards) // 50)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(rounds[:len(smoothed)], smoothed, label=name, linewidth=1.5)

    ax.set_xlabel("Training Round")
    ax.set_ylabel(f"{role.capitalize()} Mean Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150, bbox_inches="tight")
        logging.info("Saved reward curves to %s", output)
    else:
        plt.show()
    plt.close()


def plot_strategy_analysis(
    action_freqs: dict[str, np.ndarray],
    action_names: list[str] | None = None,
    title: str = "Action Distribution",
    output: str | None = None,
):
    """Plot action frequency distributions for emergent strategy analysis."""
    fig, axes = plt.subplots(1, len(action_freqs), figsize=(6 * len(action_freqs), 5), sharey=True)
    if len(action_freqs) == 1:
        axes = [axes]

    for ax, (name, freqs) in zip(axes, action_freqs.items()):
        n_actions = len(freqs)
        top_k = min(15, n_actions)
        top_idx = np.argsort(freqs)[-top_k:][::-1]
        top_freqs = freqs[top_idx]

        labels = [action_names[i] if action_names else str(i) for i in top_idx]
        ax.barh(range(top_k), top_freqs)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Frequency")
        ax.set_title(name)
        ax.invert_yaxis()

    fig.suptitle(title)
    plt.tight_layout()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150, bbox_inches="tight")
        logging.info("Saved strategy analysis to %s", output)
    else:
        plt.show()
    plt.close()


def plot_comparative_results(config: config.ExperimentBatchConfig) -> None:
    """Generate comparative reward curves across all configurations."""
    save_dir = Path(config.save_dir)
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, object] = {}
    for strategy in config.strategies:
        for seed in config.seeds:
            metrics_path = (
                save_dir / f"{strategy.tag}_seed{seed}" / strategy.name / "metrics.json"
            )
            if metrics_path.exists():
                with open(metrics_path) as f:
                    all_metrics[f"{strategy.tag}_s{seed}"] = json.load(f)

    if not all_metrics:
        logging.warning("No metrics found. Run training first.")
        return

    plot_reward_curves(
        all_metrics,
        role="red",
        title="Red Reward Across All Experiments",
        output=str(results_dir / "all_red_rewards.png"),
    )
    plot_reward_curves(
        all_metrics,
        role="blue",
        title="Blue Reward Across All Experiments",
        output=str(results_dir / "all_blue_rewards.png"),
    )
    logging.info("Plots saved to %s/", results_dir)
