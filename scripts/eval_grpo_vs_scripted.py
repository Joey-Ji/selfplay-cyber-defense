"""Evaluate GRPO agents against scripted red opponents (B-line, Meander).

Produces heatmaps of blue win rate and mean reward for each GRPO strategy
vs each scripted red agent, aggregated over seeds.

Usage::

    uv run python scripts/eval_grpo_vs_scripted.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from selfplay.env import BLUE_OBS_DIM, BLUE_ACT_N
from selfplay.eval.evaluate import evaluate, PPOAgent
from selfplay.training.ppo import PPO as CustomPPO
from selfplay.utils import make_scripted_agent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_DIR = Path("out/models_grpo")
OUTPUT_DIR = Path("out")
SEEDS = [42, 123, 456]
NUM_EPISODES = 200
MAX_STEPS = 100

STRATEGIES = [
    ("Vanilla",   "vanilla",        "vanilla"),
    ("Delta 0.3", "delta_uniform_d0.3", "delta_uniform"),
    ("Delta 0.5", "delta_uniform_d0.5", "delta_uniform"),
    ("Delta 0.7", "delta_uniform_d0.7", "delta_uniform"),
    ("PFSP",      "pfsp",           "pfsp"),
]

SCRIPTED_REDS = [
    ("B-line",  "red_bline"),
    ("Meander", "red_meander"),
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_grpo_blue(tag: str, strategy_name: str, seed: int) -> PPOAgent | None:
    """Load the final GRPO blue agent for a given strategy + seed."""
    path = MODELS_DIR / f"{tag}_seed{seed}" / strategy_name / "blue" / "latest.pt"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return None
    model = CustomPPO(obs_dim=BLUE_OBS_DIM, act_n=BLUE_ACT_N, device="cpu")
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model.actor_critic.load_state_dict(state_dict["model_state_dict"])
    else:
        model.actor_critic.load_state_dict(state_dict)
    return PPOAgent(model)


def run_evaluation() -> tuple[np.ndarray, np.ndarray]:
    """Return (win_rate_matrix, reward_matrix) shaped (n_strategies, n_scripted)."""
    n_strat = len(STRATEGIES)
    n_red = len(SCRIPTED_REDS)

    win_rate_matrix = np.zeros((n_strat, n_red))
    reward_matrix = np.zeros((n_strat, n_red))
    red_reward_matrix = np.zeros((n_strat, n_red))

    for i, (label, tag, name) in enumerate(STRATEGIES):
        for j, (red_label, red_name) in enumerate(SCRIPTED_REDS):
            blue_win_rates = []
            blue_rewards = []
            red_rewards = []

            for seed in SEEDS:
                blue_agent = load_grpo_blue(tag, name, seed)
                if blue_agent is None:
                    continue

                red_agent = make_scripted_agent(red_name)
                print(f"  Evaluating {label} seed={seed} vs {red_label} ...", flush=True)

                results = evaluate(
                    red_agent=red_agent,
                    blue_agent=blue_agent,
                    num_episodes=NUM_EPISODES,
                    max_steps=MAX_STEPS,
                )

                blue_win_rates.append(1.0 - results["red_win_rate"])
                blue_rewards.append(results["blue_reward_mean"])
                red_rewards.append(results["red_reward_mean"])

            if blue_win_rates:
                win_rate_matrix[i, j] = float(np.mean(blue_win_rates))
                reward_matrix[i, j] = float(np.mean(blue_rewards))
                red_reward_matrix[i, j] = float(np.mean(red_rewards))
            else:
                win_rate_matrix[i, j] = float("nan")
                reward_matrix[i, j] = float("nan")
                red_reward_matrix[i, j] = float("nan")

    return win_rate_matrix, reward_matrix, red_reward_matrix


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cbar_label: str,
    fmt: str,
    cmap: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=0,
        vmax=1 if "Win" in cbar_label else None,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        cbar_kws={"label": cbar_label},
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Scripted Red Opponent", fontsize=11)
    ax.set_ylabel("GRPO Blue Strategy", fontsize=11)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def plot_combined(
    win_matrix: np.ndarray,
    reward_matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    sns.heatmap(
        win_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=axes[0],
        cbar_kws={"label": "Blue Win Rate"},
        linewidths=0.5,
    )
    axes[0].set_title("GRPO vs Scripted Red\n(Blue Win Rate)", fontsize=12)
    axes[0].set_xlabel("Scripted Red Opponent")
    axes[0].set_ylabel("GRPO Blue Strategy")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        reward_matrix,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=axes[1],
        cbar_kws={"label": "Blue Mean Reward"},
        linewidths=0.5,
    )
    axes[1].set_title("GRPO vs Scripted Red\n(Blue Mean Reward)", fontsize=12)
    axes[1].set_xlabel("Scripted Red Opponent")
    axes[1].set_ylabel("GRPO Blue Strategy")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].tick_params(axis="y", rotation=0)

    plt.suptitle("GRPO Blue Agents vs Scripted Red Opponents", fontsize=13, y=1.02)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    strategy_labels = [s[0] for s in STRATEGIES]
    red_labels = [r[0] for r in SCRIPTED_REDS]

    print("=== GRPO vs Scripted Red Evaluation ===")
    win_matrix, reward_matrix, red_reward_matrix = run_evaluation()

    print("\nBlue Win Rate matrix:")
    for i, lbl in enumerate(strategy_labels):
        row = "  ".join(f"{win_matrix[i, j]:.3f}" for j in range(len(red_labels)))
        print(f"  {lbl:12s}: {row}")

    print("\nBlue Mean Reward matrix:")
    for i, lbl in enumerate(strategy_labels):
        row = "  ".join(f"{reward_matrix[i, j]:7.2f}" for j in range(len(red_labels)))
        print(f"  {lbl:12s}: {row}")

    print("\nRed Mean Reward matrix:")
    for i, lbl in enumerate(strategy_labels):
        row = "  ".join(f"{red_reward_matrix[i, j]:7.2f}" for j in range(len(red_labels)))
        print(f"  {lbl:12s}: {row}")

    # Save raw results
    np.savez(
        OUTPUT_DIR / "grpo_vs_scripted.npz",
        win_rate=win_matrix,
        blue_reward=reward_matrix,
        red_reward=red_reward_matrix,
        strategy_labels=strategy_labels,
        red_labels=red_labels,
    )

    # Individual heatmaps
    plot_heatmap(
        win_matrix,
        strategy_labels,
        red_labels,
        title="GRPO Blue vs Scripted Red (Blue Win Rate)",
        cbar_label="Blue Win Rate",
        fmt=".2f",
        cmap="RdYlGn",
        output=OUTPUT_DIR / "grpo_vs_scripted_winrate.png",
    )
    plot_heatmap(
        reward_matrix,
        strategy_labels,
        red_labels,
        title="GRPO Blue vs Scripted Red (Blue Mean Reward)",
        cbar_label="Blue Mean Reward",
        fmt=".1f",
        cmap="RdYlGn",
        output=OUTPUT_DIR / "grpo_vs_scripted_reward.png",
    )
    # Red reward heatmap: smaller = green (blue doing better)
    plot_heatmap(
        red_reward_matrix,
        strategy_labels,
        red_labels,
        title="GRPO Blue vs Scripted Red (Red Mean Reward)",
        cbar_label="Red Mean Reward",
        fmt=".1f",
        cmap="RdYlGn_r",
        output=OUTPUT_DIR / "grpo_vs_scripted_red_reward.png",
    )

    # Combined figure
    plot_combined(
        win_matrix,
        reward_matrix,
        strategy_labels,
        red_labels,
        output=OUTPUT_DIR / "grpo_vs_scripted.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
