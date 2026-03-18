"""Plot Red-only ablation learning curves for the final report.

Based on scripts/plot_ablation.py but removes the Blue reward twin axis
and uses shared y-axis range across all 4 subplots.

Usage:
    python scripts/plot_ablation_red_only.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("out_vm")
OUTPUT = Path("docs/final_report/figures/ablation_learning_curves.png")
WINDOW = 20

ABLATION_RUNS = {
    "No BC, No Curriculum": "models_ablation_no_bc_no_curriculum",
    "BC Only": "models_ablation_bc_only",
    "BC + Curriculum (p=0.3)": "models_final",
    "BC + Curriculum (p=0.5)": "models_ablation_bc_curriculum_p05",
}

COLORS = ["#1f3b73", "#2a9d8f", "#e07a5f", "#7a5c61"]


def rolling_mean(values, window):
    if window <= 1:
        return values.copy()
    kernel = np.ones(window) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    red_ceiling = 2500.0

    for ax, color, (label, folder) in zip(axes, COLORS, ABLATION_RUNS.items()):
        paths = sorted(ROOT.joinpath(folder).glob("vanilla_seed*/vanilla/metrics.json"))
        red_series = []
        for path in paths:
            with open(path) as f:
                metrics = json.load(f)
            red_series.append([step["red_mean_reward"] for step in metrics])

        if not red_series:
            print(f"Skipping {label}: no metrics found")
            ax.set_title(label)
            continue

        red_arr = np.array(red_series, dtype=float)
        rounds = np.arange(1, red_arr.shape[1] + 1)

        # Raw seed traces
        for seed_idx in range(red_arr.shape[0]):
            ax.plot(rounds, red_arr[seed_idx], color=color, alpha=0.15, linewidth=0.9)

        # Smoothed mean + std
        smoothed = np.array([rolling_mean(row, WINDOW) for row in red_arr])
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)

        ax.plot(rounds, mean, color=color, linewidth=2.2, label="Red reward")
        ax.fill_between(
            rounds,
            np.clip(mean - std, 0, red_ceiling),
            np.clip(mean + std, 0, red_ceiling),
            color=color, alpha=0.2,
        )

        ax.set_title(label)
        ax.set_xlabel("Training Round")
        ax.set_ylabel("Red Reward")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, red_ceiling)

    fig.suptitle("Ablation Learning Curves: Red Reward (rolling mean, window=20)", fontsize=13)
    plt.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
