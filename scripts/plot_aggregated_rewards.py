"""Plot Phase 1 aggregated Red rewards (single panel) for the final report.

Reads metrics.json from batch1 (vanilla, delta_0.7, pfsp) and plots
Red mean reward with ±1 std bands over 500 rounds.

Usage:
    python scripts/plot_aggregated_rewards.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("out/models/batch1")
OUTPUT = Path("docs/final_report/figures/aggregated_rewards.png")

STRATEGIES = {
    "Vanilla": "vanilla",
    r"$\delta=0.7$": "delta_uniform_d0.7",
    "PFSP": "pfsp",
}
SEEDS = [42, 123, 456]
COLORS = ["#1f77b4", "#2ca02c", "#d62728"]


def rolling_mean(arr, window=10):
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(arr, (pad, window - 1 - pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main():
    fig, ax = plt.subplots(figsize=(7, 4))

    for (label, tag), color in zip(STRATEGIES.items(), COLORS):
        series = []
        for seed in SEEDS:
            path = ROOT / f"{tag}_seed{seed}" / "metrics.json"
            if not path.exists():
                print(f"  Missing: {path}")
                continue
            with open(path) as f:
                metrics = json.load(f)
            series.append([m["red_mean_reward"] for m in metrics])

        if not series:
            print(f"Skipping {label}: no data")
            continue

        arr = np.array(series)
        rounds = np.arange(1, arr.shape[1] + 1)
        smoothed = np.array([rolling_mean(row, 10) for row in arr])
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)

        ax.plot(rounds, mean, color=color, linewidth=1.8, label=label)
        ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Training Round")
    ax.set_ylabel("Red Mean Reward")
    ax.set_title("Phase 1: Red Mean Reward (3 seeds)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
