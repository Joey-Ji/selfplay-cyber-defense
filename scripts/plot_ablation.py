"""Plot smoothed Blue and Red reward learning curves for the ablations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("root", "out_vm", "Root directory containing VM experiment outputs.")
flags.DEFINE_string(
    "output",
    "out_vm/results/ablation_learning_curves.png",
    "Path to the output plot.",
)
flags.DEFINE_integer("window", 20, "Rolling-average window size.")
flags.DEFINE_float("blue_floor", -500.0, "Lower y-limit for Blue reward plots.")
flags.DEFINE_float("blue_ceiling", 50.0, "Upper y-limit for Blue reward plots.")
flags.DEFINE_float("red_ceiling", 2500.0, "Upper y-limit for Red reward plots.")

ABLATION_RUNS = {
    "No BC, No Curriculum": "models_ablation_no_bc_no_curriculum",
    "BC Only": "models_ablation_bc_only",
    "BC + Curriculum (p=0.3)": "models_final",
    "BC + Curriculum (p=0.5)": "models_ablation_bc_curriculum_p05",
}


def _load_metrics(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def _metrics_paths(root: Path, folder: str) -> list[Path]:
    return sorted(root.joinpath(folder).glob("vanilla_seed*/vanilla/metrics.json"))


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main(_):
    root = Path(FLAGS.root)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()
    colors = ["#1f3b73", "#2a9d8f", "#e07a5f", "#7a5c61"]

    for ax, color, (label, folder) in zip(axes, colors, ABLATION_RUNS.items()):
        blue_series = []
        red_series = []
        for path in _metrics_paths(root, folder):
            metrics = _load_metrics(path)
            blue_series.append([step["blue_mean_reward"] for step in metrics])
            red_series.append([step["red_mean_reward"] for step in metrics])

        if not blue_series:
            print(f"Skipping {label}: no metrics found")
            continue

        blue_arr = np.array(blue_series, dtype=float)
        red_arr = np.array(red_series, dtype=float)
        rounds = np.arange(1, blue_arr.shape[1] + 1)

        for seed_idx in range(blue_arr.shape[0]):
            ax.plot(
                rounds,
                np.clip(blue_arr[seed_idx], FLAGS.blue_floor, FLAGS.blue_ceiling),
                color=color,
                alpha=0.15,
                linewidth=0.9,
            )

        blue_smoothed = np.array([_rolling_mean(row, FLAGS.window) for row in blue_arr])
        blue_mean = np.clip(blue_smoothed.mean(axis=0), FLAGS.blue_floor, FLAGS.blue_ceiling)
        blue_std = blue_smoothed.std(axis=0)

        ax.plot(rounds, blue_mean, color=color, linewidth=2.4, label="Blue reward")
        ax.fill_between(
            rounds,
            np.clip(blue_mean - blue_std, FLAGS.blue_floor, FLAGS.blue_ceiling),
            np.clip(blue_mean + blue_std, FLAGS.blue_floor, FLAGS.blue_ceiling),
            color=color,
            alpha=0.2,
        )
        ax.set_ylim(FLAGS.blue_floor, FLAGS.blue_ceiling)
        ax.set_title(label)
        ax.set_xlabel("Training Round")
        ax.set_ylabel("Blue Reward", color=color)
        ax.tick_params(axis="y", colors=color)
        ax.grid(True, alpha=0.25)

        red_color = "#a83232"
        twin = ax.twinx()
        for seed_idx in range(red_arr.shape[0]):
            twin.plot(rounds, red_arr[seed_idx], color=red_color, alpha=0.12, linewidth=0.9)

        red_smoothed = np.array([_rolling_mean(row, FLAGS.window) for row in red_arr])
        red_mean = red_smoothed.mean(axis=0)
        red_std = red_smoothed.std(axis=0)
        twin.plot(rounds, red_mean, color=red_color, linewidth=2.0, label="Red reward")
        twin.fill_between(
            rounds,
            np.clip(red_mean - red_std, 0, FLAGS.red_ceiling),
            np.clip(red_mean + red_std, 0, FLAGS.red_ceiling),
            color=red_color,
            alpha=0.12,
        )
        twin.set_ylim(0, FLAGS.red_ceiling)
        twin.set_ylabel("Red Reward", color=red_color)
        twin.tick_params(axis="y", colors=red_color)

    fig.suptitle("Ablation Learning Curves (raw seeds + rolling mean, window=20)")
    plt.tight_layout()

    output = Path(FLAGS.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ablation plot to {output}")


if __name__ == "__main__":
    app.run(main)
