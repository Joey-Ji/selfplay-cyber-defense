"""Cross-play evaluation: compute reward matrices across checkpoint pools."""

from __future__ import annotations

from pathlib import Path

from absl import logging
import numpy as np

from selfplay import config
from selfplay.env import RED_OBS_DIM, RED_ACT_N, BLUE_OBS_DIM, BLUE_ACT_N
from selfplay.eval.evaluate import evaluate, PPOAgent
from selfplay.utils import load_ppo
from selfplay.viz.plots import plot_crossplay_heatmap


def compute_crossplay_matrix(
    red_ckpts: list[Path],
    blue_ckpts: list[Path],
    num_episodes: int = 200,
    max_steps: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    n_red = len(red_ckpts)
    n_blue = len(blue_ckpts)
    red_matrix = np.zeros((n_red, n_blue))
    blue_matrix = np.zeros((n_red, n_blue))

    total_pairs = n_red * n_blue
    done = 0

    for i, red_path in enumerate(red_ckpts):
        red_model = load_ppo(red_path, obs_dim=RED_OBS_DIM, act_n=RED_ACT_N)
        red_agent = PPOAgent(red_model)

        for j, blue_path in enumerate(blue_ckpts):
            blue_model = load_ppo(blue_path, obs_dim=BLUE_OBS_DIM, act_n=BLUE_ACT_N)
            blue_agent = PPOAgent(blue_model)

            results = evaluate(
                red_agent, blue_agent,
                num_episodes=num_episodes,
                max_steps=max_steps,
            )

            red_matrix[i, j] = results["red_reward_mean"]
            blue_matrix[i, j] = results["blue_reward_mean"]

            done += 1
            if done % 10 == 0:
                logging.info("Evaluated %d/%d pairs", done, total_pairs)

    return red_matrix, blue_matrix


def evaluate_crossplay(config: config.ExperimentBatchConfig) -> None:
    save_dir = Path(config.save_dir)
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    n = config.eval.crossplay_checkpoints

    for strategy in config.strategies:
        for seed in config.seeds:
            exp_dir = save_dir / f"{strategy.tag}_seed{seed}" / strategy.name
            red_dir = exp_dir / "red"
            blue_dir = exp_dir / "blue"

            if not red_dir.exists() or not blue_dir.exists():
                logging.info("Skipping %s seed=%d (no checkpoints)", strategy.tag, seed)
                continue

            logging.info("Cross-play: %s seed=%d", strategy.tag, seed)
            red_ckpts = sorted(red_dir.glob("checkpoint_*.pt"))
            blue_ckpts = sorted(blue_dir.glob("checkpoint_*.pt"))

            if len(red_ckpts) > n:
                idx = np.linspace(0, len(red_ckpts) - 1, n, dtype=int)
                red_ckpts = [red_ckpts[i] for i in idx]
            if len(blue_ckpts) > n:
                idx = np.linspace(0, len(blue_ckpts) - 1, n, dtype=int)
                blue_ckpts = [blue_ckpts[i] for i in idx]

            red_mat, blue_mat = compute_crossplay_matrix(
                red_ckpts,
                blue_ckpts,
                num_episodes=config.eval.crossplay_episodes,
            )
            out_path = results_dir / f"crossplay_{strategy.tag}_s{seed}.npz"
            np.savez(out_path, red_matrix=red_mat, blue_matrix=blue_mat)
            plot_crossplay_heatmap(
                red_mat,
                title=f"Cross-Play: {strategy.tag} seed={seed}",
                output=str(results_dir / f"crossplay_{strategy.tag}_s{seed}.png"),
            )

    logging.info("Cross-play evaluation complete.")
