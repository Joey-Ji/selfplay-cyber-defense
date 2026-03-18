"""
Self-play training loop for Red vs Blue PPO agents in MiniCAGE.

Supports three opponent-sampling strategies:
  - vanilla: train against latest opponent
  - delta_uniform: sample from recent (1-delta) fraction of pool
  - pfsp: prioritized fictitious self-play (AlphaStar-style)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from absl import logging
import numpy as np
import torch

from selfplay import config
from selfplay.env import HOSTS
from selfplay.env.wrappers import SelfPlayBlueWrapper, SelfPlayRedWrapper
from selfplay.eval.evaluate import evaluate, PPOAgent
from selfplay.utils import make_scripted_agent
from selfplay.training.checkpoints import CheckpointManager
from selfplay.training.ppo import PPO, ActorCritic
from selfplay.training.grpo import GRPO


def make_red_env(max_steps: int = 100):
    env = SelfPlayRedWrapper(remove_bugs=True, max_steps=max_steps)
    return env


def make_blue_env(max_steps: int = 100):
    env = SelfPlayBlueWrapper(remove_bugs=True, max_steps=max_steps)
    return env


def create_ppo(env, ppo_cfg: config.PPOConfig, seed: int, device: str = "auto") -> PPO:
    obs_dim = env.observation_space.shape[0]
    act_n = env.action_space.n
    return PPO(
        obs_dim=obs_dim,
        act_n=act_n,
        lr=ppo_cfg.learning_rate,
        gamma=ppo_cfg.gamma,
        clip_range=ppo_cfg.clip_range,
        n_epochs=ppo_cfg.n_epochs,
        n_steps=ppo_cfg.n_steps,
        batch_size=ppo_cfg.batch_size,
        ent_coef=ppo_cfg.ent_coef,
        vf_coef=ppo_cfg.vf_coef,
        max_grad_norm=ppo_cfg.max_grad_norm,
        gae_lambda=ppo_cfg.gae_lambda,
        device=device,
        seed=seed,
    )


def create_grpo(env, grpo_cfg: config.GRPOConfig, seed: int, device: str = "auto") -> GRPO:
    obs_dim = env.observation_space.shape[0]
    act_n = env.action_space.n
    return GRPO(
        obs_dim=obs_dim,
        act_n=act_n,
        lr=grpo_cfg.learning_rate,
        gamma=grpo_cfg.gamma,
        clip_range=grpo_cfg.clip_range,
        n_epochs=grpo_cfg.n_epochs,
        group_size=grpo_cfg.group_size,
        batch_size=grpo_cfg.batch_size,
        ent_coef=grpo_cfg.ent_coef,
        kl_coef=grpo_cfg.kl_coef,
        max_grad_norm=grpo_cfg.max_grad_norm,
        device=device,
        seed=seed,
    )


def create_model(env, cfg: config.ExperimentConfig, seed: int, device: str = "auto"):
    """Create PPO or GRPO model based on config.algorithm."""
    if cfg.algorithm == "grpo":
        return create_grpo(env, cfg.grpo, seed, device)
    else:
        return create_ppo(env, cfg.ppo, seed, device)


def set_opponent_in_env(env, checkpoint_path):
    env.set_opponent(checkpoint_path)


def sample_blue_training_opponent(
    red_pool: CheckpointManager,
    sp_cfg: config.SelfPlayConfig,
) -> tuple[Path | str | None, str]:
    """Sample a Red opponent for Blue training with optional scripted anchors."""
    learned = red_pool.sample(sp_cfg.strategy, delta=sp_cfg.delta, p=sp_cfg.pfsp_p)
    scripted_prob = float(np.clip(sp_cfg.blue_scripted_opponent_prob, 0.0, 1.0))
    scripted = [name for name in sp_cfg.blue_scripted_opponents if name]

    if scripted and np.random.rand() < scripted_prob:
        weights = np.asarray(sp_cfg.blue_scripted_opponent_weights, dtype=np.float64)
        valid = (
            (len(weights) == len(scripted))
            and np.all(weights >= 0)
            and (weights.sum() > 0)
        )
        probs = (
            weights / weights.sum()
            if valid
            else np.ones(len(scripted), dtype=np.float64) / len(scripted)
        )
        return str(np.random.choice(scripted, p=probs)), "scripted"

    if learned is None and scripted:
        return str(np.random.choice(scripted)), "scripted"

    return learned, "learned"


def quick_eval(
    model, opponent_model, role: str, num_episodes: int = 50, max_steps: int = 100
):
    """Quick evaluation of model vs opponent. Returns win rate for model's side."""

    class _Agent:
        def __init__(self, ppo_model):
            self.model = ppo_model
        def get_action(self, observation, *args, **kwargs):
            a, _ = self.model.predict(observation)
            return np.array(a).reshape(-1, 1).astype(np.int32)
        def reset(self):
            pass

    if role == "Red":
        red = _Agent(model)
        blue = _Agent(opponent_model)
    else:
        red = _Agent(opponent_model)
        blue = _Agent(model)

    results = evaluate(red, blue, num_episodes=num_episodes, max_steps=max_steps)
    if role == "Red":
        return results["red_win_rate"]
    else:
        return 1.0 - results["red_win_rate"]


def update_pfsp_win_rates(
    model: PPO,
    opponent_pool: CheckpointManager,
    own_role: str,
    opponent_role: str,
    eval_episodes: int = 50,
    max_steps: int = 100,
):
    """Evaluate current model against each checkpoint in the opponent pool."""
    from selfplay.env import BLUE_ACT_N, BLUE_OBS_DIM, RED_ACT_N, RED_OBS_DIM
    from selfplay.utils import load_ppo

    if opponent_role == "Red":
        obs_dim, act_n = RED_OBS_DIM, RED_ACT_N
    else:
        obs_dim, act_n = BLUE_OBS_DIM, BLUE_ACT_N

    for ckpt_path in opponent_pool.all_checkpoints():
        opp_model = load_ppo(ckpt_path, obs_dim, act_n)
        wr = quick_eval(model, opp_model, own_role, eval_episodes, max_steps)
        opponent_pool.record_win_rate(ckpt_path, wr)


def run_selfplay(cfg: config.ExperimentConfig):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    sp = cfg.selfplay
    strategy = sp.strategy
    save_dir = Path(cfg.save_dir) / strategy
    save_dir.mkdir(parents=True, exist_ok=True)

    red_pool = CheckpointManager(save_dir, "red", sp.max_pool_size)
    blue_pool = CheckpointManager(save_dir, "blue", sp.max_pool_size)

    red_env = make_red_env(sp.max_steps)
    blue_env = make_blue_env(sp.max_steps)

    red_model = create_model(red_env, cfg, seed=cfg.seed, device="auto")
    blue_model = create_model(blue_env, cfg, seed=cfg.seed + 1, device="auto")

    # Load pre-trained weights if available (config path or convention)
    def _load_pretrained(label, model, pt_path):
        logging.info("Loading pre-trained %s from %s", label, pt_path)
        sd = torch.load(pt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            model.policy.load_state_dict(sd["model_state_dict"])
        else:
            model.policy.load_state_dict(sd)

    for label, model, path_attr in [
        ("Red", red_model, sp.red_pretrain_path),
        ("Blue", blue_model, sp.blue_pretrain_path),
    ]:
        pt_path = Path(path_attr) if path_attr else None
        if pt_path and pt_path.exists():
            _load_pretrained(label, model, pt_path)
        else:
            fallback = Path(cfg.save_dir) / f"{label.lower()}_pretrained.pt"
            if fallback.exists():
                _load_pretrained(label, model, fallback)

    wandb_run = None
    if cfg.wandb.use_wandb:
        import wandb

        wandb_config = {
            "strategy": strategy,
            "delta": sp.delta,
            "seed": cfg.seed,
            "total_rounds": sp.total_rounds,
            "algorithm": cfg.algorithm,
        }
        if cfg.algorithm == "grpo":
            wandb_config.update({
                "grpo_lr": cfg.grpo.learning_rate,
                "grpo_group_size": cfg.grpo.group_size,
                "grpo_kl_coef": cfg.grpo.kl_coef,
            })
        else:
            wandb_config["ppo_lr"] = cfg.ppo.learning_rate

        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group or f"selfplay_{strategy}",
            name=f"{cfg.algorithm}_{strategy}_seed{cfg.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=wandb_config,
        )

    red_pool.save_checkpoint(red_model, 0)
    blue_pool.save_checkpoint(blue_model, 0)

    metrics_log = []

    # Compute steps per round based on algorithm
    if cfg.algorithm == "grpo":
        # For GRPO: estimate based on group_size * max_steps (avg episode length)
        steps_per_round = cfg.grpo.group_size * sp.max_steps * sp.rounds_per_side
    else:
        steps_per_round = cfg.ppo.n_steps * sp.rounds_per_side

    for rnd in range(1, sp.total_rounds + 1):
        # --- Red training ---
        blue_ckpt = blue_pool.sample(strategy, delta=sp.delta, p=sp.pfsp_p)
        set_opponent_in_env(red_env, blue_ckpt)
        if cfg.algorithm == "grpo":
            red_stats = red_model.learn(
                red_env, total_timesteps=steps_per_round,
                ref_update_freq=cfg.grpo.ref_update_freq
            )
        else:
            red_stats = red_model.learn(red_env, total_timesteps=steps_per_round)

        if rnd % sp.checkpoint_freq == 0:
            red_pool.save_checkpoint(red_model, rnd)

        # --- Blue training (with optional scripted opponent anchoring) ---
        red_opp, red_opp_type = sample_blue_training_opponent(red_pool, sp)
        if red_opp_type == "scripted":
            blue_env.set_scripted_opponent(make_scripted_agent(red_opp))
        else:
            blue_env.set_scripted_opponent(None)
            set_opponent_in_env(blue_env, red_opp)
        if cfg.algorithm == "grpo":
            blue_stats = blue_model.learn(
                blue_env, total_timesteps=steps_per_round,
                ref_update_freq=cfg.grpo.ref_update_freq
            )
        else:
            blue_stats = blue_model.learn(blue_env, total_timesteps=steps_per_round)

        if rnd % sp.checkpoint_freq == 0:
            blue_pool.save_checkpoint(blue_model, rnd)

        if strategy == "pfsp" and rnd % 5 == 0:
            update_pfsp_win_rates(
                red_model,
                blue_pool,
                "Red",
                "Blue",
                eval_episodes=sp.pfsp_eval_episodes,
                max_steps=sp.max_steps,
            )
            update_pfsp_win_rates(
                blue_model,
                red_pool,
                "Blue",
                "Red",
                eval_episodes=sp.pfsp_eval_episodes,
                max_steps=sp.max_steps,
            )

        metrics = {
            "round": rnd,
            "red_mean_reward": red_stats["mean_reward"],
            "red_win_rate": red_stats["win_rate"],
            "blue_mean_reward": blue_stats["mean_reward"],
            "blue_win_rate": blue_stats["win_rate"],
            "red_episodes": red_stats["n_episodes"],
            "blue_episodes": blue_stats["n_episodes"],
            "blue_opponent_type": red_opp_type,
            "blue_opponent_id": red_opp.name
            if isinstance(red_opp, Path)
            else str(red_opp),
        }
        metrics_log.append(metrics)

        if rnd % 10 == 0 or rnd == 1:
            logging.info(
                "Round %d/%d: R=%.1f Rwr=%.2f B=%.1f Bwr=%.2f opp=%s",
                rnd, sp.total_rounds,
                red_stats["mean_reward"], red_stats["win_rate"],
                blue_stats["mean_reward"], blue_stats["win_rate"],
                red_opp_type,
            )

        if wandb_run is not None:
            import wandb

            wandb.log(metrics, step=rnd)

    red_model.save(save_dir / "red" / "latest.pt")
    blue_model.save(save_dir / "blue" / "latest.pt")

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    if wandb_run is not None:
        wandb_run.finish()

    logging.info("Self-play complete. Models saved to %s", save_dir)
    return red_model, blue_model, metrics_log
