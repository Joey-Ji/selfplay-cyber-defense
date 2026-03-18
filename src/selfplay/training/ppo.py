"""Custom PPO implementation from scratch with action masking support.

Replaces stable-baselines3 PPO with a minimal, transparent implementation
using GAE and clipped surrogate objective.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Separate actor-critic with optional action masking."""

    def __init__(self, obs_dim: int, act_n: int, net_arch: list[int] = (64, 64)):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_n = act_n

        # Actor
        actor_layers = []
        prev = obs_dim
        for h in net_arch:
            actor_layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        actor_layers.append(nn.Linear(prev, act_n))
        self.actor = nn.Sequential(*actor_layers)

        # Critic
        critic_layers = []
        prev = obs_dim
        for h in net_arch:
            critic_layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        critic_layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def get_action(
        self, obs: torch.Tensor, action_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e8)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e8)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy()


class RolloutBuffer:
    """Stores rollout data and computes GAE."""

    def __init__(self):
        self.obs: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []
        self.action_masks: list[np.ndarray | None] = []

    def add(self, obs, action, log_prob, reward, value, done, action_mask=None):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def compute_gae(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        n = len(self.rewards)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            done_mask = 1 - int(self.dones[t])
            delta = self.rewards[t] + gamma * next_value * done_mask - self.values[t]
            gae = delta + gamma * gae_lambda * done_mask * gae
            self.advantages[t] = gae
        self.returns = self.advantages + np.array(self.values, dtype=np.float32)

    def iterate_batches(self, batch_size: int, device: torch.device):
        n = len(self.obs)
        indices = np.random.permutation(n)
        has_masks = self.action_masks[0] is not None

        obs_arr = np.array(self.obs, dtype=np.float32)
        act_arr = np.array(self.actions, dtype=np.int64)
        old_lp_arr = np.array(self.log_probs, dtype=np.float32)
        if has_masks:
            mask_arr = np.array(self.action_masks, dtype=np.float32)

        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            batch = {
                "obs": torch.tensor(obs_arr[idx], device=device),
                "actions": torch.tensor(act_arr[idx], device=device),
                "old_log_probs": torch.tensor(old_lp_arr[idx], device=device),
                "advantages": torch.tensor(self.advantages[idx], device=device),
                "returns": torch.tensor(self.returns[idx], device=device),
                "action_masks": (
                    torch.tensor(mask_arr[idx], device=device) if has_masks else None
                ),
            }
            yield batch

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.action_masks.clear()


class PPO:
    """Proximal Policy Optimization with clipped surrogate and GAE."""

    def __init__(
        self,
        obs_dim: int,
        act_n: int,
        lr: float = 0.002,
        gamma: float = 0.99,
        clip_range: float = 0.2,
        n_epochs: int = 6,
        n_steps: int = 2048,
        batch_size: int = 64,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        gae_lambda: float = 0.95,
        net_arch: list[int] = (64, 64),
        device: str = "auto",
        seed: int = 42,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        torch.manual_seed(seed)

        self.actor_critic = ActorCritic(obs_dim, act_n, list(net_arch)).to(self.device)
        self.policy = self.actor_critic  # compatibility with CheckpointManager
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda

        self.obs_dim = obs_dim
        self.act_n = act_n
        self._total_timesteps = 0

    def collect_rollouts(self, env, n_steps: int) -> tuple[RolloutBuffer, dict]:
        """Collect n_steps of experience. Returns buffer and episode stats."""
        buffer = RolloutBuffer()
        episode_rewards = []
        ep_reward = 0.0

        obs, info = env.reset()
        action_mask = info.get("action_mask", None)

        self.actor_critic.eval()
        with torch.no_grad():
            for _ in range(n_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                mask_t = None
                if action_mask is not None:
                    mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

                action, log_prob, value, _ = self.actor_critic.get_action(obs_t, mask_t)
                action_int = action.item()
                log_prob_f = log_prob.item()
                value_f = value.item()

                next_obs, reward, terminated, truncated, info = env.step(action_int)
                done = terminated or truncated
                ep_reward += reward

                buffer.add(obs, action_int, log_prob_f, reward, value_f, done, action_mask)

                if done:
                    episode_rewards.append(ep_reward)
                    ep_reward = 0.0
                    obs, info = env.reset()
                    action_mask = info.get("action_mask", None)
                else:
                    obs = next_obs
                    action_mask = info.get("action_mask", None)

            # Compute last value for GAE
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, last_value = self.actor_critic(obs_t)
            last_value = last_value.item()

        buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        if episode_rewards:
            rewards_arr = np.array(episode_rewards)
            stats = {
                "mean_reward": float(rewards_arr.mean()),
                "std_reward": float(rewards_arr.std()),
                "win_rate": float((rewards_arr > 0).mean()),
                "n_episodes": len(episode_rewards),
            }
        else:
            stats = {"mean_reward": 0.0, "std_reward": 0.0, "win_rate": 0.0, "n_episodes": 0}

        return buffer, stats

    def update(self, buffer: RolloutBuffer) -> dict:
        """Run PPO update on collected buffer."""
        self.actor_critic.train()

        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_ent = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for batch in buffer.iterate_batches(self.batch_size, self.device):
                new_log_probs, new_values, entropy = self.actor_critic.evaluate_actions(
                    batch["obs"], batch["actions"], batch["action_masks"]
                )

                advantages = batch["advantages"]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss = nn.functional.mse_loss(new_values, batch["returns"])

                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_ent += entropy.mean().item()
                n_updates += 1

        return {
            "pg_loss": total_pg_loss / max(n_updates, 1),
            "vf_loss": total_vf_loss / max(n_updates, 1),
            "entropy": total_ent / max(n_updates, 1),
        }

    def learn(self, env, total_timesteps: int) -> dict:
        """Outer training loop: collect rollouts + update. Returns episode stats."""
        all_mean_rewards = []
        all_win_rates = []
        all_n_episodes = []
        steps_done = 0

        while steps_done < total_timesteps:
            n = min(self.n_steps, total_timesteps - steps_done)
            buffer, stats = self.collect_rollouts(env, n)
            self.update(buffer)
            steps_done += n
            self._total_timesteps += n
            if stats["n_episodes"] > 0:
                all_mean_rewards.append(stats["mean_reward"])
                all_win_rates.append(stats["win_rate"])
                all_n_episodes.append(stats["n_episodes"])

        if all_n_episodes:
            total_eps = sum(all_n_episodes)
            weighted_reward = sum(r * n for r, n in zip(all_mean_rewards, all_n_episodes)) / total_eps
            weighted_wr = sum(w * n for w, n in zip(all_win_rates, all_n_episodes)) / total_eps
            return {
                "mean_reward": float(weighted_reward),
                "std_reward": 0.0,
                "win_rate": float(weighted_wr),
                "n_episodes": total_eps,
            }
        return {"mean_reward": 0.0, "std_reward": 0.0, "win_rate": 0.0, "n_episodes": 0}

    def predict(
        self, obs: np.ndarray, deterministic: bool = False, action_mask: np.ndarray | None = None
    ) -> tuple[int, None]:
        """Inference: return (action, None) matching SB3 interface."""
        self.actor_critic.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)

            logits, _ = self.actor_critic(obs_t)

            if action_mask is not None:
                mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device)
                if mask_t.dim() == 1:
                    mask_t = mask_t.unsqueeze(0)
                logits = logits.masked_fill(~mask_t.bool(), -1e8)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()

        action = action.cpu()
        if action.numel() == 1:
            return int(action.item()), None
        return action.numpy(), None

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "obs_dim": self.obs_dim,
                "act_n": self.act_n,
                "total_timesteps": self._total_timesteps,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu", obs_dim: int = 0, act_n: int = 0) -> "PPO":
        data = torch.load(path, map_location=device, weights_only=False)
        if isinstance(data, dict) and "model_state_dict" in data:
            # Full save format from PPO.save()
            model = cls(obs_dim=data["obs_dim"], act_n=data["act_n"], device=device)
            model.actor_critic.load_state_dict(data["model_state_dict"])
            if "optimizer_state_dict" in data:
                model.optimizer.load_state_dict(data["optimizer_state_dict"])
            model._total_timesteps = data.get("total_timesteps", 0)
        else:
            # Bare state_dict (checkpoint pool format)
            if obs_dim == 0 or act_n == 0:
                raise ValueError("obs_dim and act_n required when loading bare state_dict")
            model = cls(obs_dim=obs_dim, act_n=act_n, device=device)
            model.actor_critic.load_state_dict(data)
        return model
