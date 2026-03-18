"""Group Relative Policy Optimization (GRPO) implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Policy network for GRPO, compatible with ActorCritic checkpoint format."""

    def __init__(self, obs_dim: int, act_n: int, net_arch: list[int] = (64, 64)):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_n = act_n

        # Use 'actor' naming for checkpoint compatibility with ActorCritic
        actor_layers = []
        prev = obs_dim
        for h in net_arch:
            actor_layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        actor_layers.append(nn.Linear(prev, act_n))
        self.actor = nn.Sequential(*actor_layers)

        # Dummy critic for checkpoint compatibility (not trained in GRPO)
        critic_layers = []
        prev = obs_dim
        for h in net_arch:
            critic_layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        critic_layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, value) for ActorCritic compatibility."""
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def get_action(
        self, obs: torch.Tensor, action_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and return action, log_prob, entropy."""
        logits, _ = self(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e8)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log_prob and entropy for given actions."""
        logits, _ = self(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e8)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()

    def get_log_probs(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get log probabilities for actions (used for KL computation)."""
        logits, _ = self(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e8)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions)

    def kl_divergence(
        self,
        obs: torch.Tensor,
        other: "PolicyNetwork",
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute KL(self || other) analytically for discrete actions."""
        logits, _ = self(obs)
        other_logits, _ = other(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e8)
            other_logits = other_logits.masked_fill(~action_mask.bool(), -1e8)

        p = torch.softmax(logits, dim=-1)
        log_p = torch.log_softmax(logits, dim=-1)
        log_q = torch.log_softmax(other_logits, dim=-1)

        # KL(p || q) = Σ p(a) * [log p(a) - log q(a)]
        kl = (p * (log_p - log_q)).sum(dim=-1)
        return kl


class TrajectoryBuffer:
    """Stores trajectory data for a single episode."""

    def __init__(self):
        self.obs: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.action_masks: list[np.ndarray | None] = []

    def add(self, obs, action, log_prob, reward, action_mask=None):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.action_masks.append(action_mask)

    def total_reward(self) -> float:
        return sum(self.rewards)

    def __len__(self) -> int:
        return len(self.obs)


class GroupBuffer:
    """Stores multiple trajectories and computes group-relative advantages."""

    def __init__(self):
        self.trajectories: list[TrajectoryBuffer] = []

    def add_trajectory(self, trajectory: TrajectoryBuffer):
        self.trajectories.append(trajectory)

    def compute_advantages(
        self, gamma: float = 0.99, normalize: bool = True
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Compute per-step advantages using group-relative returns.

        For GRPO, we compute the discounted return for each trajectory,
        then normalize across the group to get relative advantages.
        Each step in the trajectory receives the trajectory's relative advantage.
        """
        if not self.trajectories:
            return [], []

        # Compute discounted returns for each trajectory
        trajectory_returns = []
        per_step_returns = []

        for traj in self.trajectories:
            returns = []
            G = 0.0
            for r in reversed(traj.rewards):
                G = r + gamma * G
                returns.insert(0, G)
            trajectory_returns.append(returns[0] if returns else 0.0)
            per_step_returns.append(np.array(returns, dtype=np.float32))

        # Compute group statistics
        returns_array = np.array(trajectory_returns, dtype=np.float32)
        mean_return = returns_array.mean()
        std_return = returns_array.std() + 1e-8

        # Compute normalized advantages for each trajectory
        all_advantages = []
        all_returns = []

        for i, traj in enumerate(self.trajectories):
            if normalize and len(self.trajectories) > 1:
                # Group-relative advantage: how much better than group average
                traj_advantage = (trajectory_returns[i] - mean_return) / std_return
            else:
                traj_advantage = trajectory_returns[i] - mean_return

            # All steps in trajectory get same advantage
            step_returns = per_step_returns[i]
            advantages = np.full(len(traj), traj_advantage, dtype=np.float32)

            all_advantages.append(advantages)
            all_returns.append(step_returns)

        return all_advantages, all_returns

    def iterate_batches(
        self,
        advantages: list[np.ndarray],
        batch_size: int,
        device: torch.device,
    ):
        """Yield mini-batches from all trajectories."""
        # Flatten all trajectories
        all_obs = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_action_masks = []
        has_masks = False

        for i, traj in enumerate(self.trajectories):
            all_obs.extend(traj.obs)
            all_actions.extend(traj.actions)
            all_old_log_probs.extend(traj.log_probs)
            all_advantages.extend(advantages[i])
            all_action_masks.extend(traj.action_masks)
            if traj.action_masks[0] is not None:
                has_masks = True

        n = len(all_obs)
        indices = np.random.permutation(n)

        obs_arr = np.array(all_obs, dtype=np.float32)
        act_arr = np.array(all_actions, dtype=np.int64)
        old_lp_arr = np.array(all_old_log_probs, dtype=np.float32)
        adv_arr = np.array(all_advantages, dtype=np.float32)
        if has_masks:
            mask_arr = np.array(all_action_masks, dtype=np.float32)

        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            batch = {
                "obs": torch.tensor(obs_arr[idx], device=device),
                "actions": torch.tensor(act_arr[idx], device=device),
                "old_log_probs": torch.tensor(old_lp_arr[idx], device=device),
                "advantages": torch.tensor(adv_arr[idx], device=device),
                "action_masks": (
                    torch.tensor(mask_arr[idx], device=device) if has_masks else None
                ),
            }
            yield batch

    def clear(self):
        self.trajectories.clear()

    def total_steps(self) -> int:
        return sum(len(t) for t in self.trajectories)


class GRPO:
    """Group Relative Policy Optimization.

    GRPO computes advantages by comparing trajectory returns within a group,
    eliminating the need for a learned value function. It uses a KL penalty
    to the reference policy for stability.
    """

    def __init__(
        self,
        obs_dim: int,
        act_n: int,
        lr: float = 0.001,
        gamma: float = 0.99,
        clip_range: float = 0.2,
        n_epochs: int = 4,
        group_size: int = 8,
        batch_size: int = 64,
        ent_coef: float = 0.01,
        kl_coef: float = 0.1,
        max_grad_norm: float = 0.5,
        net_arch: list[int] = (64, 64),
        device: str = "auto",
        seed: int = 42,
    ):
        """Initialize GRPO.

        Args:
            obs_dim: Observation space dimension
            act_n: Number of discrete actions
            lr: Learning rate
            gamma: Discount factor
            clip_range: PPO-style clipping range (set to 0 to disable)
            n_epochs: Number of optimization epochs per update
            group_size: Number of trajectories per group for advantage estimation
            batch_size: Mini-batch size for optimization
            ent_coef: Entropy bonus coefficient
            kl_coef: KL penalty coefficient to reference policy
            max_grad_norm: Maximum gradient norm for clipping
            net_arch: Hidden layer sizes for policy network
            device: Device to use ('auto', 'cpu', or 'cuda')
            seed: Random seed
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        torch.manual_seed(seed)

        self.policy = PolicyNetwork(obs_dim, act_n, list(net_arch)).to(self.device)
        self.ref_policy = PolicyNetwork(obs_dim, act_n, list(net_arch)).to(self.device)
        self.ref_policy.load_state_dict(self.policy.state_dict())

        # Freeze reference policy
        for param in self.ref_policy.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.group_size = group_size
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.kl_coef = kl_coef
        self.max_grad_norm = max_grad_norm

        self.obs_dim = obs_dim
        self.act_n = act_n
        self._total_timesteps = 0
        self._updates = 0

    def collect_group_rollouts(self, env, n_trajectories: int) -> tuple[GroupBuffer, dict]:
        """Collect a group of trajectories for GRPO update. Returns GroupBuffer with collected trajectories and episode statistics"""
        buffer = GroupBuffer()
        episode_rewards = []

        self.policy.eval()
        with torch.no_grad():
            for _ in range(n_trajectories):
                trajectory = TrajectoryBuffer()
                obs, info = env.reset()
                action_mask = info.get("action_mask", None)
                done = False

                while not done:
                    obs_t = torch.tensor(
                        obs, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    mask_t = None
                    if action_mask is not None:
                        mask_t = torch.tensor(
                            action_mask, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)

                    action, log_prob, _ = self.policy.get_action(obs_t, mask_t)
                    action_int = action.item()
                    log_prob_f = log_prob.item()

                    next_obs, reward, terminated, truncated, info = env.step(action_int)
                    done = terminated or truncated

                    trajectory.add(obs, action_int, log_prob_f, reward, action_mask)

                    obs = next_obs
                    action_mask = info.get("action_mask", None)

                buffer.add_trajectory(trajectory)
                episode_rewards.append(trajectory.total_reward())
                self._total_timesteps += len(trajectory)

        # Compute statistics
        rewards_arr = np.array(episode_rewards)
        stats = {
            "mean_reward": float(rewards_arr.mean()),
            "std_reward": float(rewards_arr.std()),
            "win_rate": float((rewards_arr > 0).mean()),
            "n_episodes": len(episode_rewards),
        }

        return buffer, stats

    def update(self, buffer: GroupBuffer) -> dict:
        """Run GRPO update on collected group buffer."""
        self.policy.train()

        # Compute group-relative advantages (already normalized across group)
        advantages, _ = buffer.compute_advantages(self.gamma, normalize=True)

        total_pg_loss = 0.0
        total_kl_loss = 0.0
        total_ent = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for batch in buffer.iterate_batches(advantages, self.batch_size, self.device):
                # Get current policy log probs and entropy
                new_log_probs, entropy = self.policy.evaluate_actions(
                    batch["obs"], batch["actions"], batch["action_masks"]
                )

                # Use pre-computed group-relative advantages directly
                # (no per-batch normalization to preserve group structure)
                adv = batch["advantages"]

                # Policy gradient with optional clipping
                ratio = torch.exp(new_log_probs - batch["old_log_probs"])

                if self.clip_range > 0:
                    # PPO-style clipping
                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(
                        ratio, 1 - self.clip_range, 1 + self.clip_range
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                else:
                    # Vanilla policy gradient
                    pg_loss = -(adv * ratio).mean()

                # Compute proper KL divergence analytically
                kl = self.policy.kl_divergence(
                    batch["obs"], self.ref_policy, batch["action_masks"]
                )
                kl_loss = kl.mean()

                # Total loss
                loss = pg_loss + self.kl_coef * kl_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_kl_loss += kl_loss.item()
                total_ent += entropy.mean().item()
                n_updates += 1

        self._updates += 1

        return {
            "pg_loss": total_pg_loss / max(n_updates, 1),
            "kl_loss": total_kl_loss / max(n_updates, 1),
            "entropy": total_ent / max(n_updates, 1),
        }

    def update_reference_policy(self):
        """Update reference policy to current policy (call periodically)."""
        self.ref_policy.load_state_dict(self.policy.state_dict())

    def learn(
        self, env, total_timesteps: int, ref_update_freq: int = 10,
    ) -> dict:
        """Training loop: collect groups and update policy.

        Args:
            env: Gymnasium-compatible environment
            total_timesteps: Total environment steps to collect
            ref_update_freq: Update reference policy every N iterations

        Returns:
            Aggregated episode statistics
        """
        all_mean_rewards = []
        all_win_rates = []
        all_n_episodes = []
        steps_done = 0
        iteration = 0

        while steps_done < total_timesteps:
            buffer, stats = self.collect_group_rollouts(env, self.group_size)
            self.update(buffer)
            steps_done = self._total_timesteps

            if stats["n_episodes"] > 0:
                all_mean_rewards.append(stats["mean_reward"])
                all_win_rates.append(stats["win_rate"])
                all_n_episodes.append(stats["n_episodes"])

            iteration += 1
            if iteration % ref_update_freq == 0:
                self.update_reference_policy()

        if all_n_episodes:
            total_eps = sum(all_n_episodes)
            weighted_reward = (
                sum(r * n for r, n in zip(all_mean_rewards, all_n_episodes)) / total_eps
            )
            weighted_wr = (
                sum(w * n for w, n in zip(all_win_rates, all_n_episodes)) / total_eps
            )
            return {
                "mean_reward": float(weighted_reward),
                "std_reward": 0.0,
                "win_rate": float(weighted_wr),
                "n_episodes": total_eps,
            }
        return {"mean_reward": 0.0, "std_reward": 0.0, "win_rate": 0.0, "n_episodes": 0}

    def predict(
        self, obs: np.ndarray, deterministic: bool = False, action_mask: np.ndarray | None = None,
    ) -> tuple[int, None]:
        """Inference: return (action, None) matching SB3/PPO interface."""
        self.policy.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)

            logits, _ = self.policy(obs_t)

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
                "model_state_dict": self.policy.state_dict(),
                "ref_model_state_dict": self.ref_policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "obs_dim": self.obs_dim,
                "act_n": self.act_n,
                "total_timesteps": self._total_timesteps,
                "updates": self._updates,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu", obs_dim: int = 0, act_n: int = 0) -> "GRPO":
        data = torch.load(path, map_location=device, weights_only=False)
        if isinstance(data, dict) and "model_state_dict" in data:
            model = cls(obs_dim=data["obs_dim"], act_n=data["act_n"], device=device)
            model.policy.load_state_dict(data["model_state_dict"])
            if "ref_model_state_dict" in data:
                model.ref_policy.load_state_dict(data["ref_model_state_dict"])
            if "optimizer_state_dict" in data:
                model.optimizer.load_state_dict(data["optimizer_state_dict"])
            model._total_timesteps = data.get("total_timesteps", 0)
            model._updates = data.get("updates", 0)
        else:
            # Bare state_dict
            if obs_dim == 0 or act_n == 0:
                raise ValueError("obs_dim and act_n required when loading bare state_dict")
            model = cls(obs_dim=obs_dim, act_n=act_n, device=device)
            model.policy.load_state_dict(data)
        return model
