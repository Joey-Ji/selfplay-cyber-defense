"""Checkpoint pool management for self-play training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class CheckpointManager:
    """Manages a pool of model checkpoints for opponent sampling."""

    def __init__(self, save_dir: Path, role: str, max_pool_size: int = 200):
        self.save_dir = Path(save_dir) / role.lower()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.role = role
        self.max_pool_size = max_pool_size

        self._checkpoints: list[Path] = []
        self._metadata: dict[str, dict] = {}
        self._win_rates: dict[str, float] = {}

        self._meta_path = self.save_dir / "pool_metadata.json"
        self._load_existing()

    def _load_existing(self):
        """Load existing checkpoints from disk and prune stale metadata."""
        if self._meta_path.exists():
            with open(self._meta_path) as f:
                data = json.load(f)
            self._metadata = data.get("metadata", {})
            self._win_rates = data.get("win_rates", {})

        pts = sorted(self.save_dir.glob("checkpoint_*.pt"), key=lambda p: p.stem)
        self._checkpoints = pts

        on_disk = {p.name for p in pts}
        self._metadata = {k: v for k, v in self._metadata.items() if k in on_disk}
        self._win_rates = {k: v for k, v in self._win_rates.items() if k in on_disk}

    def _save_meta(self):
        with open(self._meta_path, "w") as f:
            json.dump({
                "metadata": self._metadata,
                "win_rates": self._win_rates,
            }, f, indent=2)

    def save_checkpoint(self, model, update_idx: int) -> Path:
        """Save model policy state_dict as .pt file."""
        path = self.save_dir / f"checkpoint_{update_idx:06d}.pt"
        state_dict = model.policy.state_dict()
        torch.save(state_dict, path)

        self._checkpoints.append(path)
        self._metadata[path.name] = {"update_idx": update_idx}

        # Evict oldest if pool is full
        while len(self._checkpoints) > self.max_pool_size:
            old = self._checkpoints.pop(0)
            self._metadata.pop(old.name, None)
            self._win_rates.pop(old.name, None)
            if old.exists():
                old.unlink()

        self._save_meta()
        return path

    @staticmethod
    def load_checkpoint(path: Path, model) -> None:
        """Load state_dict into an existing PPO model's policy."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.policy.load_state_dict(state_dict)

    def pool_size(self) -> int:
        return len(self._checkpoints)

    def latest(self) -> Optional[Path]:
        return self._checkpoints[-1] if self._checkpoints else None

    def sample_vanilla(self) -> Optional[Path]:
        """Return the latest checkpoint."""
        return self.latest()

    def sample_delta_uniform(self, delta: float) -> Optional[Path]:
        """
        Sample uniformly from the most recent (1-delta) fraction of pool.
        delta=1 -> vanilla (latest only), delta=0 -> full fictitious play.
        """
        if not self._checkpoints:
            return None
        n = len(self._checkpoints)
        window = max(1, int(n * (1 - delta)))
        start = n - window
        idx = np.random.randint(start, n)
        return self._checkpoints[idx]

    def sample_pfsp(self, p: float = 1.0) -> Optional[Path]:
        """
        Prioritized Fictitious Self-Play: sample proportional to (1 - win_rate)^p.
        Agents with low win rates (harder opponents) are sampled more.
        """
        if not self._checkpoints:
            return None

        weights = []
        for ckpt in self._checkpoints:
            wr = self._win_rates.get(ckpt.name, 0.5)
            weights.append((1.0 - wr) ** p)

        weights = np.array(weights, dtype=np.float64)
        total = weights.sum()
        if total < 1e-12:
            # Uniform fallback
            idx = np.random.randint(0, len(self._checkpoints))
        else:
            probs = weights / total
            idx = np.random.choice(len(self._checkpoints), p=probs)

        return self._checkpoints[idx]

    def record_win_rate(self, checkpoint_path: Path, win_rate: float):
        """Update win rate for a specific checkpoint."""
        self._win_rates[checkpoint_path.name] = float(win_rate)
        self._save_meta()

    def all_checkpoints(self) -> list[Path]:
        return list(self._checkpoints)

    def sample(self, strategy: str, delta: float = 0.5, p: float = 1.0) -> Optional[Path]:
        """Dispatch to the appropriate sampling strategy."""
        if strategy == "vanilla":
            return self.sample_vanilla()
        elif strategy == "delta_uniform":
            return self.sample_delta_uniform(delta)
        elif strategy == "pfsp":
            return self.sample_pfsp(p)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
