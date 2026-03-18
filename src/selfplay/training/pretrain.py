"""Pre-train policy via behavioral cloning from expert demonstrations."""

from __future__ import annotations

from pathlib import Path

from absl import logging
import numpy as np
import torch
import torch.nn as nn

from selfplay.training.ppo import ActorCritic


def pretrain(
    data_path: str,
    output_path: str,
    obs_dim: int,
    act_n: int,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
):
    """Train ActorCritic policy network with cross-entropy loss on demos."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    data = np.load(data_path)
    obs = torch.tensor(data["observations"], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.long)
    logging.info("Loaded %d transitions from %s", len(obs), data_path)

    model = ActorCritic(obs_dim, act_n)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    dataset_size = len(obs)
    n_batches = dataset_size // batch_size

    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        obs_shuffled = obs[perm]
        act_shuffled = actions[perm]

        total_loss = 0.0
        correct = 0

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            batch_obs = obs_shuffled[start:end]
            batch_act = act_shuffled[start:end]

            logits = model.actor(batch_obs)

            loss = loss_fn(logits, batch_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_act).sum().item()

        avg_loss = total_loss / max(n_batches, 1)
        accuracy = correct / max(n_batches * batch_size, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logging.info("Epoch %d/%d: loss=%.4f, accuracy=%.3f", epoch + 1, epochs, avg_loss, accuracy)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logging.info("Pre-trained model saved to %s", output_path)
    return model
