"""Evaluate one Blue checkpoint for anchored-self-play pilot reporting.

Outputs:
  - Blue performance vs scripted B-line and Meander
  - Blue performance vs held-out learned Red checkpoints
  - Average Red reward across the eval set
  - One Blue action histogram snapshot
  - Host-level Blue defense allocation snapshot
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from absl import app, flags

from selfplay.env import SimplifiedCAGE
from selfplay.env.agents import BaseAgent
from selfplay.eval.evaluate import load_agent

FLAGS = flags.FLAGS
flags.DEFINE_string("blue", None, "Blue agent checkpoint (.zip).", required=True)
flags.DEFINE_multi_string(
    "heldout_red",
    [],
    "Held-out learned Red checkpoints (.zip). Pass multiple times for multiple opponents.",
)
flags.DEFINE_integer("episodes", 200, "Episodes per opponent.")
flags.DEFINE_integer("max_steps", 100, "Max steps per episode.")
flags.DEFINE_integer("seed", 42, "Numpy RNG seed.")
flags.DEFINE_integer("top_k", 12, "Top-k actions/hosts to save in snapshot.")
flags.DEFINE_string(
    "snapshot_opponent",
    "red_bline",
    "Opponent label to use for action/host telemetry snapshot.",
)
flags.DEFINE_string(
    "output",
    "out/results_anchor_pilot/anchor_eval.json",
    "Where to save JSON evaluation output.",
)


def _matchup_label(spec: str, idx: int) -> str:
    if spec in {"red_bline", "red_meander"}:
        return spec
    return f"heldout_{idx}_{Path(spec).stem}"


def _top_counts(counter: Counter[str], top_k: int) -> list[dict]:
    total = sum(counter.values())
    if total == 0:
        return []
    rows = []
    for key, count in counter.most_common(top_k):
        rows.append(
            {
                "name": key,
                "count": int(count),
                "fraction": float(count / total),
            }
        )
    return rows


def evaluate_with_snapshot(
    red_agent,
    blue_agent,
    num_episodes: int,
    max_steps: int,
    top_k: int,
) -> dict:
    sim = SimplifiedCAGE(num_envs=num_episodes, remove_bugs=True)
    action_names = sim.action_mapping["Blue"]
    action_counts = np.zeros(len(action_names), dtype=np.int64)
    host_counts: Counter[str] = Counter()
    host_action_counts: Counter[str] = Counter()

    if hasattr(red_agent, "reset"):
        red_agent.reset()
    if hasattr(blue_agent, "reset"):
        blue_agent.reset()

    obs_dict, _ = sim.reset()
    red_obs = obs_dict["Red"]
    blue_obs = obs_dict["Blue"]

    total_red_reward = np.zeros(num_episodes, dtype=np.float64)
    total_blue_reward = np.zeros(num_episodes, dtype=np.float64)

    for _ in range(max_steps):
        red_action = np.asarray(red_agent.get_action(red_obs), dtype=np.int32).reshape(
            num_episodes, 1
        )
        blue_action = np.asarray(
            blue_agent.get_action(blue_obs), dtype=np.int32
        ).reshape(num_episodes, 1)

        flat_blue = blue_action[:, 0]
        action_counts += np.bincount(flat_blue, minlength=len(action_names))

        for a in flat_blue:
            name = action_names[int(a)]
            if "_" not in name:
                continue
            verb, host = name.split("_", 1)
            host_counts[host] += 1
            host_action_counts[f"{host}:{verb}"] += 1

        obs_dict, reward_dict, _, _ = sim.step(
            red_action=red_action,
            blue_action=blue_action,
            red_agent=red_agent if isinstance(red_agent, BaseAgent) else None,
        )
        red_obs = obs_dict["Red"]
        blue_obs = obs_dict["Blue"]
        total_red_reward += reward_dict["Red"].reshape(-1)
        total_blue_reward += reward_dict["Blue"].reshape(-1)

    red_reward_mean = float(total_red_reward.mean())
    red_reward_std = float(total_red_reward.std())
    blue_reward_mean = float(total_blue_reward.mean())
    blue_reward_std = float(total_blue_reward.std())
    red_win_rate = float((total_red_reward > 0).mean())

    action_total = int(action_counts.sum())
    top_action_idx = np.argsort(action_counts)[::-1][:top_k]
    top_actions = [
        {
            "action_id": int(i),
            "action_name": action_names[int(i)],
            "count": int(action_counts[int(i)]),
            "fraction": float(action_counts[int(i)] / action_total)
            if action_total
            else 0.0,
        }
        for i in top_action_idx
        if action_counts[int(i)] > 0
    ]

    return {
        "red_reward_mean": red_reward_mean,
        "red_reward_std": red_reward_std,
        "blue_reward_mean": blue_reward_mean,
        "blue_reward_std": blue_reward_std,
        "red_win_rate": red_win_rate,
        "num_episodes": int(num_episodes),
        "snapshot": {
            "top_actions": top_actions,
            "top_hosts": _top_counts(host_counts, top_k),
            "top_host_actions": _top_counts(host_action_counts, top_k),
        },
    }


def main(_):
    np.random.seed(FLAGS.seed)

    blue_agent = load_agent(FLAGS.blue, "blue")
    opponent_specs = ["red_bline", "red_meander", *FLAGS.heldout_red]

    rows = []
    snapshot_label = FLAGS.snapshot_opponent.strip().lower()
    snapshot = None
    first_snapshot = None

    for idx, spec in enumerate(opponent_specs, start=1):
        label = _matchup_label(spec, idx)
        red_agent = load_agent(spec, "red")
        result = evaluate_with_snapshot(
            red_agent=red_agent,
            blue_agent=blue_agent,
            num_episodes=FLAGS.episodes,
            max_steps=FLAGS.max_steps,
            top_k=FLAGS.top_k,
        )

        row = {
            "opponent": label,
            "opponent_spec": spec,
            "red_reward_mean": result["red_reward_mean"],
            "red_reward_std": result["red_reward_std"],
            "blue_reward_mean": result["blue_reward_mean"],
            "blue_reward_std": result["blue_reward_std"],
            "red_win_rate": result["red_win_rate"],
            "num_episodes": result["num_episodes"],
        }
        rows.append(row)
        if first_snapshot is None:
            first_snapshot = {"opponent": label, **result["snapshot"]}

        if snapshot is None and label.lower() == snapshot_label:
            snapshot = {"opponent": label, **result["snapshot"]}

        print(
            f"{label:24s} | Red {row['red_reward_mean']:8.2f} +/- {row['red_reward_std']:7.2f} "
            f"| Blue {row['blue_reward_mean']:8.2f} +/- {row['blue_reward_std']:7.2f} "
            f"| Red WR {row['red_win_rate']:.3f}"
        )

    if snapshot is None:
        snapshot = first_snapshot

    avg_red_reward = (
        float(np.mean([r["red_reward_mean"] for r in rows])) if rows else float("nan")
    )
    output = {
        "blue_agent": FLAGS.blue,
        "episodes_per_opponent": FLAGS.episodes,
        "max_steps": FLAGS.max_steps,
        "matchups": rows,
        "average_red_reward_eval_set": avg_red_reward,
        "action_snapshot": snapshot,
    }

    out_path = Path(FLAGS.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nAverage Red reward over eval set: {avg_red_reward:.2f}")
    print(f"Saved anchored pilot eval to {out_path}")


if __name__ == "__main__":
    app.run(main)
