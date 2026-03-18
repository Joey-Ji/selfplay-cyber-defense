"""Anchor sweep plots — ICML single-column format, SVG output."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 0.8,
})

# ICML single-column: 3.25 in wide
COL_W = 3.25

CONFIGS = [
    {
        "name": "anchor_p03_meander",
        "label": "$p{=}0.3$, meander-biased",
        "short": "$p{=}0.3$ M",
        "color": "#2ca02c",
        "marker": "s",
    },
    {
        "name": "anchor_p05_meander",
        "label": "$p{=}0.5$, meander-biased",
        "short": "$p{=}0.5$ M",
        "color": "#d62728",
        "marker": "D",
    },
    {
        "name": "anchor_p07_meander",
        "label": "$p{=}0.7$, meander-biased",
        "short": "$p{=}0.7$ M",
        "color": "#9467bd",
        "marker": "^",
    },
    {
        "name": "anchor_p05_equal",
        "label": "$p{=}0.5$, equal mix",
        "short": "$p{=}0.5$ E",
        "color": "#1f77b4",
        "marker": "o",
    },
]

SEEDS = [42, 123, 456]
DOCS_DIR = Path("docs/experiments/anchor")
FIG_DIR = DOCS_DIR / "figures"


def load_per_seed_scripted(config_name: str) -> dict:
    summary = json.loads(
        Path(f"out/results_{config_name}/summary.json").read_text()
    )
    per_seed = summary["eval_comparison"]["per_seed"]
    bline_vals = []
    meander_vals = []
    for s in per_seed:
        bline_vals.append(s["opponents"]["red_bline"]["anchor_red_reward_mean"])
        meander_vals.append(s["opponents"]["red_meander"]["anchor_red_reward_mean"])
    return {"bline": bline_vals, "meander": meander_vals}


def load_baseline_per_seed() -> dict:
    bline_vals = []
    meander_vals = []
    for seed in SEEDS:
        data = json.loads(
            Path(f"out/results_anchor_baseline_delta07/eval_seed{seed}.json").read_text()
        )
        for m in data["matchups"]:
            if m["opponent_spec"] == "red_bline":
                bline_vals.append(m["red_reward_mean"])
            if m["opponent_spec"] == "red_meander":
                meander_vals.append(m["red_reward_mean"])
    return {"bline": bline_vals, "meander": meander_vals}


def plot_scripted_transfer() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.85))

    baseline = load_baseline_per_seed()
    b_mean = np.mean(baseline["bline"])
    m_mean = np.mean(baseline["meander"])

    # Baseline individual seeds
    ax.scatter(
        baseline["bline"], baseline["meander"],
        marker="x", s=16, color="#666666", alpha=0.5, zorder=2, linewidths=0.6,
    )
    # Baseline mean
    ax.scatter(
        [b_mean], [m_mean],
        marker="X", s=50, color="#222222", zorder=4, linewidths=0.5,
    )
    ax.errorbar(
        b_mean, m_mean,
        xerr=np.std(baseline["bline"], ddof=1),
        yerr=np.std(baseline["meander"], ddof=1),
        color="#222222", capsize=2, capthick=0.7, linewidth=0.7, zorder=2,
    )

    # Manual label offsets to avoid overlap: (dx, dy) in points
    label_offsets = {
        "anchor_p03_meander": (6, 4),
        "anchor_p05_meander": (6, -12),
        "anchor_p07_meander": (6, 4),
        "anchor_p05_equal": (6, -12),
    }

    for cfg in CONFIGS:
        data = load_per_seed_scripted(cfg["name"])
        b_m = np.mean(data["bline"])
        m_m = np.mean(data["meander"])
        b_s = np.std(data["bline"], ddof=1)
        m_s = np.std(data["meander"], ddof=1)

        # Individual seeds
        ax.scatter(
            data["bline"], data["meander"],
            marker=cfg["marker"], s=12, color=cfg["color"], alpha=0.35,
            zorder=2, linewidths=0.4,
        )
        # Mean
        ax.scatter(
            [b_m], [m_m],
            marker=cfg["marker"], s=40, color=cfg["color"],
            edgecolor="white", linewidth=0.5, zorder=4,
        )
        # Error bars
        ax.errorbar(
            b_m, m_m, xerr=b_s, yerr=m_s,
            color=cfg["color"], capsize=2, capthick=0.7, linewidth=0.7,
            zorder=2, alpha=0.6,
        )
        dx, dy = label_offsets[cfg["name"]]
        ax.annotate(
            cfg["short"], (b_m, m_m),
            textcoords="offset points", xytext=(dx, dy),
            fontsize=6.5, color=cfg["color"], fontweight="medium",
        )

    # Baseline label
    ax.annotate(
        "Baseline", (b_mean, m_mean),
        textcoords="offset points", xytext=(6, 6),
        fontsize=6.5, color="#222222", fontweight="medium",
    )

    ax.set_xlabel(r"B-line Red reward $\downarrow$")
    ax.set_ylabel(r"Meander Red reward $\downarrow$")
    ax.grid(alpha=0.15, linewidth=0.4)

    # Add "better" arrow in corner
    ax.annotate(
        "", xy=(0.08, 0.08), xytext=(0.22, 0.22),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8),
    )
    ax.text(
        0.06, 0.06, "better", transform=ax.transAxes,
        fontsize=6, color="#999999", ha="center", style="italic",
    )

    fig.savefig(FIG_DIR / "scripted_transfer_tradeoff_errorbars.svg", format="svg")
    fig.savefig(FIG_DIR / "scripted_transfer_tradeoff_errorbars.png", dpi=300)
    plt.close(fig)
    print(f"Saved scripted_transfer_tradeoff_errorbars.{{svg,png}}")


def plot_internal_vs_external() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline internal Red WR
    baseline_analysis = json.loads(Path("out/results/analysis.json").read_text())
    baseline_wr_str = baseline_analysis["comparison"]["delta_uniform_d0.7"]["red_wr_final"]
    baseline_internal = float(baseline_wr_str.split("+/-")[0].strip())

    # Baseline external
    baseline_external_seeds = []
    for seed in SEEDS:
        data = json.loads(
            Path(f"out/results_anchor_baseline_delta07/eval_seed{seed}.json").read_text()
        )
        baseline_external_seeds.append(data["average_red_reward_eval_set"])
    baseline_external = np.mean(baseline_external_seeds)

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.85))

    # Baseline
    ax.scatter(
        [baseline_internal], [baseline_external],
        marker="X", s=50, color="#222222", zorder=4, linewidths=0.5,
    )
    ax.errorbar(
        baseline_internal, baseline_external,
        yerr=np.std(baseline_external_seeds, ddof=1),
        color="#222222", capsize=2, capthick=0.7, linewidth=0.7, zorder=2,
    )
    ax.annotate(
        "Baseline", (baseline_internal, baseline_external),
        textcoords="offset points", xytext=(6, 6),
        fontsize=6.5, color="#222222", fontweight="medium",
    )

    # Manual offsets: configs at x~0.65 are crowded
    label_offsets = {
        "anchor_p03_meander": (-42, 8),
        "anchor_p05_meander": (6, -10),
        "anchor_p07_meander": (-42, -10),
        "anchor_p05_equal": (-42, 4),
    }

    for cfg in CONFIGS:
        summary = json.loads(
            Path(f"out/results_{cfg['name']}/summary.json").read_text()
        )
        wr_str = summary["anchor_analysis"]["comparison"]["delta_uniform_d0.7"]["red_wr_final"]
        internal = float(wr_str.split("+/-")[0].strip())

        per_seed = summary["eval_comparison"]["per_seed"]
        external_seeds = [s["average_red_reward_eval_set_anchor"] for s in per_seed]
        external = np.mean(external_seeds)
        external_std = np.std(external_seeds, ddof=1)

        ax.scatter(
            [internal], [external],
            marker=cfg["marker"], s=40, color=cfg["color"],
            edgecolor="white", linewidth=0.5, zorder=4,
        )
        ax.errorbar(
            internal, external, yerr=external_std,
            color=cfg["color"], capsize=2, capthick=0.7, linewidth=0.7,
            zorder=2, alpha=0.6,
        )
        # Individual seeds
        ax.scatter(
            [internal] * len(external_seeds), external_seeds,
            marker=cfg["marker"], s=10, color=cfg["color"], alpha=0.3,
            zorder=2, linewidths=0.3,
        )
        dx, dy = label_offsets[cfg["name"]]
        ax.annotate(
            cfg["short"], (internal, external),
            textcoords="offset points", xytext=(dx, dy),
            fontsize=6.5, color=cfg["color"], fontweight="medium",
        )

    ax.set_xlabel("Final internal Red win rate")
    ax.set_ylabel(r"Eval-set avg Red reward $\downarrow$")
    ax.grid(alpha=0.15, linewidth=0.4)

    fig.savefig(FIG_DIR / "internal_vs_external_errorbars.svg", format="svg")
    fig.savefig(FIG_DIR / "internal_vs_external_errorbars.png", dpi=300)
    plt.close(fig)
    print(f"Saved internal_vs_external_errorbars.{{svg,png}}")


def main() -> None:
    plot_scripted_transfer()
    plot_internal_vs_external()


if __name__ == "__main__":
    main()
