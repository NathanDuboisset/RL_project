"""Benchmark PPO greedy vs MCTS First Only vs MCTS Full Triplet on BlockBlast 3P
(report, Section 5.4, Figure 8). Needs a trained PPO checkpoint (not versioned).

    from ppo.ppo_agent import PPOTrainer
    trainer = PPOTrainer([BlockBlast3PEnv()]); trainer.load("checkpoints/ppo/....pt")
    run_benchmark(trainer.model, lambda: BlockBlast3PEnv(), n_episodes=100, value_weight=0.3)
"""
import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.evaluation import print_table, run_episodes, save_npz
from ppo.mcts_agent import MCTSAgent, MCTSAgentFirstOnly, PPOGreedy


def _plot_benchmark(results: dict, save_path: str):
    labels   = list(results.keys())
    colors   = ["steelblue", "darkorange", "forestgreen"]
    x        = np.arange(len(labels))

    means    = [results[k]["mean_return"]   for k in labels]
    stds     = [results[k]["std_return"]    for k in labels]
    medians  = [results[k]["median_return"] for k in labels]
    lengths  = [results[k]["mean_length"]   for k in labels]
    all_rets = [results[k]["returns"]       for k in labels]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("BlockBlast3P — Planning Strategy Benchmark", fontsize=14, fontweight="bold")

    bars = axes[0, 0].bar(x, means, yerr=stds, capsize=6,
                           color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, fontsize=10)
    axes[0, 0].set_title("Mean Episode Return (± std)")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, means):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds)*0.02,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    bars = axes[0, 1].bar(x, medians, color=colors, alpha=0.85,
                           edgecolor="black", linewidth=0.8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, fontsize=10)
    axes[0, 1].set_title("Median Episode Return")
    axes[0, 1].set_ylabel("Return")
    axes[0, 1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, medians):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(medians)*0.02,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    bars = axes[1, 0].bar(x, lengths, color=colors, alpha=0.85,
                           edgecolor="black", linewidth=0.8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, fontsize=10)
    axes[1, 0].set_title("Mean Episode Length (steps)")
    axes[1, 0].set_ylabel("Steps")
    axes[1, 0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, lengths):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(lengths)*0.02,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    parts = axes[1, 1].violinplot(all_rets, positions=x, showmedians=True, showextrema=True)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, fontsize=10)
    axes[1, 1].set_title("Return Distribution")
    axes[1, 1].set_ylabel("Return")
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved -> {save_path}")


def run_benchmark(
    model,
    env_fn:       Callable,
    device        = torch.device("cuda"),
    n_episodes:   int   = 100,
    gamma:        float = 0.99,
    value_weight: float = 0.0,
    save_dir:     str   = ".",
    seed:         int   = 0,
) -> dict:
    """value_weight only affects MCTS Full Triplet (alpha in the report)."""
    model = model.to(device).eval()
    policies = {
        "PPO Greedy":        PPOGreedy(model, device),
        "MCTS First Only":   MCTSAgentFirstOnly(model, device=device, gamma=gamma),
        "MCTS Full Triplet": MCTSAgent(model, device=device, gamma=gamma, value_weight=value_weight),
    }
    stats = {name: run_episodes(env_fn(), policy, n_episodes, seed=seed, desc=name)
             for name, policy in policies.items()}
    print_table(stats, f"BlockBlast 3P — {n_episodes} episodes, value_weight={value_weight}")

    results = {}
    for name, st in stats.items():
        s = st.summary()
        results[name] = {
            "returns":       st.returns,
            "lengths":       st.lengths,
            "mean_return":   s["return"]["mean"],
            "std_return":    s["return"]["std"],
            "median_return": s["return"]["median"],
            "mean_length":   s["length"]["mean"],
            "ms_per_decision": s["ms_per_decision"],
        }

    _plot_benchmark(results, os.path.join(save_dir, "benchmark_3p.png"))
    npz_path = os.path.join(save_dir, "benchmark_3p.npz")
    save_npz(npz_path, stats, seed=seed, value_weight=value_weight, gamma=gamma)
    print(f"Raw data saved -> {npz_path}")
    return results
