import numpy as np
import torch
import matplotlib.pyplot as plt
from mct.mcts_agent import MCTSAgent


def run_sweep(
    model,
    env_fn,
    device          = torch.device("cuda"),
    n_episodes: int = 50,
    weights: list   = None,
    plot_path: str  = None,
) -> dict:
    if weights is None:
        weights = [0.0, 0.05, 0.1, 0.3, 0.5, 1.0]

    results = {}

    print(f"Value weight sweep — {len(weights)} values × {n_episodes} episodes\n")
    print(f"{'Weight':>8} | {'Mean Ret':>10} | {'Median Ret':>11} | "
          f"{'Mean Len':>9} | {'Std Ret':>9} | {'ms/round':>9}")
    print("─" * 70)

    for w in weights:
        agent = MCTSAgent(
            model        = model,
            device       = device,
            gamma        = 0.99,
            batch_size   = 512,
            verbose      = False,
            value_weight = w,
        )

        stats = agent.evaluate(env_fn, n_episodes=n_episodes, use_mcts=True)
        results[w] = stats

        print(
            f"{w:>8.3f} | {stats['mean_return']:>10.2f} | "
            f"{stats['median_return']:>11.2f} | "
            f"{stats['mean_length']:>9.1f} | "
            f"{stats['std_return']:>9.2f} | "
            f"{stats['mean_time_per_round_ms']:>9.1f}"
        )

    best_w     = max(results, key=lambda w: results[w]["mean_return"])
    best_stats = results[best_w]

    print(f"\n=== Best value_weight : {best_w} ===")
    print(f"  Mean return   : {best_stats['mean_return']:.2f}")
    print(f"  Median return : {best_stats['median_return']:.2f}")
    print(f"  Mean length   : {best_stats['mean_length']:.1f} steps")

    if plot_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("MCTS Value Weight Sweep", fontsize=14)

        wl = [str(w) for w in weights]

        means = [results[w]["mean_return"] for w in weights]
        stds  = [results[w]["std_return"]  for w in weights]
        axes[0].bar(wl, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
        axes[0].axhline(means[0], color='red', linestyle='--', alpha=0.5, label='w=0 baseline')
        axes[0].set_title("Mean Return")
        axes[0].set_xlabel("value_weight")
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        medians = [results[w]["median_return"] for w in weights]
        axes[1].bar(wl, medians, color='orange', alpha=0.8)
        axes[1].axhline(medians[0], color='red', linestyle='--', alpha=0.5, label='w=0 baseline')
        axes[1].set_title("Median Return")
        axes[1].set_xlabel("value_weight")
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

        lengths = [results[w]["mean_length"] for w in weights]
        axes[2].bar(wl, lengths, color='green', alpha=0.8)
        axes[2].axhline(lengths[0], color='red', linestyle='--', alpha=0.5, label='w=0 baseline')
        axes[2].set_title("Mean Episode Length")
        axes[2].set_xlabel("value_weight")
        axes[2].legend()
        axes[2].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"\nPlot saved -> {plot_path}")

    return results
