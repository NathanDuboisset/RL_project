import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from typing import Callable

from mct.mcts_agent import MCTSAgent
from mct.mcts_agent_first_only import MCTSAgentFirstOnly
from mct.ppo_agent import obs_to_tensors, valid_to_mask


def _run_ppo_greedy(model, env_fn, device, n_episodes):
    model.eval()
    returns, lengths = [], []

    for ep in range(n_episodes):
        env = env_fn()
        obs, _ = env.reset()
        total_r, n_steps = 0.0, 0

        while True:
            obs_t = {
                "board":       torch.as_tensor(obs["board"][None].astype(np.float32),       device=device),
                "pieces":      torch.as_tensor(obs["pieces"][None].astype(np.float32),      device=device),
                "pieces_used": torch.as_tensor(obs["pieces_used"][None].astype(np.float32), device=device),
                "combo":       torch.as_tensor(obs["combo"][None].astype(np.float32),       device=device),
            }
            mask_t = torch.as_tensor(
                valid_to_mask(obs["valid_placements"][None]), device=device
            )
            with torch.no_grad():
                actions, *_ = model.get_action(obs_t, mask_t, deterministic=True)
            action = int(actions[0])

            obs, r, term, trunc, _ = env.step(action)
            total_r += r
            n_steps += 1

            if term or trunc:
                break

        returns.append(total_r)
        lengths.append(n_steps)
        env.close()

        if (ep + 1) % 20 == 0:
            print(f"  [PPO greedy]      ep {ep+1:>4}/{n_episodes} | "
                  f"return {total_r:>8.1f} | len {n_steps:>4}")

    return np.array(returns), np.array(lengths)


def _run_mcts_first_only(model, env_fn, device, n_episodes, gamma, value_weight):
    agent = MCTSAgentFirstOnly(
        model      = model,
        device     = device,
        gamma      = gamma,
        batch_size = 512,
        verbose    = False,
    )
    returns, lengths, round_times = [], [], []

    for ep in range(n_episodes):
        env = env_fn()
        obs, _ = env.reset()
        total_r, n_steps = 0.0, 0

        while True:
            t0 = time.time()
            action = agent.select_action(env)
            round_times.append((time.time() - t0) * 1000)

            obs, r, term, trunc, _ = env.step(action)
            total_r += r
            n_steps += 1

            if term or trunc:
                break

        returns.append(total_r)
        lengths.append(n_steps)
        env.close()

        if (ep + 1) % 20 == 0:
            print(f"  [MCTS first-only] ep {ep+1:>4}/{n_episodes} | "
                  f"return {total_r:>8.1f} | len {n_steps:>4} | "
                  f"avg search {np.mean(round_times):.0f}ms")

    return np.array(returns), np.array(lengths), np.mean(round_times)


def _run_mcts_full_triplet(model, env_fn, device, n_episodes, gamma, value_weight):
    agent = MCTSAgent(
        model        = model,
        device       = device,
        gamma        = gamma,
        batch_size   = 512,
        verbose      = False,
        value_weight = value_weight,
    )
    returns, lengths, round_times = [], [], []

    for ep in range(n_episodes):
        env = env_fn()
        obs, _ = env.reset()
        total_r, n_steps = 0.0, 0
        action_queue = []

        while True:
            if not action_queue:
                t0 = time.time()
                action_queue = agent.select_round(env)
                round_times.append((time.time() - t0) * 1000)
                if not action_queue:
                    action_queue = [agent._greedy_fallback(env)]

            action = action_queue.pop(0)
            obs, r, term, trunc, _ = env.step(action)
            total_r += r
            n_steps += 1

            if term or trunc:
                action_queue = []
                break

            if np.all(obs["pieces_used"] == 0):
                action_queue = []

        returns.append(total_r)
        lengths.append(n_steps)
        env.close()

        if (ep + 1) % 20 == 0:
            print(f"  [MCTS full triplet] ep {ep+1:>4}/{n_episodes} | "
                  f"return {total_r:>8.1f} | len {n_steps:>4} | "
                  f"avg search {np.mean(round_times):.0f}ms")

    return np.array(returns), np.array(lengths), np.mean(round_times)


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
) -> dict:
    model = model.to(device)
    model.eval()

    results = {}

    print(f"\n{'='*60}")
    print(f"1/3 — PPO Greedy ({n_episodes} episodes)")
    print(f"{'='*60}")
    t0 = time.time()
    rets, lens = _run_ppo_greedy(model, env_fn, device, n_episodes)
    results["PPO Greedy"] = {
        "returns":       rets,
        "lengths":       lens,
        "mean_return":   float(np.mean(rets)),
        "std_return":    float(np.std(rets)),
        "median_return": float(np.median(rets)),
        "mean_length":   float(np.mean(lens)),
        "time_min":      (time.time() - t0) / 60,
        "ms_per_round":  0.0,
    }

    print(f"\n{'='*60}")
    print(f"2/3 — MCTS First Action Only ({n_episodes} episodes)")
    print(f"{'='*60}")
    t0 = time.time()
    rets, lens, ms = _run_mcts_first_only(model, env_fn, device, n_episodes, gamma, value_weight)
    results["MCTS First Only"] = {
        "returns":       rets,
        "lengths":       lens,
        "mean_return":   float(np.mean(rets)),
        "std_return":    float(np.std(rets)),
        "median_return": float(np.median(rets)),
        "mean_length":   float(np.mean(lens)),
        "time_min":      (time.time() - t0) / 60,
        "ms_per_round":  ms,
    }

    print(f"\n{'='*60}")
    print(f"3/3 — MCTS Full Triplet ({n_episodes} episodes)")
    print(f"{'='*60}")
    t0 = time.time()
    rets, lens, ms = _run_mcts_full_triplet(model, env_fn, device, n_episodes, gamma, value_weight)
    results["MCTS Full Triplet"] = {
        "returns":       rets,
        "lengths":       lens,
        "mean_return":   float(np.mean(rets)),
        "std_return":    float(np.std(rets)),
        "median_return": float(np.median(rets)),
        "mean_length":   float(np.mean(lens)),
        "time_min":      (time.time() - t0) / 60,
        "ms_per_round":  ms,
    }

    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS — {n_episodes} episodes, value_weight={value_weight}")
    print(f"{'='*60}")
    print(f"{'Strategy':<22} | {'Mean Ret':>9} | {'±Std':>8} | {'Median':>8} | "
          f"{'MeanLen':>8} | {'ms/round':>9} | {'vs PPO':>8}")
    print("─" * 90)

    ppo_mean = results["PPO Greedy"]["mean_return"]
    for name, r in results.items():
        delta = f"{(r['mean_return'] - ppo_mean) / max(abs(ppo_mean), 1) * 100:+.1f}%" \
                if name != "PPO Greedy" else "baseline"
        print(
            f"{name:<22} | {r['mean_return']:>9.1f} | {r['std_return']:>8.1f} | "
            f"{r['median_return']:>8.1f} | {r['mean_length']:>8.1f} | "
            f"{r['ms_per_round']:>9.1f} | {delta:>8}"
        )

    plot_path = os.path.join(save_dir, "benchmark_3p.png")
    _plot_benchmark(results, plot_path)

    npz_path = os.path.join(save_dir, "benchmark_3p.npz")
    np.savez(npz_path, **{
        f"{k.replace(' ', '_')}_{metric}": v
        for k, r in results.items()
        for metric, v in [("returns", r["returns"]), ("lengths", r["lengths"])]
    })
    print(f"Raw data saved -> {npz_path}")

    return results
