import argparse
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Allow running the script directly from src/dvn without requiring editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blockblast.block_blast_env import BlockBlastEnv
from src.dvn.agent import DVNAgent1P


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark DVN vs greedy on BlockBlast 1P and compare reward/episode length distributions."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to DVN checkpoint (.pt).",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Number of evaluation episodes.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum number of steps per episode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for DVN inference (e.g. cpu, cuda).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base seed for deterministic episode resets.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory where plots are saved.",
    )
    return parser.parse_args()


def load_dvn_agent(checkpoint_path: Path, device: str) -> DVNAgent1P:
    agent = DVNAgent1P(device=device)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.policy_net.eval()
    agent.target_net.eval()
    return agent


def random_action(obs: dict, rng: np.random.Generator) -> int:
    valid_mask = obs["valid_placements"].reshape(-1).astype(bool)
    valid_actions = np.flatnonzero(valid_mask)
    if valid_actions.size == 0:
        return 0
    return int(rng.choice(valid_actions))


def greedy_action(obs: dict) -> int:
    valid_mask = obs["valid_placements"].reshape(-1).astype(bool)
    valid_actions = np.flatnonzero(valid_mask)
    if valid_actions.size == 0:
        return 0

    hyp_rewards = obs["placements_result"][1].reshape(-1)
    best_local_idx = int(np.argmax(hyp_rewards[valid_actions]))
    return int(valid_actions[best_local_idx])


def dvn_action(agent: DVNAgent1P, obs: dict) -> int:
    return int(agent.select_action(obs, epsilon=0.0))


def run_policy(
    env: BlockBlastEnv,
    policy_name: str,
    episodes: int,
    max_steps: int,
    seed: int,
    agent: DVNAgent1P | None = None,
) -> dict[str, np.ndarray]:
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    step_rewards: list[float] = []
    rng = np.random.default_rng(seed=seed)

    for ep in tqdm(range(episodes), desc=f"Eval {policy_name}"):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0

        for step in range(max_steps):
            if policy_name == "dvn":
                assert agent is not None
                action = dvn_action(agent, obs)
            elif policy_name == "greedy":
                action = greedy_action(obs)
            elif policy_name == "random":
                action = random_action(obs, rng)
            else:
                raise ValueError(f"Unknown policy: {policy_name}")

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step_rewards.append(float(reward))

            if terminated or truncated:
                break

        episode_returns.append(float(total_reward))
        episode_lengths.append(step + 1)

    return {
        "episode_returns": np.array(episode_returns, dtype=np.float32),
        "episode_lengths": np.array(episode_lengths, dtype=np.int32),
        "step_rewards": np.array(step_rewards, dtype=np.float32),
    }


def print_summary(name: str, metrics: dict[str, np.ndarray]) -> None:
    returns = metrics["episode_returns"]
    lengths = metrics["episode_lengths"]
    step_rewards = metrics["step_rewards"]

    print(f"\n{name.upper()}")
    print(f"Episode reward: mean={returns.mean():.2f} std={returns.std():.2f} median={np.median(returns):.2f}")
    print(f"Episode length: mean={lengths.mean():.2f} std={lengths.std():.2f} median={np.median(lengths):.2f}")
    print(
        f"Step reward: mean={step_rewards.mean():.2f} std={step_rewards.std():.2f} median={np.median(step_rewards):.2f}"
    )


def save_distribution_plots(
    dvn_metrics: dict[str, np.ndarray],
    greedy_metrics: dict[str, np.ndarray],
    random_metrics: dict[str, np.ndarray],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = output_dir / f"benchmark_dvn_vs_greedy_vs_random_{timestamp}.png"

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(dvn_metrics["episode_returns"], bins=30, alpha=0.55, label="DVN")
    plt.hist(greedy_metrics["episode_returns"], bins=30, alpha=0.55, label="Greedy")
    plt.hist(random_metrics["episode_returns"], bins=30, alpha=0.55, label="Random")
    plt.title("Episode Reward Distribution")
    plt.xlabel("Episode Reward")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(dvn_metrics["episode_lengths"], bins=30, alpha=0.55, label="DVN")
    plt.hist(greedy_metrics["episode_lengths"], bins=30, alpha=0.55, label="Greedy")
    plt.hist(random_metrics["episode_lengths"], bins=30, alpha=0.55, label="Random")
    plt.title("Episode Length Distribution")
    plt.xlabel("Episode Length")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    env_dvn = BlockBlastEnv(punish_for_invalid=-100)
    env_greedy = BlockBlastEnv(punish_for_invalid=-100)
    env_random = BlockBlastEnv(punish_for_invalid=-100)

    agent = load_dvn_agent(checkpoint_path=checkpoint_path, device=args.device)

    dvn_metrics = run_policy(
        env=env_dvn,
        policy_name="dvn",
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        agent=agent,
    )
    greedy_metrics = run_policy(
        env=env_greedy,
        policy_name="greedy",
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    random_metrics = run_policy(
        env=env_random,
        policy_name="random",
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    print_summary("dvn", dvn_metrics)
    print_summary("greedy", greedy_metrics)
    print_summary("random", random_metrics)

    fig_path = save_distribution_plots(
        dvn_metrics=dvn_metrics,
        greedy_metrics=greedy_metrics,
        random_metrics=random_metrics,
        output_dir=Path(args.output_dir),
    )

    print(f"\nSaved comparison plot to: {fig_path}")


if __name__ == "__main__":
    main()
