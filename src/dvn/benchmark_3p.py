import argparse
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blockblast.block_blast_3p_env import BlockBlast3PEnv
from src.dvn.agent import DVNAgent1P,RoundPlanner3P
from src.dvn.models import BlockBlastValueNet1PmultikernelFlattenned
from src.dvn.models import BlockBlastValueNet1P


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark DVN vs greedy vs random on BlockBlast 3P and compare "
            "reward/episode length distributions."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to DVN checkpoint (.pt).",
    )
    parser.add_argument("--episodes", type=int, default=200, help="Number of evaluation episodes.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
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
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor used by the DVN planner.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory where plots are saved.",
    )
    return parser.parse_args()


def load_dvn_agent(checkpoint_path: Path, device: str) -> DVNAgent1P:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
    policy_keys = set(checkpoint["policy_state_dict"].keys())
    if any(k.startswith("branches.") for k in policy_keys):
        model_cls = BlockBlastValueNet1PmultikernelFlattenned
    else:
        model_cls = BlockBlastValueNet1P
    agent = DVNAgent1P(policy_net=model_cls, device=device)
    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.policy_net.eval()
    agent.target_net.eval()
    return agent

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _compute_immediate_reward(obs: dict, env: BlockBlast3PEnv, action: int) -> float:
    """Simulate one placement and return its immediate reward."""
    piece_idx = action // (env.grid_size * env.grid_size)
    pos = action % (env.grid_size * env.grid_size)
    row = pos // env.grid_size
    col = pos % env.grid_size

    piece_grid = env.pieces_grids[piece_idx]
    h, w = piece_grid.shape

    sim_board = obs["board"].copy()
    sim_board[row : row + h, col : col + w] += piece_grid

    n_cleared = int(np.all(sim_board == 1, axis=1).sum() + np.all(sim_board == 1, axis=0).sum())
    combo = int(obs["combo"][0])

    if n_cleared > 0:
        return float(env.base_points * (n_cleared ** 2 + combo))
    return 0.1


def greedy_action(obs: dict, env: BlockBlast3PEnv) -> int | None:
    valid_mask = obs["valid_placements"].reshape(-1).astype(bool)
    valid_actions = np.flatnonzero(valid_mask)
    if valid_actions.size == 0:
        return None

    rewards = np.array([_compute_immediate_reward(obs, env, a) for a in valid_actions])
    return int(valid_actions[int(np.argmax(rewards))])


def random_action(obs: dict, rng: np.random.Generator) -> int | None:
    valid_mask = obs["valid_placements"].reshape(-1).astype(bool)
    valid_actions = np.flatnonzero(valid_mask)
    if valid_actions.size == 0:
        return None
    return int(rng.choice(valid_actions))


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_policy(
    env: BlockBlast3PEnv,
    policy_name: str,
    episodes: int,
    max_steps: int,
    seed: int,
    agent: DVNAgent1P | None = None,
    gamma: float = 0.99,
) -> dict[str, np.ndarray]:
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    step_rewards: list[float] = []
    rng = np.random.default_rng(seed=seed)

    planner: RoundPlanner3P | None = None
    if policy_name == "dvn":
        assert agent is not None
        planner = RoundPlanner3P(gamma=gamma, agent=agent)

    for ep in tqdm(range(episodes), desc=f"Eval {policy_name}"):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0

        if planner is not None:
            planner.reset_round_plan()

        step = 0
        for step in range(max_steps):
            if policy_name == "dvn":
                assert planner is not None
                action = planner.select_action(env)
            elif policy_name == "greedy":
                action = greedy_action(obs, env)
            elif policy_name == "random":
                action = random_action(obs, rng)
            else:
                raise ValueError(f"Unknown policy: {policy_name}")

            if action is None:
                break

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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(name: str, metrics: dict[str, np.ndarray]) -> None:
    returns = metrics["episode_returns"]
    lengths = metrics["episode_lengths"]
    step_rewards = metrics["step_rewards"]

    print(f"\n{name.upper()}")
    print(
        f"Episode reward : mean={returns.mean():.2f}  std={returns.std():.2f}  "
        f"median={np.median(returns):.2f}  max={returns.max():.2f}"
    )
    print(
        f"Episode length : mean={lengths.mean():.2f}  std={lengths.std():.2f}  "
        f"median={np.median(lengths):.2f}"
    )
    print(
        f"Step reward    : mean={step_rewards.mean():.2f}  std={step_rewards.std():.2f}  "
        f"median={np.median(step_rewards):.2f}"
    )


def save_distribution_plots(
    dvn_metrics: dict[str, np.ndarray],
    greedy_metrics: dict[str, np.ndarray],
    random_metrics: dict[str, np.ndarray],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = output_dir / f"benchmark_3p_dvn_vs_greedy_vs_random_{timestamp}.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(dvn_metrics["episode_returns"], bins=30, alpha=0.55, label="DVN-3P")
    ax.hist(greedy_metrics["episode_returns"], bins=30, alpha=0.55, label="Greedy")
    ax.hist(random_metrics["episode_returns"], bins=30, alpha=0.55, label="Random")
    ax.set_title("Episode Reward Distribution (3P)")
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Frequency")
    ax.legend()

    ax = axes[1]
    ax.hist(dvn_metrics["episode_lengths"], bins=30, alpha=0.55, label="DVN-3P")
    ax.hist(greedy_metrics["episode_lengths"], bins=30, alpha=0.55, label="Greedy")
    ax.hist(random_metrics["episode_lengths"], bins=30, alpha=0.55, label="Random")
    ax.set_title("Episode Length Distribution (3P)")
    ax.set_xlabel("Episode Length")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    gamma = args.gamma

    env_dvn = BlockBlast3PEnv(lookahead_gamma=gamma)
    env_greedy = BlockBlast3PEnv(lookahead_gamma=gamma)
    env_random = BlockBlast3PEnv(lookahead_gamma=gamma)

    agent = load_dvn_agent(checkpoint_path=checkpoint_path, device=args.device)

    dvn_metrics = run_policy(
        env=env_dvn,
        policy_name="dvn",
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        agent=agent,
        gamma=gamma,
    )
    greedy_metrics = run_policy(
        env=env_greedy,
        policy_name="greedy",
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        gamma=gamma,
    )
    random_metrics = run_policy(
        env=env_random,
        policy_name="random",
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        gamma=gamma,
    )

    print_summary("dvn-3p", dvn_metrics)
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
