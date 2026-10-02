"""Train the DVN on BlockBlast 1P (report, Section 4.3).

    python -m dvn.train                      # defaults = configuration of the final run
    python -m dvn.train --episodes 200 --wandb
    python -m dvn.train --resume checkpoints/dvn/dvn_ep_500.pt

Checkpoints (model + full training state, for exact resumption) and a CSV log
go to --output-dir. --wandb additionally logs to Weights & Biases.
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import random
from typing import Any, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from blockblast.block_blast_env import BlockBlastEnv
from common.utils import default_device, set_seed
from dvn.agent import DVNAgent1P
from dvn.models import BlockBlastValueNet1PmultikernelFlattenned


def _torch_load_compat(path: str, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _save_training_state(
    state_path: str,
    *,
    episode: int,
    epsilon: float,
    iteration: int,
    agent: DVNAgent1P,
) -> None:
    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "episode": episode,
        "epsilon": epsilon,
        "iteration": iteration,
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "scheduler_state_dict": agent.scheduler.state_dict() if hasattr(agent, "scheduler") else None,
        "memory": list(agent.memory) if hasattr(agent, "memory") else None,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, state_path)


def _load_training_state(
    state_path: str,
    *,
    agent: DVNAgent1P,
) -> Tuple[int, float, int]:
    state = _torch_load_compat(state_path, map_location=agent.device)

    if "optimizer_state_dict" in state:
        agent.optimizer.load_state_dict(state["optimizer_state_dict"])

    if state.get("scheduler_state_dict") is not None and hasattr(agent, "scheduler"):
        agent.scheduler.load_state_dict(state["scheduler_state_dict"])

    if state.get("memory") is not None and hasattr(agent, "memory"):
        agent.memory.clear()
        agent.memory.extend(state["memory"])

    if state.get("python_random_state") is not None:
        random.setstate(state["python_random_state"])
    if state.get("numpy_random_state") is not None:
        np.random.set_state(state["numpy_random_state"])
    if state.get("torch_random_state") is not None:
        torch.set_rng_state(state["torch_random_state"])
    if torch.cuda.is_available() and state.get("torch_cuda_random_state_all") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_random_state_all"])

    episode = int(state.get("episode", 0))
    epsilon = float(state.get("epsilon", 1.0))
    iteration = int(state.get("iteration", 0))

    return episode + 1, epsilon, iteration

def train_agent(env: BlockBlastEnv, agent: DVNAgent1P,
                num_episodes: int,
                max_steps_per_episode: int,
                eps_start: float,
                eps_end: float,
                eps_decay: float,
                target_update_freq: int,
                checkpoint_freq: int,
                model_update_freq: int = 1,
                resume_model_path: Optional[str] = None,
                resume_state_path: Optional[str] = None,
                checkpoints_dir: str = "checkpoints/dvn",
                use_wandb: bool = False,
                project_name="blockblast-rl", run_name=None):
    config = {
            "num_episodes": num_episodes,
            "eps_start": eps_start,
            "eps_end": eps_end,
            "eps_decay": eps_decay,
            "gamma": agent.gamma,
            "batch_size": agent.batch_size,
            "target_update_freq": target_update_freq,
            "action_size": agent.action_size,
            "buffer_size": agent.memory.maxlen,
            "initial_learning_rate": agent.optimizer.param_groups[0]['lr'],
            "reward_for_survival": env.reward_for_survival,
            "punish_for_invalid": env.punish_for_invalid,
            "base_points": env.base_points
    }
    if use_wandb:
        import wandb
        wandb.init(project=project_name, name=run_name, config=config)
        wandb.watch(agent.policy_net, log="all", log_freq=10)

    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(checkpoints_dir / "log.csv", "a", newline="")
    log_writer = None

    epsilon = eps_start
    iteration = 0
    start_episode = 1

    if resume_model_path is not None:
        agent.load_model(resume_model_path)

    if resume_state_path is not None:
        start_episode, epsilon, iteration = _load_training_state(
            resume_state_path,
            agent=agent,
        )
        print(f"[Resume] start_episode={start_episode}, epsilon={epsilon:.6f}, iteration={iteration}")

    for episode in tqdm(range(start_episode, num_episodes + 1), desc="Training"):
        obs, _ = env.reset()
        
        episode_return = 0.0 
        episode_losses = []
        
        for step in range(max_steps_per_episode):
            action = agent.select_action(obs, epsilon)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(obs, action, reward, next_obs, done)

            episode_return += reward
            obs = next_obs

            if iteration % model_update_freq == 0:
            
                loss = agent.update_model()
                if loss is not None:
                    episode_losses.append(loss)

            if iteration % target_update_freq == 0:
                agent.update_target_model()
            iteration += 1
            if done:
                break

        epsilon = max(eps_end, epsilon * eps_decay)
        
        avg_loss = np.mean(episode_losses) if len(episode_losses) > 0 else 0.0
        
        metrics = {
            "Episode": episode,
            "Return (Score)": episode_return,
            "Episode Length (Steps)": step + 1, # type: ignore
            "Exploration Rate (Epsilon)": epsilon,
            "Average TD Loss": avg_loss,
            "Mean Learning Rate": np.mean([param_group['lr'] for param_group in agent.optimizer.param_groups]),
            "Buffer size" : len(agent.memory)
        }
        if log_writer is None:
            log_writer = csv.DictWriter(log_file, fieldnames=list(metrics))
            if log_file.tell() == 0:
                log_writer.writeheader()
        log_writer.writerow(metrics)
        if use_wandb:
            wandb.log(metrics)
        
        if episode % checkpoint_freq == 0:
            model_path = checkpoints_dir / f"dvn_ep_{episode}.pt"
            state_path = checkpoints_dir / f"dvn_ep_{episode}_state.pt"
            agent.save_model(str(model_path))
            _save_training_state(
                str(state_path),
                episode=episode,
                epsilon=epsilon,
                iteration=iteration,
                agent=agent,
            )
            
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_model_path = checkpoints_dir / f"dvn_final_{timestamp}.pt"
    final_state_path = checkpoints_dir / f"dvn_final_{timestamp}_state.pt"
    agent.save_model(str(final_model_path))
    _save_training_state(
        str(final_state_path),
        episode=num_episodes,
        epsilon=epsilon,
        iteration=iteration,
        agent=agent,
    )
    log_file.close()
    if use_wandb:
        wandb.finish()
    print(f"Final model: {final_model_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps per episode.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.01)
    parser.add_argument("--eps-decay", type=float, default=0.999, help="Multiplicative decay per episode.")
    parser.add_argument("--target-update-freq", type=int, default=800, help="In environment steps.")
    parser.add_argument("--model-update-freq", type=int, default=4, help="One gradient step every N environment steps.")
    parser.add_argument("--checkpoint-freq", type=int, default=500, help="In episodes.")
    parser.add_argument("--punish-for-invalid", type=float, default=-100.0)
    parser.add_argument("--resume", default=None, help="Model checkpoint dvn_ep_<N>.pt; its _state.pt file is loaded too.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--output-dir", default="checkpoints/dvn")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    run_name = f"DVN_1P_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    env = BlockBlastEnv(
        reward_for_survival=5.0,
        punish_for_invalid=args.punish_for_invalid,
        base_points=10.0,
    )
    env.reset(seed=args.seed)  # seeds the environment RNG once for the whole run
    agent = DVNAgent1P(
        policy_net=BlockBlastValueNet1PmultikernelFlattenned,
        lr=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        punish_for_invalid=args.punish_for_invalid,
        device=torch.device(args.device),
    )

    resume_state = None
    if args.resume is not None:
        resume_state = str(Path(args.resume).with_name(Path(args.resume).stem + "_state.pt"))

    train_agent(env, agent,
                num_episodes=args.episodes,
                max_steps_per_episode=args.max_steps,
                eps_start=args.eps_start,
                eps_end=args.eps_end,
                eps_decay=args.eps_decay,
                target_update_freq=args.target_update_freq,
                checkpoint_freq=args.checkpoint_freq,
                model_update_freq=args.model_update_freq,
                resume_model_path=args.resume,
                resume_state_path=resume_state,
                checkpoints_dir=args.output_dir,
                use_wandb=args.wandb,
                project_name="blockblast-rl",
                run_name=run_name)


if __name__ == "__main__":
    main()
