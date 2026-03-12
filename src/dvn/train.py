import sys
from pathlib import Path
import random
from typing import Any, Optional, Tuple

import wandb
import numpy as np
import torch
from tqdm import tqdm

# Allow running this file directly without requiring editable installation.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.dvn.agent import DVNAgent1P
from src.blockblast.block_blast_env import BlockBlastEnv
from datetime import datetime
from src.dvn.models import *


def _torch_load_compat(path: str, map_location: torch.device) -> dict[str, Any]:
    """Load torch checkpoints across torch versions."""
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
    """Save non-model training state for exact resume."""
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
    """Restore non-model training state and return resume cursor."""
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
                num_episodes:int, 
                max_steps_per_episode:int, 
                eps_start:float, 
                eps_end:float, 
                eps_decay:float, 
                target_update_freq:int, 
                checkpoint_freq:int, 
                model_update_freq:int = 1,
                resume_model_path: Optional[str] = None,
                resume_state_path: Optional[str] = None,
                project_name="blockblast-rl", run_name=None):
    """
    Boucle d'entraînement formelle pour un agent RL (TD-Learning / Q-Learning).
    
    Args:
        env: L'environnement Gymnasium (ex: BlockBlastEnv).
        agent: L'agent implémentant select_action, store_transition, update_model, etc.
        num_episodes (int): Nombre total d'épisodes (trajectoires) à simuler.
        max_steps_per_episode (int): Sécurité pour éviter les boucles infinies.
        eps_start (float): Probabilité d'exploration initiale eps.
        eps_end (float): Borne inférieure de l'exploration.
        eps_decay (float): Facteur de décroissance géométrique de eps.
        target_update_freq (int): Fréquence (en épisodes) de synchronisation du réseau cible.
        checkpoint_freq (int): Fréquence (en épisodes) de sauvegarde des poids theta.
        resume_model_path (Optional[str]): Chemin d'un checkpoint des poids à recharger.
        resume_state_path (Optional[str]): Chemin d'un checkpoint d'état d'entraînement à recharger.
    """
    
    wandb.init(
        project=project_name,
        name=run_name,
        config={
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
    )

    wandb.watch(agent.policy_net, log="all", log_freq=10)

    checkpoints_dir = Path("/Data/KAT/checkpoints/")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

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

    for episode in tqdm(range(start_episode, num_episodes + 1), desc="Entraînement"):
        obs, _ = env.reset()
        
        episode_return = 0.0 
        episode_losses = []
        
        for step in range(max_steps_per_episode):
            action = agent.select_action(obs, epsilon)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(obs, action, reward, next_obs, done)

            if iteration % model_update_freq == 0:
            
                loss = agent.update_model()
                if loss is not None:
                    episode_losses.append(loss)
                    
                episode_return += reward
                obs = next_obs

                if iteration % target_update_freq == 0:
                    agent.update_target_model()
                iteration += 1
                if done:
                    break

        epsilon = max(eps_end, epsilon * eps_decay)
        
        avg_loss = np.mean(episode_losses) if len(episode_losses) > 0 else 0.0
        
        wandb.log({
            "Episode": episode,
            "Return (Score)": episode_return,
            "Episode Length (Steps)": step + 1, # type: ignore
            "Exploration Rate (Epsilon)": epsilon,
            "Average TD Loss": avg_loss,
            "Mean Learning Rate": np.mean([param_group['lr'] for param_group in agent.optimizer.param_groups]),
            "Buffer size" : len(agent.memory)
        })
        
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
    wandb.finish()

def main():
    run_name = f"DVN_1P_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    env = BlockBlastEnv(
        reward_for_survival= 5.0,
        punish_for_invalid= -100.0,
        base_points= 10.0
    )
    agent = DVNAgent1P(
        policy_net=BlockBlastValueNet1PmultikernelFlattenned,
        lr = 1e-4,
        buffer_size=100_000,
        batch_size=512,
        punish_for_invalid=-100.0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # To resume, set both paths to the same episode index.
    resume_model_path = None
    resume_state_path = None

    train_agent(env, agent,
                num_episodes=10_000,
                max_steps_per_episode=100,
                eps_start=1.0,
                eps_end=0.01,
                eps_decay=0.999,
                target_update_freq=200,
                checkpoint_freq=500,
                model_update_freq=4,
                resume_model_path=resume_model_path,
                resume_state_path=resume_state_path,
                project_name="blockblast-rl",
                run_name=run_name)
    
if __name__ == "__main__":
    main()
