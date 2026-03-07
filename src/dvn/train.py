import wandb
import numpy as np
import torch
from tqdm import tqdm
from src.dvn.agent import DVNAgent1P
from src.blockblast.block_blast_env import BlockBlastEnv
from datetime import datetime

def train_agent(env: BlockBlastEnv, agent: DVNAgent1P, 
                num_episodes:int, 
                max_steps_per_episode:int, 
                eps_start:float, 
                eps_end:float, 
                eps_decay:float, 
                target_update_freq:int, 
                checkpoint_freq:int, 
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
    epsilon = eps_start
    iteration = 0

    for episode in tqdm(range(1, num_episodes + 1), desc="Entraînement"):
        obs, _ = env.reset()
        
        episode_return = 0.0 
        episode_losses = []
        
        for step in range(max_steps_per_episode):
            action = agent.select_action(obs, epsilon)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(obs, action, reward, next_obs, done)
            
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
            path = f"checkpoints/dvn_ep_{episode}.pt"
            agent.save_model(path)
            
    path = f"checkpoints/dvn_final.pt"
    agent.save_model(path)
    wandb.finish()

def main():
    run_name = f"DVN_1P_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    env = BlockBlastEnv(
        reward_for_survival= 0.05,
        punish_for_invalid=-5.0,
        base_points= 0.1
    )
    agent = DVNAgent1P(
        lr = 1e-4,
        buffer_size=100_000,
        batch_size=128,
        punish_for_invalid=-5.0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    train_agent(env, agent,
                num_episodes=10_000,
                max_steps_per_episode=100,
                eps_start=1.0,
                eps_end=0.01,
                eps_decay=0.999,
                target_update_freq=200,
                checkpoint_freq=500,
                project_name="blockblast-rl",
                run_name=run_name)
    
if __name__ == "__main__":
    main()
