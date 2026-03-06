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
            "action_size": agent.action_size
        }
    )

    wandb.watch(agent.policy_net, log="all", log_freq=10)
    epsilon = eps_start

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
            
            if done:
                break

        epsilon = max(eps_end, epsilon * eps_decay)
        
        if episode % target_update_freq == 0:
            agent.update_target_model()
            
        avg_loss = np.mean(episode_losses) if len(episode_losses) > 0 else 0.0
        
        wandb.log({
            "Episode": episode,
            "Return (Score)": episode_return,
            "Episode Length (Steps)": step + 1, # type: ignore
            "Exploration Rate (Epsilon)": epsilon,
            "Average TD Loss": avg_loss
        })
        
        if episode % checkpoint_freq == 0:
            path = f"checkpoints/dvn_ep_{episode}.pt"
            agent.save_model(path)
            
    path = f"checkpoints/dvn_final.pt"
    agent.save_model(path)
    wandb.finish()

def main():
    run_name = f"DVN_1P_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    env = BlockBlastEnv()
    agent = DVNAgent1P()
    train_agent(env, agent, 
                num_episodes=1000,
                max_steps_per_episode=100,
                eps_start=1.0,
                eps_end=0.01,
                eps_decay=0.995,
                target_update_freq=10,
                checkpoint_freq=100,
                project_name="blockblast-rl",
                run_name=run_name)
    
if __name__ == "__main__":
    main()