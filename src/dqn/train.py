import argparse
import time
import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from blockblast.block_blast_env import BlockBlastEnv
from dqn import RainbowAgent1P

def parse_args():
    parser = argparse.ArgumentParser(description="Train Rainbow Agent on 1-Piece BlockBlastEnv")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--target_update_freq", type=int, default=10, help="Target update frequency")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.9995, help="Epsilon decay rate")
    parser.add_argument("--training_name", type=str, default="rainbow_v0", help="Name of the training run (checkpoint folder name)")
    parser.add_argument("--save_freq", type=int, default=None, help="Save frequency (defaults to num_episodes // 3)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = BlockBlastEnv(render_mode=None)
    agent = RainbowAgent1P(action_size=64, device=device)
    
    num_params = len(torch.nn.utils.parameters_to_vector(agent.policy_net.parameters()))
    print(f"Number of policy_net parameters: {num_params}")

    project_root = Path(__file__).resolve().parent.parent.parent
    checkpoint_dir = project_root / "checkpoints" / args.training_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_freq = args.save_freq if args.save_freq is not None else max(1, args.num_episodes // 3)

    rewards_history = []
    epsilon = args.epsilon

    print(f"Starting training for {args.num_episodes} episodes...")
    for episode in tqdm(range(args.num_episodes)):
        state, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, epsilon)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update_model()
            state = next_state
            episode_reward += reward
            
        if episode % args.target_update_freq == 0:
            agent.update_target_model()

        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
        rewards_history.append(episode_reward)

        if episode > 0 and episode % save_freq == 0:
            agent.save_model(checkpoint_dir / f"{args.training_name}_{episode}.pth")

    agent.save_model(checkpoint_dir / f"{args.training_name}_final.pth")
    print("Training finished. Final model saved.")

    history = [range(len(rewards_history)), rewards_history]
    result_df = pd.DataFrame(
        np.array(history).T,
        columns=["num_episodes", "mean_final_episode_reward"],
    )
    result_df["agent"] = "Rainbow v0"
    
    csv_path = checkpoint_dir / f"{args.training_name}_results.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"Rewards history saved to {csv_path}")

if __name__ == "__main__":
    main()
