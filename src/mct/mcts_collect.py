"""
mcts_collect.py
===============
Generate a behavioral cloning dataset by running MCTS for N episodes
and recording every (obs, action) pair.

Saved as a .npz file with keys:
  boards        : (N_steps, 8, 8)       float32
  pieces        : (N_steps, 3, 5, 5)    float32
  pieces_used   : (N_steps, 3)          float32
  combos        : (N_steps, 1)          float32
  valid_masks   : (N_steps, 192)        bool
  actions       : (N_steps,)            int64   <- MCTS-chosen actions

Usage (notebook):
    from mct.mcts_collect import collect_mcts_dataset
    collect_mcts_dataset(
        trainer      = trainer,
        env_fn       = lambda: BlockBlast3PEnv(),
        n_episodes   = 1000,
        save_path    = "/Data/roman.lendormy/rl_checkpoints/mcts_dataset.npz",
        device       = torch.device("cuda"),
    )
"""

import time
import numpy as np
import torch
from mct.mcts_agent import MCTSAgent
from mct.ppo_agent import valid_to_mask


def collect_mcts_dataset(
    trainer,
    env_fn,
    n_episodes: int = 1000,
    save_path: str  = "mcts_dataset.npz",
    device          = torch.device("cuda"),
    gamma: float    = 0.99,
):
    """
    Run MCTS for n_episodes and record every (obs, action) pair.

    Parameters
    ----------
    trainer     : PPOTrainer  (needs trainer.model)
    env_fn      : callable()  -> BlockBlast3PEnv instance
    n_episodes  : number of MCTS episodes to run
    save_path   : where to save the .npz dataset
    device      : torch device for value network inference
    gamma       : discount factor (should match training)
    """
    agent = MCTSAgent(
        model      = trainer.model,
        device     = device,
        gamma      = gamma,
        batch_size = 512,
        verbose    = False,
    )

    # Accumulators
    all_boards      = []
    all_pieces      = []
    all_pieces_used = []
    all_combos      = []
    all_masks       = []
    all_actions     = []
    all_rewards     = []
    all_dones       = []

    ep_returns = []
    ep_lengths = []

    t_start = time.time()

    for ep in range(n_episodes):
        env = env_fn()
        obs, _ = env.reset()

        ep_ret = 0.0
        ep_len = 0

        while True:
            # Record obs BEFORE action
            all_boards.append(obs["board"].astype(np.float32))
            all_pieces.append(obs["pieces"].astype(np.float32))
            all_pieces_used.append(obs["pieces_used"].astype(np.float32))
            all_combos.append(obs["combo"].astype(np.float32))

            mask = valid_to_mask(obs["valid_placements"][None])[0]  # (192,)
            all_masks.append(mask)

            # MCTS or greedy depending on round state
            if not np.any(obs["pieces_used"]):
                action = agent.select_action(env)
            else:
                action = agent._greedy_fallback(env)

            all_actions.append(action)

            obs, reward, term, trunc, _ = env.step(action)
            done = float(term or trunc)
            all_rewards.append(float(reward))
            all_dones.append(done)
            ep_ret += reward
            ep_len += 1

            if term or trunc:
                break

        ep_returns.append(ep_ret)
        ep_lengths.append(ep_len)
        env.close()

        # Progress logging
        if (ep + 1) % 50 == 0:
            elapsed  = time.time() - t_start
            per_ep   = elapsed / (ep + 1)
            remaining = per_ep * (n_episodes - ep - 1)
            print(
                f"  Episode {ep+1:>5}/{n_episodes} | "
                f"return {ep_ret:>8.1f} | "
                f"len {ep_len:>4} | "
                f"mean_ret {np.mean(ep_returns):.1f} | "
                f"ETA {remaining/60:.1f}min"
            )

    # Stack everything
    dataset = {
        "boards":      np.stack(all_boards,      axis=0),
        "pieces":      np.stack(all_pieces,      axis=0),
        "pieces_used": np.stack(all_pieces_used, axis=0),
        "combos":      np.stack(all_combos,      axis=0),
        "valid_masks": np.stack(all_masks,       axis=0),
        "actions":     np.array(all_actions,     dtype=np.int64),
        "rewards":     np.array(all_rewards,     dtype=np.float32),
        "dones":       np.array(all_dones,       dtype=np.float32),
    }

    n_steps = dataset["actions"].shape[0]
    np.savez_compressed(save_path, **dataset)

    elapsed = time.time() - t_start
    print(
        f"\n=== Dataset saved to {save_path} ===\n"
        f"  Episodes  : {n_episodes}\n"
        f"  Steps     : {n_steps:,}\n"
        f"  Mean ret  : {np.mean(ep_returns):.2f} +/- {np.std(ep_returns):.2f}\n"
        f"  Mean len  : {np.mean(ep_lengths):.1f} steps\n"
        f"  Time      : {elapsed/60:.1f} min\n"
    )
    return dataset
