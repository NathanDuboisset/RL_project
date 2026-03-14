"""
ppo_finetune.py
===============
Fine-tune PPO on MCTS-generated trajectories.

Unlike Behavioral Cloning, this uses the full PPO update (clipped PG +
value loss + entropy bonus) on the MCTS dataset. The key difference from
BC is that we compute GAE advantages from the dataset rewards, so the
value network is updated correctly alongside the policy.

Pipeline
--------
1. Load the MCTS dataset (.npz from mcts_collect.py)
2. Reconstruct full episodes and compute GAE returns/advantages
3. Run PPO updates (same loss as training, just on offline data)
4. Evaluate before/after to confirm improvement

Usage (notebook)
----------------
    from mct.ppo_finetune import ppo_finetune_on_mcts

    history = ppo_finetune_on_mcts(
        trainer      = trainer,
        dataset_path = "/Data/roman.lendormy/rl_checkpoints_2/mcts_dataset.npz",
        save_path    = "/Data/roman.lendormy/rl_checkpoints_2/ppo_mcts_ft.pt",
        device       = torch.device("cuda"),
    )
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import defaultdict

from mct.ppo_agent import symlog


# ---------------------------------------------------------------------------
# GAE computation on offline dataset
# ---------------------------------------------------------------------------

def compute_gae(
    rewards:    np.ndarray,
    values:     np.ndarray,
    dones:      np.ndarray,
    gamma:      float = 0.99,
    gae_lambda: float = 0.98,
) -> tuple:
    """
    Compute GAE advantages and returns on a flat sequence of steps.

    rewards, values, dones : (N,) float32
    Returns: advantages (N,), returns (N,)
    """
    N = len(rewards)
    advantages = np.zeros(N, dtype=np.float32)
    last_gae   = 0.0

    for t in reversed(range(N)):
        next_val = values[t + 1] if t < N - 1 else 0.0
        delta    = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Main fine-tuning function
# ---------------------------------------------------------------------------

def ppo_finetune_on_mcts(
    trainer,
    dataset_path:   str,
    save_path:      str,
    device          = torch.device("cuda"),
    n_epochs:       int   = 5,
    batch_size:     int   = 512,
    lr:             float = 5e-5,
    gamma:          float = 0.99,
    gae_lambda:     float = 0.98,
    clip_eps:       float = 0.2,
    vf_coef:        float = 0.25,
    ent_coef:       float = 0.01,
    max_grad_norm:  float = 0.5,
    plot_path:      str   = None,
):
    """
    Fine-tune PPO on MCTS trajectories using the standard PPO loss.

    Parameters
    ----------
    trainer       : PPOTrainer (modified in-place)
    dataset_path  : .npz file from mcts_collect.py
    save_path     : where to save the fine-tuned checkpoint
    device        : torch device
    n_epochs      : passes over the dataset
    batch_size    : mini-batch size
    lr            : learning rate (lower than PPO training lr)
    gamma         : discount factor
    gae_lambda    : GAE lambda
    clip_eps      : PPO clipping epsilon
    vf_coef       : value loss coefficient
    ent_coef      : entropy bonus coefficient
    max_grad_norm : gradient clipping
    plot_path     : if set, save loss curves here
    """
    # --- Load dataset ---
    print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path)

    boards      = data["boards"]       # (N, 8, 8)
    pieces      = data["pieces"]       # (N, 3, 5, 5)
    pieces_used = data["pieces_used"]  # (N, 3)
    combos      = data["combos"]       # (N, 1)
    masks       = data["valid_masks"]  # (N, 192)
    actions     = data["actions"]      # (N,)

    N = len(actions)
    print(f"  {N:,} steps loaded")

    # --- Compute value estimates on the dataset (no grad) ---
    print("Computing value estimates...")
    model = trainer.model.to(device)
    model.eval()

    all_values = np.zeros(N, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end   = min(start + batch_size, N)
            obs_b = {
                "board":       torch.as_tensor(boards[start:end],       dtype=torch.float32, device=device),
                "pieces":      torch.as_tensor(pieces[start:end],       dtype=torch.float32, device=device),
                "pieces_used": torch.as_tensor(pieces_used[start:end],  dtype=torch.float32, device=device),
                "combo":       torch.as_tensor(combos[start:end],       dtype=torch.float32, device=device),
            }
            mask_b = torch.as_tensor(masks[start:end], dtype=torch.bool, device=device)
            _, v   = model.forward(obs_b, action_mask=mask_b)
            all_values[start:end] = v.cpu().numpy()

    # --- Compute old log probs (for PPO ratio) ---
    print("Computing old log probabilities...")
    all_old_logprobs = np.zeros(N, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end   = min(start + batch_size, N)
            obs_b = {
                "board":       torch.as_tensor(boards[start:end],       dtype=torch.float32, device=device),
                "pieces":      torch.as_tensor(pieces[start:end],       dtype=torch.float32, device=device),
                "pieces_used": torch.as_tensor(pieces_used[start:end],  dtype=torch.float32, device=device),
                "combo":       torch.as_tensor(combos[start:end],       dtype=torch.float32, device=device),
            }
            mask_b   = torch.as_tensor(masks[start:end],   dtype=torch.bool,  device=device)
            acts_b   = torch.as_tensor(actions[start:end], dtype=torch.long,  device=device)
            logits, _ = model.forward(obs_b, action_mask=mask_b)
            dist      = Categorical(logits=logits)
            all_old_logprobs[start:end] = dist.log_prob(acts_b).cpu().numpy()

    # --- Load real rewards and dones from dataset ---
    print("Loading real rewards and dones...")
    raw_rewards = data["rewards"]   # (N,) true env rewards
    dones       = data["dones"]     # (N,) episode boundaries

    # Apply symlog so scale matches what the value network was trained on
    transformed_rewards = symlog(raw_rewards)

    # --- Compute GAE advantages using real rewards ---
    print("Computing GAE advantages...")
    advantages, returns = compute_gae(
        transformed_rewards, all_values, dones, gamma, gae_lambda
    )

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # --- PPO fine-tuning loop ---
    ft_optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    history      = defaultdict(list)

    print(f"\nPPO fine-tuning: {n_epochs} epochs × {N:,} steps\n")
    print(f"{'Epoch':>6} | {'PG Loss':>9} | {'VF Loss':>9} | {'Entropy':>9} | {'Clip Fr':>9} | {'Accuracy':>9}")
    print("─" * 65)

    idx = np.arange(N)

    for epoch in range(1, n_epochs + 1):
        model.train()
        np.random.shuffle(idx)

        pg_losses, vf_losses, entropies, clip_fracs, accs = [], [], [], [], []

        for start in range(0, N, batch_size):
            b = idx[start : start + batch_size]

            obs_b = {
                "board":       torch.as_tensor(boards[b],       dtype=torch.float32, device=device),
                "pieces":      torch.as_tensor(pieces[b],       dtype=torch.float32, device=device),
                "pieces_used": torch.as_tensor(pieces_used[b],  dtype=torch.float32, device=device),
                "combo":       torch.as_tensor(combos[b],       dtype=torch.float32, device=device),
            }
            mask_b    = torch.as_tensor(masks[b],              dtype=torch.bool,  device=device)
            acts_b    = torch.as_tensor(actions[b],            dtype=torch.long,  device=device)
            old_lp_b  = torch.as_tensor(all_old_logprobs[b],  dtype=torch.float32, device=device)
            adv_b     = torch.as_tensor(advantages[b],        dtype=torch.float32, device=device)
            ret_b     = torch.as_tensor(returns[b],           dtype=torch.float32, device=device)

            logits, values = model.forward(obs_b, action_mask=mask_b)
            dist    = Categorical(logits=logits)
            new_lp  = dist.log_prob(acts_b)
            entropy = dist.entropy().mean()

            ratio = (new_lp - old_lp_b).exp()
            clip_fracs.append(((ratio - 1).abs() > clip_eps).float().mean().item())

            pg_loss = torch.max(
                -adv_b * ratio,
                -adv_b * ratio.clamp(1 - clip_eps, 1 + clip_eps),
            ).mean()

            vf_loss = F.mse_loss(values, ret_b)
            loss    = pg_loss + vf_coef * vf_loss - ent_coef * entropy

            ft_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            ft_optimizer.step()

            pg_losses.append(pg_loss.item())
            vf_losses.append(vf_loss.item())
            entropies.append(entropy.item())

            # Agreement with MCTS actions
            acc = (logits.argmax(-1) == acts_b).float().mean().item()
            accs.append(acc)

        mean_pg  = np.mean(pg_losses)
        mean_vf  = np.mean(vf_losses)
        mean_ent = np.mean(entropies)
        mean_cf  = np.mean(clip_fracs)
        mean_acc = np.mean(accs)

        history["loss_policy"].append(mean_pg)
        history["loss_value"].append(mean_vf)
        history["entropy"].append(mean_ent)
        history["clip_frac"].append(mean_cf)
        history["accuracy"].append(mean_acc)

        print(
            f"{epoch:>6} | {mean_pg:>9.4f} | {mean_vf:>9.4f} | "
            f"{mean_ent:>9.4f} | {mean_cf:>9.3f} | {mean_acc*100:>8.1f}%"
        )

    # --- Save ---
    trainer.save(save_path)
    print(f"\nFine-tuned checkpoint saved -> {save_path}")

    # --- Plot ---
    if plot_path:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].plot(history["loss_policy"], marker='o', label="PG Loss")
        axes[0].plot(history["loss_value"],  marker='s', label="VF Loss")
        axes[0].set_title("Losses")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history["entropy"], marker='o', color='brown')
        axes[1].set_title("Entropy")
        axes[1].set_xlabel("Epoch")
        axes[1].grid(True)

        axes[2].plot([a * 100 for a in history["accuracy"]], marker='o', color='green')
        axes[2].set_title("Policy Agreement with MCTS (%)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Top-1 Accuracy (%)")
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"Fine-tuning curves saved -> {plot_path}")

    return history
