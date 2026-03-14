"""
bc_trainer.py
=============
Fine-tune the PPO policy head via Behavioral Cloning (BC) on MCTS trajectories.

BC loss = cross-entropy(policy_logits, mcts_actions)
         masked to valid actions only

After BC, optionally continues with a short PPO fine-tuning run to recover
value network accuracy and entropy.

Usage (notebook):
    from mct.bc_trainer import bc_finetune

    bc_finetune(
        trainer       = trainer,
        dataset_path  = "/Data/roman.lendormy/rl_checkpoints/mcts_dataset.npz",
        save_path     = "/Data/roman.lendormy/rl_checkpoints/ppo_bc.pt",
        device        = torch.device("cuda"),
    )
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MCTSDataset(Dataset):
    """PyTorch dataset wrapping the collected MCTS trajectories."""

    def __init__(self, path: str):
        data = np.load(path)
        self.boards      = torch.as_tensor(data["boards"],      dtype=torch.float32)
        self.pieces      = torch.as_tensor(data["pieces"],      dtype=torch.float32)
        self.pieces_used = torch.as_tensor(data["pieces_used"], dtype=torch.float32)
        self.combos      = torch.as_tensor(data["combos"],      dtype=torch.float32)
        self.masks       = torch.as_tensor(data["valid_masks"], dtype=torch.bool)
        self.actions     = torch.as_tensor(data["actions"],     dtype=torch.long)

        print(
            f"Dataset loaded: {len(self)} steps | "
            f"action range [{self.actions.min()}, {self.actions.max()}]"
        )

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return {
            "board":       self.boards[idx],
            "pieces":      self.pieces[idx],
            "pieces_used": self.pieces_used[idx],
            "combo":       self.combos[idx],
            "mask":        self.masks[idx],
            "action":      self.actions[idx],
        }


# ---------------------------------------------------------------------------
# BC fine-tuning
# ---------------------------------------------------------------------------

def bc_finetune(
    trainer,
    dataset_path: str,
    save_path: str,
    device          = torch.device("cuda"),
    n_epochs: int   = 10,
    batch_size: int = 512,
    lr: float       = 1e-4,
    freeze_value: bool = True,
    plot_path: str  = None,
):
    """
    Fine-tune the PPO policy head on MCTS trajectories via cross-entropy.

    Parameters
    ----------
    trainer       : PPOTrainer  (modified in-place)
    dataset_path  : path to .npz file from mcts_collect
    save_path     : where to save the fine-tuned checkpoint
    device        : torch device
    n_epochs      : number of passes over the dataset
    batch_size    : mini-batch size
    lr            : learning rate for BC (lower than PPO lr)
    freeze_value  : if True, freeze value head during BC
                    (preserves value estimates, only updates policy)
    plot_path     : if set, save a loss curve plot here
    """
    model = trainer.model.to(device)
    model.train()

    # Optionally freeze value head to preserve V(s) estimates
    if freeze_value:
        for p in model.value_head.parameters():
            p.requires_grad = False
        print("Value head frozen — only policy head updated during BC")

    # Separate optimizer for BC (don't touch PPO optimizer state)
    bc_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, eps=1e-5
    )

    dataset    = MCTSDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    history = defaultdict(list)
    n_steps = 0

    print(f"\nBC fine-tuning: {n_epochs} epochs × {len(dataset):,} steps "
          f"= {n_epochs * len(dataset):,} gradient steps\n")

    for epoch in range(1, n_epochs + 1):
        epoch_losses, epoch_accs = [], []

        for batch in dataloader:
            obs_b = {
                "board":       batch["board"].to(device),
                "pieces":      batch["pieces"].to(device),
                "pieces_used": batch["pieces_used"].to(device),
                "combo":       batch["combo"].to(device),
            }
            masks_b   = batch["mask"].to(device)
            actions_b = batch["action"].to(device)

            logits, _ = model.forward(obs_b, action_mask=masks_b)

            # Cross-entropy loss against MCTS actions
            loss = F.cross_entropy(logits, actions_b)

            bc_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            bc_optimizer.step()

            # Top-1 accuracy: does the policy agree with MCTS?
            acc = (logits.argmax(-1) == actions_b).float().mean().item()

            epoch_losses.append(loss.item())
            epoch_accs.append(acc)
            n_steps += 1

        mean_loss = np.mean(epoch_losses)
        mean_acc  = np.mean(epoch_accs)
        history["loss"].append(mean_loss)
        history["accuracy"].append(mean_acc)

        print(
            f"  Epoch {epoch:>3}/{n_epochs} | "
            f"loss {mean_loss:.4f} | "
            f"accuracy {mean_acc*100:.1f}%"
        )

    # Unfreeze value head before saving
    if freeze_value:
        for p in model.value_head.parameters():
            p.requires_grad = True

    # Save checkpoint (compatible with trainer.load)
    trainer.save(save_path)
    print(f"\nBC checkpoint saved -> {save_path}")

    # Plot loss curve
    if plot_path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(history["loss"], marker='o')
        ax1.set_title("BC Loss (cross-entropy)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True)

        ax2.plot([a * 100 for a in history["accuracy"]], marker='o', color='green')
        ax2.set_title("Policy Agreement with MCTS (%)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Top-1 Accuracy (%)")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"BC curves saved -> {plot_path}")

    return history
