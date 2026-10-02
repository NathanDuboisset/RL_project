import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict


class MCTSDataset(Dataset):
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
    """Fine-tune the policy head on MCTS data via cross-entropy loss."""
    model = trainer.model.to(device)
    model.train()

    if freeze_value:
        for p in model.value_head.parameters():
            p.requires_grad = False
        print("Value head frozen — only policy head updated during BC")

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
            loss = F.cross_entropy(logits, actions_b)

            bc_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            bc_optimizer.step()

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

    if freeze_value:
        for p in model.value_head.parameters():
            p.requires_grad = True

    trainer.save(save_path)
    print(f"\nBC checkpoint saved -> {save_path}")

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
