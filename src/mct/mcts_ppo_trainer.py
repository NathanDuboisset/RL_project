"""
mcts_ppo_trainer.py
===================
PPO trainer where MCTS collects the rollouts instead of the greedy policy.

This is the AlphaZero-style training loop:
  1. MCTS plays each step  ->  higher quality trajectories
  2. Standard PPO update on those trajectories
  3. Value network recalibrates continuously on real GAE returns
  4. Better value network -> better MCTS -> better trajectories -> loop

Key difference from offline fine-tuning
-----------------------------------------
- old_log_probs come from the CURRENT model at collection time
  -> ratio new/old stays near 1.0, clip fraction stays low
- Value network updates on fresh GAE returns each rollout
  -> no distribution shift, no value network drift

Cost
----
MCTS takes ~810ms/round. With n_steps=128 and n_envs=4:
  128 steps / ~35 steps per episode = ~3.6 episodes per env per rollout
  ~3.6 * 810ms * 4 envs = ~12 minutes per PPO update
  Recommended: run overnight for 50-100 updates (~10-20M MCTS steps)

Usage (notebook)
----------------
    from mct.mcts_ppo_trainer import MCTSPPOTrainer
    from blockblast.block_blast_3p_env import BlockBlast3PEnv
    import torch

    envs = [BlockBlast3PEnv() for _ in range(4)]

    trainer = MCTSPPOTrainer(
        envs   = envs,
        device = torch.device("cuda"),
        lr     = 1e-4,
        n_steps     = 128,
        n_epochs    = 4,
        batch_size  = 256,
        ent_coef    = 0.01,
        vf_coef     = 0.25,
        gae_lambda  = 0.98,
    )
    trainer.load("/Data/roman.lendormy/rl_checkpoints_2/ckpt_42516k.pt")
    trainer.model = trainer.model.to(torch.device("cuda"))

    trainer.train(
        total_timesteps  = 5_000_000,
        log_interval     = 1,        # log every update (they're slow)
        checkpoint_every = 500_000,
        checkpoint_dir   = "/Data/roman.lendormy/rl_checkpoints_2/mcts_ppo",
    )
"""

import time
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from mct.ppo_agent import (
    PPOTrainer, stack_obs, obs_to_tensors, valid_to_mask, symlog
)
from mct.mcts_agent import MCTSAgent


class MCTSPPOTrainer(PPOTrainer):
    """
    Subclass of PPOTrainer that replaces _collect_rollout with MCTS-guided
    rollout collection. Everything else (PPO update, save/load, history) is
    inherited unchanged.
    """

    def __init__(self, *args, mcts_gamma: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        # MCTS agent is built lazily on first rollout (model must be on device)
        self._mcts_agent = None
        self.mcts_gamma  = mcts_gamma

    def _get_mcts_agent(self) -> MCTSAgent:
        """Lazy init — ensures model is on the right device before building."""
        if self._mcts_agent is None:
            self._mcts_agent = MCTSAgent(
                model      = self.model,
                device     = self.device,
                gamma      = self.mcts_gamma,
                batch_size = 512,
                verbose    = False,
            )
        return self._mcts_agent

    # -------------------------------------------------------------------------
    # MCTS rollout — replaces PPOTrainer._collect_rollout
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _collect_rollout(self):
        """
        Collect n_steps per env using MCTS at round boundaries and greedy
        fallback mid-round. Store (obs, action, log_prob, value, reward, done)
        in the buffer exactly as the standard PPO rollout does.
        """
        self.model.eval()
        agent = self._get_mcts_agent()

        for step in range(self.n_steps):
            raw_rewards = np.zeros(self.n_envs, dtype=np.float32)
            dones       = np.zeros(self.n_envs, dtype=np.float32)
            actions_np  = np.zeros(self.n_envs, dtype=np.int64)

            # --- Choose actions ---
            # MCTS at round start, greedy mid-round
            for i, env in enumerate(self.envs):
                obs_i = self._obs[i]
                if not np.any(obs_i["pieces_used"]):
                    # Round boundary: full MCTS search
                    actions_np[i] = agent.select_action(env)
                else:
                    # Mid-round: greedy single step (fast)
                    actions_np[i] = agent._greedy_fallback(env)

            # --- Get log_probs and values from current model ---
            # We need these for the PPO ratio and GAE computation.
            # Use the model's own evaluation of the chosen actions.
            batch  = stack_obs(self._obs)
            obs_t  = obs_to_tensors(batch, self.device)
            mask_np = valid_to_mask(batch["valid_placements"])
            mask_t  = torch.as_tensor(mask_np, device=self.device)

            actions_t = torch.as_tensor(actions_np, dtype=torch.long, device=self.device)

            logits, values = self.model.forward(obs_t, mask_t)
            from torch.distributions import Categorical
            dist      = Categorical(logits=logits)
            log_probs = dist.log_prob(actions_t)

            # --- Step environments ---
            for i, env in enumerate(self.envs):
                next_obs, rew, term, trunc, _ = env.step(int(actions_np[i]))
                done = term or trunc
                raw_rewards[i] = rew
                dones[i]       = float(done)

                self._ep_ret[i] += rew
                self._ep_len[i] += 1

                if done:
                    self.ep_returns.append(self._ep_ret[i])
                    self.ep_lengths.append(self._ep_len[i])
                    self._ep_ret[i] = 0.0
                    self._ep_len[i] = 0
                    next_obs, _ = env.reset()

                self._obs[i] = next_obs

            # symlog transform before storing (matches training)
            transformed_rewards = torch.as_tensor(
                symlog(raw_rewards), dtype=torch.float32, device=self.device
            )

            self.buffer.add(
                step, obs_t, mask_t, actions_t, log_probs,
                transformed_rewards,
                values,
                torch.as_tensor(dones, device=self.device),
            )
            self.total_steps += self.n_envs

        # Bootstrap value on last obs
        batch   = stack_obs(self._obs)
        obs_t   = obs_to_tensors(batch, self.device)
        mask_np = valid_to_mask(batch["valid_placements"])
        mask_t  = torch.as_tensor(mask_np, device=self.device)
        _, last_values = self.model.forward(obs_t, mask_t)
        self.buffer.compute_returns_and_advantages(last_values, self.gamma, self.gae_lambda)

    # -------------------------------------------------------------------------
    # Training loop with checkpointing
    # -------------------------------------------------------------------------

    def train(
        self,
        total_timesteps:  int,
        log_interval:     int = 1,
        checkpoint_every: int = 500_000,
        checkpoint_dir:   str = "mcts_ppo_ckpts",
    ) -> dict:
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._reset_all_envs()

        n_updates = total_timesteps // (self.n_steps * self.n_envs)

        hdr = (
            f"{'Upd':>5} | {'Steps':>10} | {'MeanRet':>9} | {'MedRet':>9} | "
            f"{'MeanLen':>8} | {'PgLoss':>8} | {'VfLoss':>8} | "
            f"{'Entropy':>8} | {'ClipFr':>7} | {'Time':>7}"
        )
        print(f"\nMCTS-PPO — {total_timesteps:,} steps · {n_updates} updates · "
              f"{self.n_envs} envs\n")
        print(hdr)
        print("─" * len(hdr))

        last_ckpt = self.total_steps

        for upd in range(1, n_updates + 1):
            t0 = time.time()
            self._collect_rollout()
            m  = self._ppo_update()
            elapsed = time.time() - t0

            if upd % log_interval == 0 or upd == 1:
                mr  = float(np.mean(self.ep_returns))  if self.ep_returns else 0.0
                mdr = float(np.median(self.ep_returns)) if self.ep_returns else 0.0
                ml  = float(np.mean(self.ep_lengths))  if self.ep_lengths else 0.0

                self.history["steps"].append(self.total_steps)
                self.history["mean_return"].append(mr)
                self.history["median_return"].append(mdr)
                self.history["mean_length"].append(ml)
                self.history["loss_policy"].append(m["loss_policy"])
                self.history["loss_value"].append(m["loss_value"])
                self.history["entropy"].append(m["entropy"])
                self.history["clip_frac"].append(m["clip_frac"])

                print(
                    f"{upd:>5} | {self.total_steps:>10,} | {mr:>9.2f} | {mdr:>9.2f} | "
                    f"{ml:>8.1f} | {m['loss_policy']:>8.4f} | {m['loss_value']:>8.4f} | "
                    f"{m['entropy']:>8.4f} | {m['clip_frac']:>7.3f} | {elapsed:>6.0f}s"
                )

            # Checkpoint
            if self.total_steps - last_ckpt >= checkpoint_every:
                tag  = f"{self.total_steps // 1000}k"
                path = os.path.join(checkpoint_dir, f"ckpt_{tag}.pt")
                self.save(path)
                self._plot_curves(checkpoint_dir, tag)
                last_ckpt = self.total_steps

        # Final save
        final_path = os.path.join(checkpoint_dir, "ckpt_final.pt")
        self.save(final_path)
        self._plot_curves(checkpoint_dir, "final")
        return self.history

    def _plot_curves(self, directory: str, tag: str):
        h = self.history
        if not h["steps"]:
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"MCTS-PPO -- BlockBlast3P @ {tag} steps")

        pairs = [
            (axes[0, 0], "mean_return",   "Mean Episode Return"),
            (axes[0, 1], "median_return", "Median Episode Return"),
            (axes[0, 2], "mean_length",   "Mean Episode Length"),
            (axes[1, 0], "loss_policy",   "Policy Loss"),
            (axes[1, 1], "loss_value",    "Value Loss"),
            (axes[1, 2], "entropy",       "Entropy"),
        ]
        for ax, key, title in pairs:
            ax.plot(h["steps"], h[key], alpha=0.3)
            if len(h[key]) >= 5:
                import pandas as pd
                smoothed = pd.Series(h[key]).rolling(5, min_periods=1).mean()
                ax.plot(h["steps"], smoothed, linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Environment steps")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(directory, f"ckpt_{tag}_curves.png")
        plt.savefig(path, dpi=120)
        plt.close()
