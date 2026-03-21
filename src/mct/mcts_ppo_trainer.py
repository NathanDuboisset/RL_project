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
    """PPOTrainer that uses MCTS at round boundaries instead of pure greedy rollouts."""

    def __init__(self, *args, mcts_gamma: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self._mcts_agent = None
        self.mcts_gamma  = mcts_gamma

    def _get_mcts_agent(self) -> MCTSAgent:
        if self._mcts_agent is None:
            self._mcts_agent = MCTSAgent(
                model      = self.model,
                device     = self.device,
                gamma      = self.mcts_gamma,
                batch_size = 512,
                verbose    = False,
            )
        return self._mcts_agent

    @torch.no_grad()
    def _collect_rollout(self):
        self.model.eval()
        agent = self._get_mcts_agent()

        for step in range(self.n_steps):
            raw_rewards = np.zeros(self.n_envs, dtype=np.float32)
            dones       = np.zeros(self.n_envs, dtype=np.float32)
            actions_np  = np.zeros(self.n_envs, dtype=np.int64)

            for i, env in enumerate(self.envs):
                obs_i = self._obs[i]
                if not np.any(obs_i["pieces_used"]):
                    actions_np[i] = agent.select_action(env)
                else:
                    actions_np[i] = agent._greedy_fallback(env)

            batch  = stack_obs(self._obs)
            obs_t  = obs_to_tensors(batch, self.device)
            mask_np = valid_to_mask(batch["valid_placements"])
            mask_t  = torch.as_tensor(mask_np, device=self.device)

            actions_t = torch.as_tensor(actions_np, dtype=torch.long, device=self.device)

            logits, values = self.model.forward(obs_t, mask_t)
            from torch.distributions import Categorical
            dist      = Categorical(logits=logits)
            log_probs = dist.log_prob(actions_t)

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

        batch   = stack_obs(self._obs)
        obs_t   = obs_to_tensors(batch, self.device)
        mask_np = valid_to_mask(batch["valid_placements"])
        mask_t  = torch.as_tensor(mask_np, device=self.device)
        _, last_values = self.model.forward(obs_t, mask_t)
        self.buffer.compute_returns_and_advantages(last_values, self.gamma, self.gae_lambda)

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
                mr  = float(np.mean(self.ep_returns))   if self.ep_returns else 0.0
                mdr = float(np.median(self.ep_returns))  if self.ep_returns else 0.0
                ml  = float(np.mean(self.ep_lengths))   if self.ep_lengths else 0.0

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

            if self.total_steps - last_ckpt >= checkpoint_every:
                tag  = f"{self.total_steps // 1000}k"
                path = os.path.join(checkpoint_dir, f"ckpt_{tag}.pt")
                self.save(path)
                self._plot_curves(checkpoint_dir, tag)
                last_ckpt = self.total_steps

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
