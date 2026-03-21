import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque


def symlog(x: np.ndarray) -> np.ndarray:
    """sign(x) * log(1 + |x|) — compresses the wide reward range for the value network."""
    return np.sign(x) * np.log1p(np.abs(x))


def symexp(y: np.ndarray) -> np.ndarray:
    """Inverse of symlog."""
    return np.sign(y) * (np.expm1(np.abs(y)))


def obs_to_tensors(obs_batch: dict, device: torch.device) -> dict:
    return {
        "board":            torch.as_tensor(obs_batch["board"],            dtype=torch.float32, device=device),
        "pieces":           torch.as_tensor(obs_batch["pieces"],           dtype=torch.float32, device=device),
        "pieces_used":      torch.as_tensor(obs_batch["pieces_used"],      dtype=torch.float32, device=device),
        "combo":            torch.as_tensor(obs_batch["combo"],            dtype=torch.float32, device=device),
        "valid_placements": torch.as_tensor(obs_batch["valid_placements"], dtype=torch.float32, device=device),
    }


def stack_obs(obs_list: list) -> dict:
    return {k: np.stack([o[k] for o in obs_list], axis=0) for k in obs_list[0]}


def valid_to_mask(valid_placements: np.ndarray) -> np.ndarray:
    vp = np.asarray(valid_placements)
    if vp.ndim == 3:
        vp = vp[None]
    return vp.reshape(vp.shape[0], -1).astype(bool)


class BoardEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
        )

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        return self.net(board.unsqueeze(1))


class PiecesEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 64), nn.ReLU(),
        )

    def forward(self, pieces: torch.Tensor) -> torch.Tensor:
        return self.net(pieces)


class ScalarEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 32), nn.ReLU())

    def forward(self, combo: torch.Tensor, pieces_used: torch.Tensor) -> torch.Tensor:
        x = torch.cat([combo / 10.0, pieces_used], dim=-1)
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, action_dim: int = 192):
        super().__init__()
        self.board_enc  = BoardEncoder()
        self.pieces_enc = PiecesEncoder()
        self.scalar_enc = ScalarEncoder()

        self.trunk = nn.Sequential(
            nn.Linear(224, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, action_dim)
        self.value_head  = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def _encode(self, obs_t: dict) -> torch.Tensor:
        b = self.board_enc(obs_t["board"])
        p = self.pieces_enc(obs_t["pieces"])
        s = self.scalar_enc(obs_t["combo"], obs_t["pieces_used"])
        return self.trunk(torch.cat([b, p, s], dim=-1))

    def forward(self, obs_t: dict, action_mask: torch.Tensor = None):
        h      = self._encode(obs_t)
        logits = self.policy_head(h)
        values = self.value_head(h).squeeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))
        return logits, values

    @torch.no_grad()
    def get_action(self, obs_t, action_mask=None, deterministic=False):
        logits, values = self.forward(obs_t, action_mask)
        dist    = Categorical(logits=logits)
        actions = logits.argmax(-1) if deterministic else dist.sample()
        return actions, dist.log_prob(actions), values, dist.entropy()


class RolloutBuffer:
    """Stores T steps × N envs, then computes GAE advantages and returns."""

    def __init__(self, n_steps: int, n_envs: int):
        self.T = n_steps
        self.N = n_envs
        self._alloc()

    def _alloc(self):
        T, N = self.T, self.N
        self.boards      = torch.zeros(T, N, 8, 8)
        self.pieces      = torch.zeros(T, N, 3, 5, 5)
        self.pieces_used = torch.zeros(T, N, 3)
        self.combos      = torch.zeros(T, N, 1)
        self.masks       = torch.zeros(T, N, 192, dtype=torch.bool)
        self.actions     = torch.zeros(T, N, dtype=torch.long)
        self.log_probs   = torch.zeros(T, N)
        self.rewards     = torch.zeros(T, N)
        self.values      = torch.zeros(T, N)
        self.dones       = torch.zeros(T, N)
        self.advantages  = None
        self.returns     = None

    def add(self, step, obs_t, masks, actions, log_probs, rewards, values, dones):
        self.boards[step]      = obs_t["board"].cpu()
        self.pieces[step]      = obs_t["pieces"].cpu()
        self.pieces_used[step] = obs_t["pieces_used"].cpu()
        self.combos[step]      = obs_t["combo"].cpu()
        self.masks[step]       = masks.cpu()
        self.actions[step]     = actions.cpu()
        self.log_probs[step]   = log_probs.cpu()
        self.rewards[step]     = rewards.cpu()
        self.values[step]      = values.cpu()
        self.dones[step]       = dones.cpu()

    def compute_returns_and_advantages(self, last_values, gamma=0.99, gae_lambda=0.95):
        advantages = torch.zeros_like(self.rewards)
        last_gae   = torch.zeros(self.N)
        last_vals  = last_values.cpu()

        for t in reversed(range(self.T)):
            next_v   = last_vals if t == self.T - 1 else self.values[t + 1]
            delta    = self.rewards[t] + gamma * next_v * (1 - self.dones[t]) - self.values[t]
            last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns    = advantages + self.values

    def get_batches(self, batch_size: int, device: torch.device):
        total = self.T * self.N
        idx   = torch.randperm(total)

        boards      = self.boards.reshape(total, 8, 8)
        pieces      = self.pieces.reshape(total, 3, 5, 5)
        pieces_used = self.pieces_used.reshape(total, 3)
        combos      = self.combos.reshape(total, 1)
        masks       = self.masks.reshape(total, 192)
        actions     = self.actions.reshape(total)
        log_probs   = self.log_probs.reshape(total)
        advantages  = self.advantages.reshape(total)
        returns     = self.returns.reshape(total)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for start in range(0, total, batch_size):
            b = idx[start : start + batch_size]
            obs_b = {
                "board":       boards[b].to(device),
                "pieces":      pieces[b].to(device),
                "pieces_used": pieces_used[b].to(device),
                "combo":       combos[b].to(device),
            }
            yield obs_b, masks[b].to(device), actions[b].to(device), \
                  log_probs[b].to(device), advantages[b].to(device), returns[b].to(device)


class PPOTrainer:
    def __init__(
        self,
        envs,
        device=torch.device("cpu"),
        lr=3e-4,
        n_steps=128,
        n_epochs=4,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
    ):
        self.envs          = envs
        self.n_envs        = len(envs)
        self.device        = device
        self.n_steps       = n_steps
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_eps      = clip_eps
        self.vf_coef       = vf_coef
        self.ent_coef      = ent_coef
        self.max_grad_norm = max_grad_norm

        self.model     = ActorCritic().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.buffer    = RolloutBuffer(n_steps, self.n_envs)

        self._obs    = [None] * self.n_envs

        self.ep_returns  = deque(maxlen=100)
        self.ep_lengths  = deque(maxlen=100)
        self._ep_ret     = [0.0] * self.n_envs
        self._ep_len     = [0]   * self.n_envs
        self.total_steps = 0

        self.history = {k: [] for k in (
            "steps", "mean_return", "median_return", "mean_length",
            "loss_policy", "loss_value", "entropy", "clip_frac",
        )}

    def _reset_all_envs(self):
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            self._obs[i] = obs

    @torch.no_grad()
    def _collect_rollout(self):
        self.model.eval()

        for step in range(self.n_steps):
            batch   = stack_obs(self._obs)
            obs_t   = obs_to_tensors(batch, self.device)
            mask_np = valid_to_mask(batch["valid_placements"])
            mask_t  = torch.as_tensor(mask_np, device=self.device)

            actions, log_probs, values, _ = self.model.get_action(obs_t, mask_t)

            raw_rewards = np.zeros(self.n_envs, dtype=np.float32)
            dones       = np.zeros(self.n_envs, dtype=np.float32)

            for i, env in enumerate(self.envs):
                next_obs, rew, term, trunc, _ = env.step(int(actions[i]))
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
                step, obs_t, mask_t, actions, log_probs,
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

    def _ppo_update(self) -> dict:
        self.model.train()
        pg_l, vf_l, ents, clips = [], [], [], []

        for _ in range(self.n_epochs):
            for obs_b, masks_b, acts_b, old_lp_b, adv_b, ret_b in \
                    self.buffer.get_batches(self.batch_size, self.device):

                logits, values = self.model.forward(obs_b, action_mask=masks_b)
                dist    = Categorical(logits=logits)
                new_lp  = dist.log_prob(acts_b)
                entropy = dist.entropy().mean()

                ratio = (new_lp - old_lp_b).exp()
                clips.append(((ratio - 1).abs() > self.clip_eps).float().mean().item())

                pg_loss = torch.max(
                    -adv_b * ratio,
                    -adv_b * ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps),
                ).mean()

                vf_loss = F.mse_loss(values, ret_b)
                loss    = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pg_l.append(pg_loss.item())
                vf_l.append(vf_loss.item())
                ents.append(entropy.item())

        return dict(
            loss_policy=np.mean(pg_l),
            loss_value=np.mean(vf_l),
            entropy=np.mean(ents),
            clip_frac=np.mean(clips),
        )

    def save(self, path: str):
        torch.save({
            "model":       self.model.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "ep_returns":  list(self.ep_returns),
            "ep_lengths":  list(self.ep_lengths),
            "_ep_ret":     self._ep_ret,
            "_ep_len":     self._ep_len,
            "history":     self.history,
        }, path)
        print(f"Saved -> {path}  (step {self.total_steps:,})")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["total_steps"]

        self.ep_returns = deque(ckpt["ep_returns"], maxlen=100)
        self.ep_lengths = deque(ckpt["ep_lengths"], maxlen=100)
        self._ep_ret    = ckpt["_ep_ret"]
        self._ep_len    = ckpt["_ep_len"]

        saved_history = ckpt["history"]
        for k in self.history:
            if k in saved_history:
                self.history[k] = saved_history[k]

        print(
            f"Loaded <- {path}  "
            f"(step {self.total_steps:,} | "
            f"mean_ret {np.mean(self.ep_returns):.1f} over last {len(self.ep_returns)} eps)"
        )
