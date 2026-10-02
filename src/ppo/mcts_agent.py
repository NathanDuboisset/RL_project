"""Tree-search planners guided by the PPO value network (report, Section 5.3).

Both planners enumerate, at the start of a round, every valid sequence of 3
placements (env.iter_t_plus_3_sequences). The pieces of the next round are
unknown, so the terminal boards S(t+3) are evaluated by the PPO critic with a
neutral piece context (empty pieces, all used, combo 0).

- MCTSAgentFirstOnly ("MCTS First Only"):
      score = R3 + gamma^3 * symexp(V(S(t+3)))
  plays the first action only; the 2 other pieces of the round are played by
  the PPO policy (argmax of the masked logits).
- MCTSAgent ("MCTS Full Triplet"):
      score = symlog(R3) + value_weight * V_symlog(S(t+3))
  and executes the 3 actions of the best sequence.
with R3 = r_t + gamma * r_t+1 + gamma^2 * r_t+2.

Despite the name, neither is a Monte Carlo Tree Search: the search is exhaustive.

All policies are callables (obs, env) -> action usable with
common.evaluation.run_episodes.
"""
import time
from typing import Callable

import numpy as np
import torch

from common.evaluation import run_episodes
from ppo.ppo_agent import obs_to_tensors, symexp, symlog, valid_to_mask


@torch.no_grad()
def ppo_values(model, boards: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    """Critic values (in symlog space) of `boards` with a neutral piece context."""
    n = boards.shape[0]
    values = np.zeros(n, dtype=np.float32)
    pieces = np.zeros((n, 3, 5, 5), dtype=np.float32)
    used = np.ones((n, 3), dtype=np.float32)
    combo = np.zeros((n, 1), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        obs_b = {
            "board":       torch.as_tensor(boards[start:end], device=device),
            "pieces":      torch.as_tensor(pieces[start:end], device=device),
            "pieces_used": torch.as_tensor(used[start:end],   device=device),
            "combo":       torch.as_tensor(combo[start:end],  device=device),
        }
        _, v = model.forward(obs_b)
        values[start:end] = v.cpu().numpy()
    return values


@torch.no_grad()
def ppo_greedy_action(model, env, device) -> int:
    """PPO policy, deterministic: argmax of the masked logits."""
    obs = env._get_obs()
    batch = {k: v[None] for k, v in obs.items()
             if k in ("board", "pieces", "pieces_used", "combo", "valid_placements")}
    obs_t = obs_to_tensors({k: v.astype(np.float32) for k, v in batch.items()}, device)
    mask_t = torch.as_tensor(valid_to_mask(batch["valid_placements"]), device=device)
    actions, *_ = model.get_action(obs_t, mask_t, deterministic=True)
    return int(actions[0])


def _enumerate_round(env, gamma):
    """(actions list, R3 array float64, S(t+3) boards float32) for the current round."""
    actions, rewards, boards = [], [], []
    for acts, r3, board3 in env.iter_t_plus_3_sequences(gamma):
        actions.append(acts)
        rewards.append(r3)
        boards.append(board3)
    if not actions:
        return [], None, None
    return actions, np.asarray(rewards, dtype=np.float64), np.stack(boards).astype(np.float32)


class PPOGreedy:
    """Pure PPO baseline ("PPO greedy")."""

    def __init__(self, model, device=torch.device("cpu")):
        self.model, self.device = model.eval(), device

    def __call__(self, obs, env) -> int:
        return ppo_greedy_action(self.model, env, self.device)


class MCTSAgentFirstOnly:
    def __init__(
        self,
        model,
        device: torch.device = torch.device("cpu"),
        gamma: float = 0.99,
        batch_size: int = 512,
        verbose: bool = False,
    ):
        self.model      = model.eval()
        self.device     = device
        self.gamma      = gamma
        self.batch_size = batch_size
        self.verbose    = verbose

    def select_action(self, env) -> int:
        t0 = time.time()
        actions, rewards, boards = _enumerate_round(env, self.gamma)
        if not actions:  # mid-round: the PPO policy plays the remaining pieces
            return ppo_greedy_action(self.model, env, self.device)

        values = symexp(ppo_values(self.model, boards, self.device, self.batch_size))
        # float64 arithmetic, as in the original per-candidate Python loop
        scores = rewards + self.gamma ** 3 * values.astype(np.float64)
        best = int(np.argmax(scores))
        if self.verbose:
            print(f"[MCTS] {len(actions)} candidates | best score {scores[best]:.2f} "
                  f"(3-step rew {rewards[best]:.2f}) | {(time.time() - t0) * 1000:.1f}ms")
        return env.encode_action(*actions[best][0])

    def __call__(self, obs, env) -> int:
        return self.select_action(env)


class MCTSAgent:
    def __init__(
        self,
        model,
        device: torch.device = torch.device("cpu"),
        gamma: float = 0.99,
        batch_size: int = 512,
        verbose: bool = False,
        value_weight: float = 0.0,
    ):
        self.model        = model.eval()
        self.device       = device
        self.gamma        = gamma
        self.batch_size   = batch_size
        self.verbose      = verbose
        self.value_weight = value_weight
        self._queue: list[int] = []

    def select_round(self, env) -> list:
        """Plan the full round at once; returns the 3 actions (or [] mid-round)."""
        t0 = time.time()
        actions, rewards, boards = _enumerate_round(env, self.gamma)
        if not actions:
            return []

        rewards_sl = symlog(rewards.astype(np.float32))
        if self.value_weight > 0.0:
            v_sl = ppo_values(self.model, boards, self.device, self.batch_size)
        else:
            v_sl = np.zeros(len(actions), dtype=np.float32)
        scores = rewards_sl + self.value_weight * v_sl
        best = int(np.argmax(scores))

        if self.verbose:
            print(f"[MCTS] {len(actions)} candidates | best score {scores[best]:.2f} "
                  f"(3-step rew {rewards[best]:.2f}) | {(time.time() - t0) * 1000:.1f}ms")
        return [env.encode_action(p, r, c) for p, r, c in actions[best]]

    def select_action(self, env) -> int:
        """Single-step wrapper (replans every step). Use the agent as a policy
        (`agent(obs, env)`) to execute the full triplet."""
        triplet = self.select_round(env)
        return triplet[0] if triplet else ppo_greedy_action(self.model, env, self.device)

    def reset(self) -> None:
        self._queue = []

    def __call__(self, obs, env) -> int:
        if np.all(obs["pieces_used"] == 0):  # new round
            self._queue = []
        if not self._queue:
            self._queue = self.select_round(env) or [ppo_greedy_action(self.model, env, self.device)]
        return self._queue.pop(0)

    def evaluate(self, env_fn: Callable, n_episodes: int = 100, use_mcts: bool = True, seed: int = 0) -> dict:
        """Kept for the notebooks. use_mcts=False evaluates the PPO policy alone."""
        policy = self if use_mcts else PPOGreedy(self.model, self.device)
        stats = run_episodes(env_fn(), policy, n_episodes, seed=seed, desc="MCTS" if use_mcts else "PPO greedy")
        return stats_as_dict(stats)


def stats_as_dict(stats) -> dict:
    s = stats.summary()
    return {
        "mean_return":            s["return"]["mean"],
        "std_return":             s["return"]["std"],
        "median_return":          s["return"]["median"],
        "mean_length":            s["length"]["mean"],
        "mean_time_per_decision_ms": s["ms_per_decision"],
        "returns":                stats.returns,
        "lengths":                stats.lengths,
    }


def compare_ppo_vs_mcts(
    model,
    env_fn: Callable,
    device: torch.device = torch.device("cpu"),
    n_episodes: int = 100,
    gamma: float = 0.99,
    value_weight: float = 0.0,
    seed: int = 0,
) -> dict:
    agent = MCTSAgent(model, device=device, gamma=gamma, value_weight=value_weight)
    ppo_stats = agent.evaluate(env_fn, n_episodes=n_episodes, use_mcts=False, seed=seed)
    mcts_stats = agent.evaluate(env_fn, n_episodes=n_episodes, use_mcts=True, seed=seed)

    delta_ret = mcts_stats["mean_return"] - ppo_stats["mean_return"]
    print("\n=== Comparison ===")
    print(f"  Return gain   : {delta_ret:+.2f}  "
          f"({delta_ret / max(abs(ppo_stats['mean_return']), 1) * 100:+.1f}%)")
    print(f"  Length gain   : {mcts_stats['mean_length'] - ppo_stats['mean_length']:+.1f} steps")
    print(f"  MCTS overhead : {mcts_stats['mean_time_per_decision_ms']:.1f} ms / decision")
    return {"ppo": ppo_stats, "mcts": mcts_stats}
