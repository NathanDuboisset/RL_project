import time
import numpy as np
import torch
from collections import defaultdict
from typing import Callable

from mct.ppo_agent import obs_to_tensors, symexp


def _board_to_obs(
    board: np.ndarray,
    pieces_padded: np.ndarray,
    pieces_used: np.ndarray,
    combo: int,
) -> dict:
    return {
        "board":       board[None].astype(np.float32),
        "pieces":      pieces_padded[None].astype(np.float32),
        "pieces_used": pieces_used[None].astype(np.float32),
        "combo":       np.array([[combo]], dtype=np.float32),
    }


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
        self.model        = model
        self.device       = device
        self.gamma        = gamma
        self.batch_size   = batch_size
        self.verbose      = verbose
        self.value_weight = value_weight

        self.model.eval()

    @torch.no_grad()
    def _value_batch(
        self,
        boards: np.ndarray,
        pieces_padded: np.ndarray,
        pieces_used: np.ndarray,
        combo: int,
    ) -> np.ndarray:
        N = boards.shape[0]
        values = np.zeros(N, dtype=np.float32)

        pieces_tiled = np.tile(pieces_padded[None], (N, 1, 1, 1))
        used_tiled   = np.tile(pieces_used[None],   (N, 1))
        combo_arr    = np.full((N, 1), combo, dtype=np.float32)

        for start in range(0, N, self.batch_size):
            end   = min(start + self.batch_size, N)
            obs_b = {
                "board":       torch.as_tensor(boards[start:end],       device=self.device),
                "pieces":      torch.as_tensor(pieces_tiled[start:end], device=self.device),
                "pieces_used": torch.as_tensor(used_tiled[start:end],   device=self.device),
                "combo":       torch.as_tensor(combo_arr[start:end],    device=self.device),
            }
            _, v = self.model.forward(obs_b)
            values[start:end] = v.cpu().numpy()

        return symexp(values)

    @torch.no_grad()
    def _value_batch_raw(
        self,
        boards: np.ndarray,
        pieces_padded: np.ndarray,
        pieces_used: np.ndarray,
        combo: int,
    ) -> np.ndarray:
        """Same as _value_batch but stays in symlog space (no symexp)."""
        N = boards.shape[0]
        values = np.zeros(N, dtype=np.float32)

        pieces_tiled = np.tile(pieces_padded[None], (N, 1, 1, 1))
        used_tiled   = np.tile(pieces_used[None],   (N, 1))
        combo_arr    = np.full((N, 1), combo, dtype=np.float32)

        for start in range(0, N, self.batch_size):
            end   = min(start + self.batch_size, N)
            obs_b = {
                "board":       torch.as_tensor(boards[start:end],       device=self.device),
                "pieces":      torch.as_tensor(pieces_tiled[start:end], device=self.device),
                "pieces_used": torch.as_tensor(used_tiled[start:end],   device=self.device),
                "combo":       torch.as_tensor(combo_arr[start:end],    device=self.device),
            }
            _, v = self.model.forward(obs_b)
            values[start:end] = v.cpu().numpy()

        return values

    def _score_candidates(self, env) -> list:
        """
        Score all 3-step candidates in symlog space:
            score = symlog(3-step rewards) + value_weight * V_symlog(S(t+3))
        """
        candidates = env.get_t_plus_3_candidates(self.gamma)
        if not candidates:
            return []

        boards = np.stack(
            [c["state_t_plus_3"] for c in candidates], axis=0
        ).astype(np.float32)

        pieces_padded = np.zeros((3, 5, 5), dtype=np.float32)
        pieces_used   = np.ones(3,          dtype=np.float32)
        combo_after   = 0

        raw_rewards = np.array(
            [c["cumulative_reward_3steps"] for c in candidates], dtype=np.float32
        )
        from mct.ppo_agent import symlog as _symlog
        rewards_sl = _symlog(raw_rewards)

        if self.value_weight > 0.0:
            v_raw = self._value_batch_raw(boards, pieces_padded, pieces_used, combo_after)
        else:
            v_raw = np.zeros(len(candidates), dtype=np.float32)

        for i, c in enumerate(candidates):
            c["score"]       = float(rewards_sl[i] + self.value_weight * v_raw[i])
            c["reward_term"] = float(rewards_sl[i])
            c["value_term"]  = float(v_raw[i])

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    def select_round(self, env) -> list:
        """Plan the full round at once; returns a list of 3 actions."""
        t0 = time.time()
        candidates = self._score_candidates(env)

        if not candidates:
            return []

        best = candidates[0]
        triplet = []
        for piece_idx, row, col in best["actions"]:
            action = piece_idx * (env.grid_size * env.grid_size) + row * env.grid_size + col
            triplet.append(int(action))

        if self.verbose:
            elapsed = time.time() - t0
            print(
                f"[MCTS] {len(candidates)} candidates | "
                f"best score {best['score']:.2f} "
                f"(3-step rew {best['cumulative_reward_3steps']:.2f}) | "
                f"{elapsed*1000:.1f}ms"
            )
        return triplet

    def select_action(self, env) -> int:
        """Single-step wrapper. Prefer select_round() for full triplet planning."""
        triplet = self.select_round(env)
        if triplet:
            return triplet[0]
        return self._greedy_fallback(env)

    def _greedy_fallback(self, env) -> int:
        obs = env._get_obs()
        batch = {k: v[None] for k, v in obs.items()
                 if k in ("board", "pieces", "pieces_used", "combo", "valid_placements")}
        obs_t = obs_to_tensors(
            {k: v.astype(np.float32) for k, v in batch.items()},
            self.device,
        )
        from mct.ppo_agent import valid_to_mask
        mask_t = torch.as_tensor(
            valid_to_mask(batch["valid_placements"]), device=self.device
        )
        with torch.no_grad():
            actions, *_ = self.model.get_action(obs_t, mask_t, deterministic=True)
        return int(actions[0])

    def evaluate(
        self,
        env_fn: Callable,
        n_episodes: int = 100,
        use_mcts: bool = True,
    ) -> dict:
        returns, lengths, round_times = [], [], []

        for ep in range(n_episodes):
            env = env_fn()
            obs, _ = env.reset()

            total_r      = 0.0
            n_steps      = 0
            action_queue = []

            while True:
                if use_mcts:
                    if not action_queue:
                        t0 = time.time()
                        action_queue = self.select_round(env)
                        round_times.append((time.time() - t0) * 1000)

                        if not action_queue:
                            action_queue = [self._greedy_fallback(env)]

                    action = action_queue.pop(0)
                else:
                    action = self._greedy_fallback(env)

                obs, r, term, trunc, _ = env.step(action)
                total_r += r
                n_steps += 1

                if term or trunc:
                    action_queue = []
                    break

                if np.all(obs["pieces_used"] == 0):
                    action_queue = []

            returns.append(total_r)
            lengths.append(n_steps)
            env.close()

            if (ep + 1) % 10 == 0:
                print(
                    f"  Episode {ep+1:>4}/{n_episodes} | "
                    f"return {total_r:>8.1f} | "
                    f"length {n_steps:>4} steps"
                )

        stats = {
            "mean_return":            float(np.mean(returns)),
            "std_return":             float(np.std(returns)),
            "median_return":          float(np.median(returns)),
            "mean_length":            float(np.mean(lengths)),
            "mean_time_per_round_ms": float(np.mean(round_times)) if round_times else 0.0,
        }

        print(
            f"\n=== MCTS Evaluation ({n_episodes} episodes) ===\n"
            f"  Mean return   : {stats['mean_return']:.2f} +/- {stats['std_return']:.2f}\n"
            f"  Median return : {stats['median_return']:.2f}\n"
            f"  Mean length   : {stats['mean_length']:.1f} steps\n"
            f"  Time/round    : {stats['mean_time_per_round_ms']:.1f} ms\n"
        )
        return stats


def compare_ppo_vs_mcts(
    model,
    env_fn: Callable,
    device: torch.device = torch.device("cpu"),
    n_episodes: int = 100,
    gamma: float = 0.99,
) -> dict:
    agent = MCTSAgent(model, device=device, gamma=gamma, verbose=False)

    print("--- PPO greedy ---")
    ppo_stats = agent.evaluate(env_fn, n_episodes=n_episodes, use_mcts=False)

    print("\n--- MCTS (depth-3 exhaustive) ---")
    mcts_stats = agent.evaluate(env_fn, n_episodes=n_episodes, use_mcts=True)

    delta_ret = mcts_stats["mean_return"] - ppo_stats["mean_return"]
    delta_len = mcts_stats["mean_length"] - ppo_stats["mean_length"]

    print("\n=== Comparison ===")
    print(f"  Return gain   : {delta_ret:+.2f}  "
          f"({delta_ret / max(abs(ppo_stats['mean_return']), 1) * 100:+.1f}%)")
    print(f"  Length gain   : {delta_len:+.1f} steps")
    print(f"  MCTS overhead : {mcts_stats['mean_time_per_round_ms']:.1f} ms / round")

    return {"ppo": ppo_stats, "mcts": mcts_stats}
