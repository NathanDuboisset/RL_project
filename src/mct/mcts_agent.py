"""
mcts_agent.py
=============
Round-level exhaustive search agent for BlockBlast3PEnv.

Strategy
--------
At the start of each round (3 pieces available), enumerate ALL valid
sequences of 3 placements via env.get_t_plus_3_candidates(). Score each
terminal state S(t+3) with:

    score = cumulative_reward_3steps + gamma^3 * symexp(V(S(t+3)))

where V is the PPO value network (which predicts symlog-transformed returns,
so we invert with symexp to get back to the raw reward scale for comparison).

Then play the FIRST action of the best-scoring sequence.

This is not stochastic MCTS -- it is a full depth-3 minimax / best-first
search, which is exact because the 3 pieces of the current round are known
and the environment is deterministic within a round.

Complexity per round
--------------------
  ~50 valid positions per piece * 3! orderings = 6 * 50^3 ~ 750k nodes
  In practice pruned to << 100k because boards fill up.
  Each node is cheap: numpy board copy + value network inference (batched).

Public API
----------
    agent = MCTSAgent(model, device, gamma=0.99, batch_size=512)
    action = agent.select_action(env)          # single step
    stats  = agent.evaluate(env_fn, n=100)     # full eval loop
"""

import time
import numpy as np
import torch
from collections import defaultdict
from typing import Callable

from mct.ppo_agent import obs_to_tensors, symexp


# ---------------------------------------------------------------------------
# Helper: build a minimal obs dict for the value network from a raw board
# ---------------------------------------------------------------------------

def _board_to_obs(
    board: np.ndarray,
    pieces_padded: np.ndarray,
    pieces_used: np.ndarray,
    combo: int,
) -> dict:
    """
    Build the obs dict expected by ActorCritic from raw numpy arrays.
    All arrays get a leading batch dimension of 1.
    """
    return {
        "board":       board[None].astype(np.float32),           # (1,8,8)
        "pieces":      pieces_padded[None].astype(np.float32),   # (1,3,5,5)
        "pieces_used": pieces_used[None].astype(np.float32),     # (1,3)
        "combo":       np.array([[combo]], dtype=np.float32),     # (1,1)
    }


# ---------------------------------------------------------------------------
# MCTSAgent
# ---------------------------------------------------------------------------

class MCTSAgent:
    """
    Parameters
    ----------
    model       : ActorCritic   trained PPO model (used for value estimates)
    device      : torch.device
    gamma       : discount factor (should match training gamma)
    batch_size  : how many S(t+3) states to score in one forward pass
    verbose     : print timing info per round
    """

    def __init__(
        self,
        model,
        device: torch.device = torch.device("cpu"),
        gamma: float = 0.99,
        batch_size: int = 512,
        verbose: bool = False,
        value_weight: float = 0.0,
    ):
        """
        value_weight : weight on the value network term in the scoring formula.

        Formula (all in symlog space for comparable scales):
            score = symlog(rewards_3_coups) + value_weight * V_symlog(S(t+3))

        OdG calibration (from dataset analysis):
            symlog(rewards_3_coups) : 0.26 -> 6.34  (mean ~3.9)
            V_symlog                : ~14.18          (always above reward range)

        Recommended values to sweep:
            0.0  -> pure reward, value ignored   (baseline)
            0.05 -> light value guidance
            0.1  -> balanced
            0.3  -> value starts to dominate
            0.5  -> value dominates
            1.0  -> value crushes rewards        (original behaviour)
        """
        self.model        = model
        self.device       = device
        self.gamma        = gamma
        self.batch_size   = batch_size
        self.verbose      = verbose
        self.value_weight = value_weight

        self.model.eval()

    # -------------------------------------------------------------------------
    # Value estimation (batched)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _value_batch(
        self,
        boards: np.ndarray,
        pieces_padded: np.ndarray,
        pieces_used: np.ndarray,
        combo: int,
    ) -> np.ndarray:
        """
        Estimate V(s) for a batch of boards.

        boards        : (N, 8, 8)  float32
        pieces_padded : (3, 5, 5)  -- same for all candidates (same round)
        pieces_used   : (3,)       -- all zeros after a full round

        Returns raw (symexp-inverted) value estimates, shape (N,).
        """
        N = boards.shape[0]
        values = np.zeros(N, dtype=np.float32)

        # tile the piece/combo info to match the batch
        pieces_tiled = np.tile(pieces_padded[None], (N, 1, 1, 1))   # (N,3,5,5)
        used_tiled   = np.tile(pieces_used[None],   (N, 1))          # (N,3)
        combo_arr    = np.full((N, 1), combo, dtype=np.float32)      # (N,1)

        for start in range(0, N, self.batch_size):
            end   = min(start + self.batch_size, N)
            obs_b = {
                "board":       torch.as_tensor(boards[start:end],       device=self.device),
                "pieces":      torch.as_tensor(pieces_tiled[start:end], device=self.device),
                "pieces_used": torch.as_tensor(used_tiled[start:end],   device=self.device),
                "combo":       torch.as_tensor(combo_arr[start:end],    device=self.device),
            }
            _, v = self.model.forward(obs_b)           # v in symlog space
            values[start:end] = v.cpu().numpy()

        # invert symlog -> raw value scale
        return symexp(values)

    @torch.no_grad()
    def _value_batch_raw(
        self,
        boards: np.ndarray,
        pieces_padded: np.ndarray,
        pieces_used: np.ndarray,
        combo: int,
    ) -> np.ndarray:
        """Same as _value_batch but returns raw symlog predictions (no symexp).
        Used when scoring in symlog space for scale-compatible mixing with rewards."""
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

        return values   # symlog space, ~14 for this model

    # -------------------------------------------------------------------------
    # Core: score all candidates for the current round
    # -------------------------------------------------------------------------

    def _score_candidates(self, env) -> list:
        """
        Call env.get_t_plus_3_candidates() and score every terminal state.

        Scoring formula (symlog space for comparable scales):
            score = symlog(rewards_3_coups) + value_weight * V_symlog(S(t+3))

        Both terms are in symlog space:
            symlog(rewards) : 0.26 -> 6.34
            V_symlog        : ~14  (biased high but relatively consistent)

        value_weight=0.0 -> pure reward ranking (ignores value network)
        value_weight=0.1 -> balanced mix
        value_weight=1.0 -> value dominates

        Returns the list of candidates sorted by score descending.
        """
        candidates = env.get_t_plus_3_candidates(self.gamma)
        if not candidates:
            return []

        # Extract terminal boards
        boards = np.stack(
            [c["state_t_plus_3"] for c in candidates], axis=0
        ).astype(np.float32)   # (N, 8, 8)

        pieces_padded = np.zeros((3, 5, 5), dtype=np.float32)
        pieces_used   = np.ones(3,          dtype=np.float32)
        combo_after   = 0

        # Reward term: symlog of cumulative 3-step reward
        raw_rewards = np.array(
            [c["cumulative_reward_3steps"] for c in candidates], dtype=np.float32
        )
        from mct.ppo_agent import symlog as _symlog
        rewards_sl = _symlog(raw_rewards)   # (N,) in symlog space, range ~0.26-6.34

        # Value term: raw symlog predictions from value network (~14 for this model)
        if self.value_weight > 0.0:
            v_raw = self._value_batch_raw(
                boards, pieces_padded, pieces_used, combo_after
            )   # (N,) in symlog space
        else:
            v_raw = np.zeros(len(candidates), dtype=np.float32)

        # Combined score
        for i, c in enumerate(candidates):
            c["score"]          = float(rewards_sl[i] + self.value_weight * v_raw[i])
            c["reward_term"]    = float(rewards_sl[i])
            c["value_term"]     = float(v_raw[i])

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    # -------------------------------------------------------------------------
    # Public: select_action  (triplet-aware)
    # -------------------------------------------------------------------------

    def select_round(self, env) -> list:
        """
        Plan the full round: score all 3-step sequences and return the
        best triplet as a list of 3 integer actions.

        Should be called ONCE at round start (pieces_used == [0,0,0]).
        Returns a list of 3 actions to execute in order.
        """
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
        """
        Single-step wrapper kept for backward compatibility.
        Prefer select_round() for full triplet planning.
        """
        triplet = self.select_round(env)
        if triplet:
            return triplet[0]
        return self._greedy_fallback(env)

    def _greedy_fallback(self, env) -> int:
        """Value-greedy single-step fallback for mid-round calls."""
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

    # -------------------------------------------------------------------------
    # Public: evaluate
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        env_fn: Callable,
        n_episodes: int = 100,
        use_mcts: bool = True,
    ) -> dict:
        """
        Run n_episodes and return aggregate stats.

        Uses triplet-aware planning: MCTS plans all 3 actions at round start
        and executes them in order without re-searching mid-round.
        3x fewer MCTS calls vs the original single-action approach.

        Parameters
        ----------
        env_fn      : callable with no args that returns a fresh env instance
        n_episodes  : number of episodes to run
        use_mcts    : if False, falls back to pure PPO greedy (for comparison)

        Returns
        -------
        dict with mean_return, std_return, median_return, mean_length,
             mean_time_per_round_ms
        """
        returns, lengths, round_times = [], [], []

        for ep in range(n_episodes):
            env = env_fn()
            obs, _ = env.reset()

            total_r      = 0.0
            n_steps      = 0
            action_queue = []   # remaining planned actions for current round

            while True:
                if use_mcts:
                    if not action_queue:
                        # Round boundary: plan all 3 actions at once
                        t0 = time.time()
                        action_queue = self.select_round(env)
                        round_times.append((time.time() - t0) * 1000)

                        if not action_queue:
                            # Fallback if MCTS returns empty
                            action_queue = [self._greedy_fallback(env)]

                    action = action_queue.pop(0)
                else:
                    action = self._greedy_fallback(env)

                obs, r, term, trunc, _ = env.step(action)
                total_r += r
                n_steps += 1

                # If episode ends mid-round, clear the queue
                if term or trunc:
                    action_queue = []
                    break

                # If round ended (all pieces used), clear queue so MCTS
                # replans fresh for the next round
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
            "mean_return":          float(np.mean(returns)),
            "std_return":           float(np.std(returns)),
            "median_return":        float(np.median(returns)),
            "mean_length":          float(np.mean(lengths)),
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


# ---------------------------------------------------------------------------
# Comparison helper: PPO greedy vs MCTS side by side
# ---------------------------------------------------------------------------

def compare_ppo_vs_mcts(
    model,
    env_fn: Callable,
    device: torch.device = torch.device("cpu"),
    n_episodes: int = 100,
    gamma: float = 0.99,
) -> dict:
    """
    Run the same env_fn n_episodes times with both PPO greedy and MCTS,
    and print a comparison table.

    Usage (notebook):
        from mct.mcts_agent import compare_ppo_vs_mcts
        from block_blast_3p_env import BlockBlast3PEnv

        results = compare_ppo_vs_mcts(
            model    = trainer.model,
            env_fn   = lambda: BlockBlast3PEnv(),
            device   = torch.device("cpu"),
            n_episodes = 100,
        )
    """
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
