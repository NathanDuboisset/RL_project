"""Baseline policies (no learning), usable with common.evaluation.run_episodes."""
import numpy as np


class RandomPolicy:
    """Uniformly random valid action (1P and 3P)."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, env):
        valid_actions = np.flatnonzero(obs["valid_placements"].reshape(-1))
        if valid_actions.size == 0:
            return None
        return int(self.rng.choice(valid_actions))


def greedy_1p(obs, env=None):
    """1P: valid action with the highest immediate (shaped) reward."""
    valid_actions = np.flatnonzero(obs["valid_placements"].reshape(-1))
    if valid_actions.size == 0:
        return None
    hyp_rewards = obs["placements_result"][1].reshape(-1)
    return int(valid_actions[int(np.argmax(hyp_rewards[valid_actions]))])


def _best_immediate_sum(env, board, combo, used, depth):
    """Max sum of immediate rewards over `depth` more placements within the round.

    Returns 0 when depth is exhausted, when the round has no piece left, or when
    no piece fits (a dead end is not penalised: this is a pure greedy baseline).
    """
    if depth == 0:
        return 0.0
    best = -np.inf
    for p in range(env.n_pieces):
        if used[p]:
            continue
        rows, cols = np.nonzero(env._valid_positions_for_piece_on_board(board, p))
        if rows.size == 0:
            continue
        used_next = used.copy()
        used_next[p] = 1
        for r, c in zip(rows.tolist(), cols.tolist()):
            next_board, reward, next_combo = env._simulate_one_hyp_step(board, combo, p, r, c)
            best = max(best, float(reward) + _best_immediate_sum(env, next_board, next_combo, used_next, depth - 1))
    return best if best > -np.inf else 0.0


class GreedyLookahead3P:
    """3P "Greedy depth-k" of the report: exhaustive search over the next `depth`
    placements of the current round, maximising the (undiscounted) sum of
    immediate rewards. Replans at every step; depth=1 is the plain greedy."""

    def __init__(self, depth: int):
        assert depth >= 1
        self.depth = depth

    def __call__(self, obs, env):
        valid_actions = np.flatnonzero(env.valid_placements.reshape(-1))
        if valid_actions.size == 0:
            return None

        best_total, best_action = -np.inf, int(valid_actions[0])
        for action in valid_actions:
            p0, row, col = env.decode_action(action)
            board1, r0, combo1 = env._simulate_one_hyp_step(env.board, int(env.combo), p0, row, col)
            used1 = env.pieces_used.copy()
            used1[p0] = 1
            total = float(r0) + _best_immediate_sum(env, board1, combo1, used1, self.depth - 1)
            if total > best_total:
                best_total, best_action = total, int(action)
        return best_action
