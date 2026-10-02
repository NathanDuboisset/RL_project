"""DVN lookahead policies for the 3-piece env ("DVN depth-2 / depth-3" of the
report, Table 2). Moved unchanged from notebooks/benchmark_3p_clean.ipynb.

Both replan at every step (receding horizon) and play only the first action of
the best sequence -- unlike dvn.planner.RoundPlanner3P, which commits to the
whole 3-action sequence of the round.

Known asymmetries, kept on purpose so that the code matches the reported results:
- depth-2: when the round ends after the first placement, or no next placement
  exists, the action is scored r0 + gamma * V(s1).
- depth-3: in the same situations the action is scored r0 alone (no value
  term); when the round ends after the second placement, the leaf is
  r1 + gamma**2 * V(s2) (one extra gamma); second-level placements after which
  no third placement exists are dropped from the max.
"""
import numpy as np
import torch


def dvn_value_batch(agent, boards, batch_size=256):
    """V(boards) with the agent's policy network, evaluated by chunks."""
    if len(boards) == 0:
        return np.array([], dtype=np.float32)
    boards_np = np.asarray(boards, dtype=np.float32)
    values = []
    for start in range(0, len(boards_np), batch_size):
        chunk = torch.from_numpy(boards_np[start:start + batch_size]).to(agent.device)
        with torch.inference_mode():
            values.append(agent.policy_net(chunk).squeeze(-1).cpu().numpy())
    return np.concatenate(values).astype(np.float32, copy=False)


def dvn_depth2_action(env, agent, gamma=0.99, batch_size=256):
    """a* = argmax_a0 [ r0 + gamma * max_a1 ( r1 + gamma * V(s2) ) ]"""
    n = env.n_pieces
    board0 = env.board
    combo0 = int(env.combo)

    valid_actions = np.flatnonzero(env.valid_placements.reshape(-1))
    if valid_actions.size == 0:
        return None

    eval_boards = []
    r0_per_l0 = []
    meta = []

    for action in valid_actions:
        piece_idx, row, col = env.decode_action(action)

        board1, r0, combo1 = env._simulate_one_hyp_step(board0, combo0, piece_idx, row, col)
        r0_per_l0.append(float(r0))

        pieces_used1 = env.pieces_used.copy()
        pieces_used1[piece_idx] = 1
        remaining = [i for i in range(n) if not pieces_used1[i]]

        if not remaining:
            idx = len(eval_boards)
            eval_boards.append(board1.astype(np.float32))
            meta.append(('d1', idx))
        else:
            start = len(eval_boards)
            l1_rewards = []
            for p1 in remaining:
                valid1 = env._valid_positions_for_piece_on_board(board1, p1)
                rows1, cols1 = np.nonzero(valid1)
                for r1v, c1v in zip(rows1.tolist(), cols1.tolist()):
                    board2, r1, _ = env._simulate_one_hyp_step(board1, combo1, p1, r1v, c1v)
                    eval_boards.append(board2.astype(np.float32))
                    l1_rewards.append(float(r1))

            count = len(eval_boards) - start
            if count == 0:
                idx = len(eval_boards)
                eval_boards.append(board1.astype(np.float32))
                meta.append(('d1', idx))
            else:
                meta.append(('d2', start, count, np.array(l1_rewards, dtype=np.float32)))

    if not eval_boards:
        return int(valid_actions[0])

    v_all = dvn_value_batch(agent, eval_boards, batch_size=batch_size)

    q0 = np.empty(len(valid_actions), dtype=np.float32)
    for i, m in enumerate(meta):
        r0 = r0_per_l0[i]
        if m[0] == 'd1':
            q0[i] = r0 + gamma * float(v_all[m[1]])
        else:
            _, start, count, r1_arr = m
            q1 = r1_arr + gamma * v_all[start:start + count]
            q0[i] = r0 + gamma * float(np.max(q1))

    return int(valid_actions[int(np.argmax(q0))])


def dvn_depth3_action(env, agent, gamma=0.99, batch_size=256):
    """a* = argmax_a0 [ r0 + gamma * max_a1 ( r1 + gamma * max_a2 ( r2 + gamma * V(s3) ) ) ]"""
    n = env.n_pieces
    board0 = env.board
    combo0 = int(env.combo)

    valid_actions0 = np.flatnonzero(env.valid_placements.reshape(-1))
    if valid_actions0.size == 0:
        return None

    leaf_boards = []
    l0_entries = []
    l0_fallback = {}

    for idx_a0, a0 in enumerate(valid_actions0):
        p0, row0, col0 = env.decode_action(a0)
        board1, r0, combo1 = env._simulate_one_hyp_step(board0, combo0, p0, row0, col0)

        used1 = env.pieces_used.copy()
        used1[p0] = 1
        rem1 = [i for i in range(n) if not used1[i]]

        if not rem1:
            l0_fallback[idx_a0] = float(r0)
            continue

        r1_list = []
        r2_list = []
        leaf_start = len(leaf_boards)

        for p1 in rem1:
            valid1 = env._valid_positions_for_piece_on_board(board1, p1)
            rows1, cols1 = np.nonzero(valid1)
            for r1v, c1v in zip(rows1.tolist(), cols1.tolist()):
                board2, r1, combo2 = env._simulate_one_hyp_step(board1, combo1, p1, r1v, c1v)

                used2 = used1.copy()
                used2[p1] = 1
                rem2 = [i for i in range(n) if not used2[i]]

                if not rem2:
                    leaf_boards.append(board2.astype(np.float32))
                    r1_list.append(float(r1))
                    r2_list.append(0.0)
                else:
                    for p2 in rem2:
                        valid2 = env._valid_positions_for_piece_on_board(board2, p2)
                        rows2, cols2 = np.nonzero(valid2)
                        for r2v, c2v in zip(rows2.tolist(), cols2.tolist()):
                            board3, r2, _ = env._simulate_one_hyp_step(board2, combo2, p2, r2v, c2v)
                            leaf_boards.append(board3.astype(np.float32))
                            r1_list.append(float(r1))
                            r2_list.append(float(r2))

        leaf_count = len(leaf_boards) - leaf_start
        if leaf_count == 0:
            l0_fallback[idx_a0] = float(r0)
        else:
            l0_entries.append((idx_a0, float(r0), np.array(r1_list, dtype=np.float32),
                               np.array(r2_list, dtype=np.float32), leaf_start, leaf_count))

    v_leaf = dvn_value_batch(agent, leaf_boards, batch_size=batch_size) if leaf_boards else np.array([], dtype=np.float32)

    q0 = np.full(len(valid_actions0), -np.inf, dtype=np.float32)
    for idx_a0, r0 in l0_fallback.items():
        q0[idx_a0] = float(r0)

    for idx_a0, r0, r1_arr, r2_arr, start, count in l0_entries:
        v = v_leaf[start:start + count]
        q_leaf = r2_arr + gamma * v
        q_l1 = r1_arr + gamma * q_leaf
        q0[idx_a0] = r0 + gamma * float(np.max(q_l1))

    return int(valid_actions0[int(np.argmax(q0))])


class DVNLookahead3P:
    """Policy wrapper (obs, env) -> action for run_episodes."""

    def __init__(self, agent, depth: int = 2, gamma: float = 0.99, batch_size: int = 256):
        assert depth in (2, 3), "depth must be 2 or 3"
        self.agent, self.depth, self.gamma, self.batch_size = agent, depth, gamma, batch_size
        self._fn = dvn_depth2_action if depth == 2 else dvn_depth3_action

    def __call__(self, obs, env):
        return self._fn(env, self.agent, gamma=self.gamma, batch_size=self.batch_size)
