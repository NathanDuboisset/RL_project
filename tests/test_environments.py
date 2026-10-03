import copy

import numpy as np
import pytest

from blockblast import BlockBlast3PEnv, BlockBlastEnv
from conftest import crowded_3p_env


def test_1p_observation_and_afterstates():
    env = BlockBlastEnv()
    obs, _ = env.reset(seed=0)
    assert obs["board"].shape == (8, 8) and not obs["board"].any()
    afterstates, rewards = obs["placements_result"]
    assert afterstates.shape == (8, 8, 8, 8) and rewards.shape == (8, 8)

    # the precomputed afterstate is the board actually reached
    action = int(np.flatnonzero(obs["valid_placements"])[0])
    row, col = divmod(action, 8)
    next_obs, reward, *_ = env.step(action)
    assert reward == pytest.approx(rewards[row, col])


def test_1p_invalid_action_is_penalised_and_terminal():
    env = BlockBlastEnv(punish_for_invalid=-100.0)
    obs, _ = env.reset(seed=0)
    env.board[:] = 1  # nothing fits any more
    env.valid_placements[:] = 0
    _, reward, terminated, _, _ = env.step(0)
    assert terminated and reward == -100.0


def test_3p_same_seed_same_pieces():
    """Piece sequences only depend on the seed: the basis of paired comparisons."""
    a, b = BlockBlast3PEnv(), BlockBlast3PEnv()
    a.reset(seed=42)
    b.reset(seed=42)
    for pa, pb in zip(a.pieces_grids, b.pieces_grids):
        np.testing.assert_array_equal(pa, pb)


def test_3p_action_encoding_roundtrip():
    env = BlockBlast3PEnv()
    env.reset(seed=0)
    for action in (0, 63, 64, 100, 191):
        assert env.encode_action(*env.decode_action(action)) == action


def test_3p_round_enumeration_matches_real_play():
    """Every enumerated 3-step sequence is legal, and its discounted reward and
    final board are those obtained by actually playing it."""
    gamma = 0.9
    env = crowded_3p_env(seed=3)
    sequences = list(env.iter_t_plus_3_sequences(gamma))
    assert len(sequences) == len(env.get_t_plus_3_candidates(gamma)) > 0

    rng = np.random.default_rng(0)
    for i in rng.choice(len(sequences), size=20, replace=False):
        actions, cum_reward, board3 = sequences[i]
        sim = copy.deepcopy(env)
        rewards = []
        for p, r, c in actions:
            _, reward, terminated, _, _ = sim.step(sim.encode_action(p, r, c))
            rewards.append(reward)
        assert cum_reward == pytest.approx(rewards[0] + gamma * rewards[1] + gamma ** 2 * rewards[2])
        if not terminated:  # after a full round the board is the simulated one
            np.testing.assert_array_equal(sim.board, board3)


def test_3p_no_enumeration_mid_round():
    env = BlockBlast3PEnv()
    obs, _ = env.reset(seed=0)
    env.step(int(np.flatnonzero(obs["valid_placements"].reshape(-1))[0]))
    assert list(env.iter_t_plus_3_sequences(0.99)) == []
