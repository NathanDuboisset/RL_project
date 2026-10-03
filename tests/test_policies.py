"""Every policy of the report plays valid moves; evaluation is reproducible."""
import numpy as np
import pytest

from blockblast import BlockBlast3PEnv, BlockBlastEnv
from common.evaluation import run_episodes
from common.policies import GreedyLookahead3P, RandomPolicy, greedy_1p
from conftest import crowded_3p_env, play
from dvn.lookahead import DVNLookahead3P
from dvn.planner import RoundPlanner3P
from ppo.mcts_agent import MCTSAgent, MCTSAgentFirstOnly, PPOGreedy


def test_1p_baselines_play_valid_moves():
    play(BlockBlastEnv(), greedy_1p, max_steps=30)
    play(BlockBlastEnv(), RandomPolicy(0), max_steps=30)


@pytest.mark.parametrize("depth", [1, 2])
def test_greedy_lookahead_3p(depth):
    play(BlockBlast3PEnv(), GreedyLookahead3P(depth), max_steps=10)


def test_greedy_depth1_maximises_immediate_reward():
    env = BlockBlast3PEnv()
    obs, _ = env.reset(seed=5)
    rewards = {}
    for a in np.flatnonzero(obs["valid_placements"].reshape(-1)):
        p, r, c = env.decode_action(a)
        rewards[int(a)] = env._simulate_one_hyp_step(env.board, env.combo, p, r, c)[1]
    assert rewards[GreedyLookahead3P(1)(obs, env)] == max(rewards.values())


def test_dvn_lookahead_depth2(dvn_agent):
    play(BlockBlast3PEnv(), DVNLookahead3P(dvn_agent, depth=2), max_steps=10)


def test_dvn_lookahead_depth3(dvn_agent):
    play(crowded_3p_env(), DVNLookahead3P(dvn_agent, depth=3), seed=None, max_steps=3)


def test_round_planner_commits_to_a_full_round(dvn_agent):
    planner = RoundPlanner3P(gamma=0.99, agent=dvn_agent, seed=0)
    env = crowded_3p_env()
    planner.reset()
    action = planner(env._get_obs(), env)
    queued = list(planner.plan_actions)
    assert len(queued) == 2  # first action returned, the other two queued
    for expected in queued:
        obs, _, terminated, _, _ = env.step(action)
        assert not terminated
        action = planner(obs, env)
        assert action == expected
    assert env.pieces_used.sum() == 2  # the third piece of the round is about to be placed


def test_ppo_greedy(ppo_model):
    play(BlockBlast3PEnv(), PPOGreedy(ppo_model), max_steps=10)


@pytest.mark.parametrize("make", [
    lambda m: MCTSAgent(m, value_weight=0.3),
    lambda m: MCTSAgent(m, value_weight=0.0),
    lambda m: MCTSAgentFirstOnly(m),
], ids=["full-triplet", "full-triplet-no-value", "first-only"])
def test_ppo_round_search(ppo_model, make):
    play(crowded_3p_env(), make(ppo_model), seed=None, max_steps=6)


def test_run_episodes_is_reproducible():
    def run():
        return run_episodes(BlockBlast3PEnv(), RandomPolicy(1), 5, seed=7, progress=False)
    a, b = run(), run()
    np.testing.assert_array_equal(a.returns, b.returns)
    np.testing.assert_array_equal(a.lengths, b.lengths)
    assert a.summary()["return"]["n"] == 5
