"""Performance regression of the trained DVN (the only versioned weights).

Small seeded samples, so thresholds keep a wide margin below the measured
values (CPU, Python 3.14):
- 1P, 30 games, 100 steps max: DVN 71.6 placements on average, greedy 29.0
  (report, Table 1: ~60 vs ~30);
- 3P, 6 games, 60 steps max: DVN depth-2 51.7 (5 games reach the cap),
  greedy depth-2 35.7.
"""
import numpy as np

from blockblast import BlockBlast3PEnv, BlockBlastEnv
from common.evaluation import run_episodes
from common.policies import GreedyLookahead3P, greedy_1p
from dvn.lookahead import DVNLookahead3P


def test_dvn_1p(dvn_agent):
    def evaluate(policy):
        return run_episodes(BlockBlastEnv(punish_for_invalid=-100.0), policy, 30,
                            seed=0, max_steps=100, progress=False)

    dvn = evaluate(lambda obs, env: dvn_agent.select_action(obs, 0.0))
    greedy = evaluate(greedy_1p)
    assert dvn.lengths.mean() > 55
    assert dvn.lengths.mean() > 1.8 * greedy.lengths.mean()


def test_dvn_depth2_3p(dvn_agent):
    def evaluate(policy):
        return run_episodes(BlockBlast3PEnv(), policy, 6, seed=0, max_steps=60, progress=False)

    dvn = evaluate(DVNLookahead3P(dvn_agent, depth=2))
    greedy = evaluate(GreedyLookahead3P(depth=2))
    assert np.median(dvn.lengths) >= 50
    assert dvn.lengths.mean() > greedy.lengths.mean()
