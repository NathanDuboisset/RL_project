from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
DVN_WEIGHTS = ROOT / "final_weights" / "dvn_final_20260313_020137.pt"
DVN_CNN_WEIGHTS = ROOT / "final_weights" / "dvn_1P_60avg.pt"


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def dvn_agent():
    from dvn.agent import load_dvn_agent
    return load_dvn_agent(DVN_WEIGHTS, device="cpu")


@pytest.fixture(scope="session")
def ppo_model():
    """Untrained PPO actor-critic: enough to check that the planners run."""
    from ppo.ppo_agent import ActorCritic
    torch.manual_seed(0)
    return ActorCritic().eval()


def crowded_3p_env(seed=0):
    """3P env whose top 5 rows are filled (except column 0): far fewer
    3-placement sequences than on an empty board (~850k), so planners are fast."""
    from blockblast import BlockBlast3PEnv
    env = BlockBlast3PEnv()
    env.reset(seed=seed)
    env.board[:5, 1:] = 1
    env._update_all_valid_placements()
    env.placements_result = env._get_all_placements_result()
    return env


def play(env, policy, seed=0, max_steps=10):
    """Play up to `max_steps` with `policy(obs, env)`; assert every action is valid.
    seed=None continues from the current state of `env` instead of resetting it."""
    obs = env._get_obs() if seed is None else env.reset(seed=seed)[0]
    if hasattr(policy, "reset"):
        policy.reset()
    actions = []
    for _ in range(max_steps):
        action = policy(obs, env)
        assert action is not None
        assert obs["valid_placements"].reshape(-1)[action], f"invalid action {action}"
        obs, _, terminated, truncated, _ = env.step(int(action))
        actions.append(int(action))
        if terminated or truncated:
            break
    return actions
