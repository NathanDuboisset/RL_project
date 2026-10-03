"""The learning machinery works: on a small *fixed* batch of data, repeated
updates must fit the targets (the loss drops). This catches broken gradients,
wrong targets or detached tensors without running a full training.

Measured loss ratios (last 10 / first 10 updates): DDQN 0.19, Rainbow 0.10,
DVN 0.20, PPO value loss 0.002; the threshold is 0.5.
"""
import random

import numpy as np
import pytest
import torch

from blockblast import BlockBlast3PEnv, BlockBlastEnv
from dqn.agent import DDQNAgent1P, RainbowAgent1P
from dvn.agent import DVNAgent1P
from dvn.models import BlockBlastValueNet1PmultikernelFlattenned
from ppo.ppo_agent import PPOTrainer


def fill_buffer(agent, n_transitions=64, seed=0):
    random.seed(seed)
    env = BlockBlastEnv(punish_for_invalid=-100.0)
    obs, _ = env.reset(seed=seed)
    for _ in range(n_transitions):
        action = agent.select_action(obs, 1.0)  # random valid actions
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.store_transition(obs, action, reward, next_obs, terminated or truncated)
        obs = next_obs if not (terminated or truncated) else env.reset()[0]


AGENTS = {
    "ddqn": lambda: DDQNAgent1P(device="cpu", batch_size=32),
    "rainbow": lambda: RainbowAgent1P(device="cpu", batch_size=32),
    # Huber loss + clipped gradients: with the default lr=1e-4 the fit is correct but slow
    "dvn": lambda: DVNAgent1P(policy_net=BlockBlastValueNet1PmultikernelFlattenned, device="cpu",
                              batch_size=32, lr=1e-3, punish_for_invalid=-100.0),
}


@pytest.mark.parametrize("kind", AGENTS)
def test_value_based_agent_fits_a_fixed_buffer(kind):
    agent = AGENTS[kind]()
    fill_buffer(agent)
    losses = [agent.update_model() for _ in range(300)]  # target network kept fixed
    losses = [l for l in losses if l is not None]
    assert np.mean(losses[-10:]) < 0.5 * np.mean(losses[:10])


def test_ppo_fits_a_fixed_rollout():
    envs = [BlockBlast3PEnv() for _ in range(2)]
    for i, env in enumerate(envs):
        env.reset(seed=i)
    trainer = PPOTrainer(envs, device=torch.device("cpu"), n_steps=64, batch_size=64, n_epochs=4)
    trainer._reset_all_envs()
    trainer._collect_rollout()
    value_losses = [trainer._ppo_update()["loss_value"] for _ in range(30)]
    assert value_losses[-1] < 0.5 * value_losses[0]
