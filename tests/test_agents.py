import numpy as np
import pytest
import torch

from blockblast import BlockBlastEnv
from conftest import DVN_CNN_WEIGHTS, DVN_WEIGHTS
from dqn.agent import DDQNAgent1P, RainbowAgent1P
from dvn.agent import DVNAgent1P, load_dvn_agent
from dvn.models import BlockBlastValueNet1P, BlockBlastValueNet1PmultikernelFlattenned
from ppo.ppo_agent import ActorCritic


def n_params(net):
    return sum(p.numel() for p in net.parameters())


@pytest.mark.parametrize("make_net, expected", [
    (lambda: DDQNAgent1P(device="cpu").policy_net, 2_674_464),
    (lambda: RainbowAgent1P(device="cpu").policy_net, 4_755_910),
    (BlockBlastValueNet1PmultikernelFlattenned, 54_529),
    (BlockBlastValueNet1P, 1_345_121),
    (ActorCritic, 880_513),
])
def test_parameter_counts(make_net, expected):
    """Pins the architectures (see 'Known differences' in the README)."""
    assert n_params(make_net()) == expected


def make_agent(kind):
    if kind == "ddqn":
        return DDQNAgent1P(device="cpu", batch_size=16)
    if kind == "rainbow":
        return RainbowAgent1P(device="cpu", batch_size=16)
    return DVNAgent1P(policy_net=BlockBlastValueNet1PmultikernelFlattenned, device="cpu",
                      batch_size=16, punish_for_invalid=-100.0)


@pytest.mark.parametrize("kind", ["ddqn", "rainbow", "dvn"])
def test_agent_learns_without_error(kind):
    agent = make_agent(kind)
    env = BlockBlastEnv(punish_for_invalid=-100.0)
    obs, _ = env.reset(seed=0)
    losses = []
    for _ in range(80):
        action = agent.select_action(obs, 0.5)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.store_transition(obs, action, reward, next_obs, terminated or truncated)
        loss = agent.update_model()
        if loss is not None:
            losses.append(loss)
        obs = next_obs if not (terminated or truncated) else env.reset()[0]
    assert losses and np.all(np.isfinite(losses))


@pytest.mark.parametrize("kind", ["ddqn", "rainbow", "dvn"])
def test_save_load_roundtrip(kind, tmp_path):
    src, dst = make_agent(kind), make_agent(kind)
    src.save_model(tmp_path / "model.pt")
    dst.load_model(tmp_path / "model.pt")
    for a, b in zip(src.policy_net.state_dict().values(), dst.policy_net.state_dict().values()):
        assert torch.equal(a, b)


@pytest.mark.parametrize("path, net", [
    (DVN_WEIGHTS, BlockBlastValueNet1PmultikernelFlattenned),
    (DVN_CNN_WEIGHTS, BlockBlastValueNet1P),
])
def test_final_weights_load_on_cpu(path, net):
    """The weights were saved on GPU; load_dvn_agent picks the right network."""
    agent = load_dvn_agent(path, device="cpu")
    assert isinstance(agent.policy_net, net)
    obs, _ = BlockBlastEnv().reset(seed=0)
    assert obs["valid_placements"].reshape(-1)[agent.select_action(obs, 0.0)]


def test_trained_dvn_beats_greedy(dvn_agent):
    """Coarse sanity check of the weights: on 20 seeded 1P games the trained DVN
    survives longer than the greedy baseline (report: ~60 vs ~30 placements)."""
    from common.evaluation import run_episodes
    from common.policies import greedy_1p

    def env():
        return BlockBlastEnv(punish_for_invalid=-100.0)

    dvn = run_episodes(env(), lambda o, e: dvn_agent.select_action(o, 0.0), 20, seed=0, max_steps=100, progress=False)
    greedy = run_episodes(env(), greedy_1p, 20, seed=0, max_steps=100, progress=False)
    assert dvn.lengths.mean() > 1.5 * greedy.lengths.mean()
