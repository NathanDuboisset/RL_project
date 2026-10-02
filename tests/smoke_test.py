"""Smoke test: each method of the report imports, runs a few steps on CPU,
and the parameter counts are printed next to the values given in the report.

Usage (from the repo root):  python tests/smoke_test.py
This checks that the code runs, NOT that training reproduces the report.
"""
import sys, traceback
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # inutile si le projet est installé (uv sync)
torch.manual_seed(0); np.random.seed(0)
dev = torch.device("cpu")
nparams = lambda m: sum(p.numel() for p in m.parameters())
results = {}

def check(name):
    def deco(f):
        try:
            results[name] = "OK  " + (f() or "")
        except Exception as e:
            results[name] = f"FAIL {type(e).__name__}: {e}"
            traceback.print_exc()
    return deco

from blockblast import BlockBlastEnv, BlockBlast3PEnv

def run_1p(agent, n=200):
    env = BlockBlastEnv(); obs, _ = env.reset(seed=0); steps = 0
    for _ in range(n):
        a = agent.select_action(obs, 0.5)
        nobs, r, term, trunc, _ = env.step(a)
        agent.store_transition(obs, a, r, nobs, term or trunc)
        agent.update_model(); steps += 1
        obs = nobs
        if term or trunc: obs, _ = env.reset()
    return steps

@check("DDQN 1P")
def _():
    from dqn.agent import DDQNAgent1P
    ag = DDQNAgent1P(device=dev, batch_size=16)
    run_1p(ag); return f"params={nparams(ag.policy_net):,} (rapport: 2,674,464)"

@check("Rainbow 1P")
def _():
    from dqn.agent import RainbowAgent1P
    ag = RainbowAgent1P(device=dev, batch_size=16)
    run_1p(ag); return f"params={nparams(ag.policy_net):,}"

@check("DVN 1P (train + poids finaux)")
def _():
    from dvn.agent import DVNAgent1P
    from dvn.models import BlockBlastValueNet1PmultikernelFlattenned as Net
    ag = DVNAgent1P(policy_net=Net, device=dev, batch_size=16, punish_for_invalid=-100.0)
    run_1p(ag)
    from dvn.models import BlockBlastValueNet1P
    out = []
    # dvn_final = multi-kernel (utilisé dans les benchmarks 3P) ; dvn_1P_60avg = CNN simple
    for w, cls in [("dvn_final_20260313_020137.pt", Net), ("dvn_1P_60avg.pt", BlockBlastValueNet1P)]:
        a = DVNAgent1P(policy_net=cls, device=dev)
        a.load_model(str(ROOT / "final_weights" / w)); out.append(f"{w}: chargé")
    return f"params={nparams(ag.policy_net):,} (rapport: 42,461) | " + "; ".join(out)

@check("DVN 3P RoundPlanner3P (1 épisode court)")
def _():
    from dvn.agent import DVNAgent1P
    from dvn.planner import RoundPlanner3P
    from dvn.models import BlockBlastValueNet1PmultikernelFlattenned as Net
    ag = DVNAgent1P(policy_net=Net, device=dev)
    pl = RoundPlanner3P(gamma=0.99, agent=ag)
    env = BlockBlast3PEnv(); env.reset(seed=0); steps = 0
    for _ in range(9):
        a = pl.select_action(env)
        if a is None: break
        _, _, term, trunc, _ = env.step(a); steps += 1
        if term or trunc: break
    return f"{steps} pas joués"

@check("PPO 3P (1 rollout + 1 update)")
def _():
    from ppo.ppo_agent import PPOTrainer
    tr = PPOTrainer([BlockBlast3PEnv() for _ in range(2)], device=dev, n_steps=16, batch_size=16, n_epochs=1)
    tr._reset_all_envs(); tr._collect_rollout(); tr._ppo_update()
    return f"params={nparams(tr.model):,} (rapport: ~848k)"

for cls_name, mod in [("MCTSAgent", "ppo.mcts_agent"), ("MCTSAgentFirstOnly", "ppo.mcts_agent")]:
    @check(f"PPO + {cls_name} (5 pas)")
    def _(cls_name=cls_name, mod=mod):
        import importlib
        from ppo.ppo_agent import ActorCritic
        cls = getattr(importlib.import_module(mod), cls_name)
        kw = {"value_weight": 0.3} if cls_name == "MCTSAgent" else {}
        ag = cls(ActorCritic().to(dev), device=dev, **kw)
        env = BlockBlast3PEnv(); env.reset(seed=0); steps = 0
        for _ in range(5):
            a = ag.select_action(env)
            _, _, term, trunc, _ = env.step(a); steps += 1
            if term or trunc: break
        return f"{steps} pas joués"

import subprocess, tempfile, os
for script, extra in [("dvn.benchmark_1p", ["--episodes", "3", "--max-steps", "10"]),
                      ("dvn.benchmark_3p", ["--episodes", "1", "--max-steps", "6", "--policies",
                                            "dvn-d2", "round-planner", "greedy-d2", "random"])]:
    @check(f"script {script}")
    def _(script=script, extra=extra):
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
            cmd = [sys.executable, "-m", script, "--device", "cpu", "--output-dir", tmp,
                   "--checkpoint", str(ROOT / "final_weights/dvn_final_20260313_020137.pt"), *extra]
            subprocess.run(cmd, check=True, capture_output=True, env=env)
            return f"{len(os.listdir(tmp))} fichiers produits (png + npz)"

print("\n================ RÉSUMÉ")
for k, v in results.items(): print(f"{k:45s} {v}")
sys.exit(any(v.startswith("FAIL") for v in results.values()))
