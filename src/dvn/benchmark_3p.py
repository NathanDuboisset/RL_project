"""Benchmark the DVN-based planners on BlockBlast 3P (report, Section 4.5, Table 2).

Policies (--policies):
  dvn-d2, dvn-d3   DVN lookahead over the next 2 / 3 placements, replanning every step
  round-planner    RoundPlanner3P: best full 3-placement sequence, committed for the round
  greedy-d1/d2/d3  exhaustive search maximising the sum of immediate rewards
  random           uniformly random valid action

Report setup: dvn-d2, greedy-d2, random with 250 episodes; dvn-d3 with 25.
    python -m dvn.benchmark_3p --checkpoint final_weights/dvn_final_20260313_020137.pt \\
        --policies dvn-d2 greedy-d2 random --episodes 250
Note: notebooks/benchmark_3p_clean.ipynb (the run used in the report) gives each
policy different seeds; here all policies share the same seeds (paired comparison).
"""
import argparse
from datetime import datetime
from pathlib import Path

import torch

from blockblast import BlockBlast3PEnv
from common.evaluation import print_table, run_episodes, save_npz
from common.plotting import plot_distributions
from common.policies import GreedyLookahead3P, RandomPolicy
from dvn.agent import load_dvn_agent
from dvn.lookahead import DVNLookahead3P
from dvn.planner import RoundPlanner3P

POLICIES = ["dvn-d2", "dvn-d3", "round-planner", "greedy-d1", "greedy-d2", "greedy-d3", "random"]


def make_policy(name: str, agent, gamma: float, seed: int):
    if name.startswith("dvn-d"):
        return DVNLookahead3P(agent, depth=int(name[-1]), gamma=gamma)
    if name == "round-planner":
        return RoundPlanner3P(gamma=gamma, agent=agent, seed=seed)
    if name.startswith("greedy-d"):
        return GreedyLookahead3P(depth=int(name[-1]))
    if name == "random":
        return RandomPolicy(seed)
    raise ValueError(f"Unknown policy: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to DVN checkpoint (.pt).")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=["dvn-d2", "greedy-d2", "random"])
    parser.add_argument("--episodes", type=int, default=250, help="Number of evaluation episodes per policy.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum number of steps per episode.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123, help="Episode i uses seed `seed + i` (same pieces for every policy).")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor used by the planners.")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory where plots and raw data are saved.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = load_dvn_agent(args.checkpoint, device=args.device)

    results = {}
    for name in args.policies:
        policy = make_policy(name, agent, args.gamma, args.seed)
        results[name] = run_episodes(BlockBlast3PEnv(lookahead_gamma=args.gamma), policy, args.episodes,
                                     seed=args.seed, max_steps=args.max_steps, desc=name)
    print_table(results, f"BlockBlast 3P — {Path(args.checkpoint).name}, gamma={args.gamma}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"benchmark_3p_{datetime.now():%Y%m%d_%H%M%S}"
    plot_distributions(results, stem.with_suffix(".png"), "BlockBlast 3P")
    save_npz(stem.with_suffix(".npz"), results, seed=args.seed, gamma=args.gamma, max_steps=args.max_steps)
    print(f"\nSaved {stem}.png / .npz")


if __name__ == "__main__":
    main()
