"""Benchmark DVN vs greedy vs random on BlockBlast 1P (report, Section 4.4).

    python -m dvn.benchmark_1p --checkpoint final_weights/dvn_final_20260313_020137.pt
"""
import argparse
from datetime import datetime
from pathlib import Path

import torch

from blockblast import BlockBlastEnv
from common.evaluation import print_table, run_episodes, save_npz
from common.plotting import plot_distributions
from common.policies import RandomPolicy, greedy_1p
from dvn.agent import load_dvn_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to DVN checkpoint (.pt).")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum number of steps per episode.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123, help="Episode i uses seed `seed + i` (same pieces for every policy).")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory where plots and raw data are saved.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = load_dvn_agent(args.checkpoint, device=args.device)

    policies = {
        "DVN": lambda obs, env: agent.select_action(obs, epsilon=0.0),
        "Greedy": greedy_1p,
        "Random": RandomPolicy(args.seed),
    }
    results = {
        name: run_episodes(BlockBlastEnv(punish_for_invalid=-100), policy, args.episodes,
                           seed=args.seed, max_steps=args.max_steps, desc=name)
        for name, policy in policies.items()
    }
    print_table(results, f"BlockBlast 1P — {Path(args.checkpoint).name}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"benchmark_1p_{datetime.now():%Y%m%d_%H%M%S}"
    plot_distributions(results, stem.with_suffix(".png"), "BlockBlast 1P")
    save_npz(stem.with_suffix(".npz"), results, seed=args.seed, max_steps=args.max_steps)
    print(f"\nSaved {stem}.png / .npz")


if __name__ == "__main__":
    main()
