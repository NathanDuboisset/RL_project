"""Train a DDQN or Rainbow agent on BlockBlast 1P (report, Section 3).

    python -m dqn.train --agent ddqn --episodes 1000
    python -m dqn.train --agent rainbow --episodes 1000

Checkpoints and the per-episode returns (CSV) go to <output-dir>/<run-name>/.
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from blockblast import BlockBlastEnv
from common.utils import default_device, set_seed
from dqn.agent import DDQNAgent1P, RainbowAgent1P

AGENTS = {"ddqn": DDQNAgent1P, "rainbow": RainbowAgent1P}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=AGENTS, default="ddqn")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-update-freq", type=int, default=10, help="Target network update period, in episodes.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon (ignored by Rainbow, which uses noisy nets).")
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995, help="Multiplicative decay per episode.")
    parser.add_argument("--save-every", type=int, default=None, help="Checkpoint period in episodes (default: episodes // 3).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--run-name", default=None, help="Default: <agent>_seed<seed>.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_name = args.run_name or f"{args.agent}_seed{args.seed}"
    out = Path(args.output_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)

    env = BlockBlastEnv()
    agent = AGENTS[args.agent](action_size=64, lr=args.lr, batch_size=args.batch_size, device=args.device)
    n_params = sum(p.numel() for p in agent.policy_net.parameters())
    print(f"{args.agent}: {n_params:,} parameters, device={args.device}, output={out}")

    save_every = args.save_every or max(1, args.episodes // 3)
    epsilon = args.epsilon
    returns, lengths = [], []

    env.reset(seed=args.seed)  # seeds the environment RNG once for the whole run
    for episode in (bar := tqdm(range(args.episodes))):
        state, _ = env.reset()
        total, steps, done = 0.0, 0, False
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            agent.update_model()
            state = next_state
            total += reward
            steps += 1

        if episode % args.target_update_freq == 0:
            agent.update_target_model()
        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
        returns.append(total)
        lengths.append(steps)
        bar.set_postfix(ret100=f"{np.mean(returns[-100:]):.1f}", len100=f"{np.mean(lengths[-100:]):.1f}")

        if episode > 0 and episode % save_every == 0:
            agent.save_model(out / f"{run_name}_ep{episode}.pt")

    agent.save_model(out / f"{run_name}_final.pt")
    np.savetxt(out / "returns.csv", np.column_stack([np.arange(args.episodes), returns, lengths]),
               delimiter=",", header="episode,return,length", comments="", fmt=["%d", "%.4f", "%d"])
    print(f"Done. Final model and returns.csv saved in {out}")


if __name__ == "__main__":
    main()
