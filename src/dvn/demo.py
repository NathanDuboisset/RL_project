"""Record a GIF of one 3P game played by the DVN depth-2 lookahead.

    python -m dvn.demo --seed 10 --out assets/dvn_3p.gif
"""
import argparse
from pathlib import Path

from blockblast import BlockBlast3PEnv
from blockblast.rendering import record_episode, save_gif
from dvn.agent import load_dvn_agent
from dvn.lookahead import DVNLookahead3P


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="final_weights/dvn_final_20260313_020137.pt")
    parser.add_argument("--depth", type=int, choices=(2, 3), default=2)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--ms-per-frame", type=int, default=150)
    parser.add_argument("--out", default="assets/dvn_3p.gif")
    args = parser.parse_args()

    agent = load_dvn_agent(args.checkpoint, device="cpu")
    frames = record_episode(BlockBlast3PEnv(), DVNLookahead3P(agent, depth=args.depth), seed=args.seed,
                            max_steps=args.max_steps, caption=f"DVN depth-{args.depth} - seed {args.seed}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, args.out, ms_per_frame=args.ms_per_frame)
    print(f"{len(frames)} frames -> {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
