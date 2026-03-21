import os
import time
from dataclasses import dataclass, field
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from blockblast import BlockBlast3PEnv
from mct.ppo_agent import PPOTrainer, obs_to_tensors, valid_to_mask


@dataclass
class TrainConfig:
    steps:            int   = 50_000
    n_envs:           int   = 4
    n_steps:          int   = 128
    epochs:           int   = 4
    batch:            int   = 256
    lr:               float = 3e-4
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_eps:         float = 0.2
    vf_coef:          float = 0.5
    ent_coef:         float = 0.01
    grad_norm:        float = 0.5
    device:           str   = "cpu"
    save:             str   = "ppo_blockblast.pt"
    load:             str   = None
    log_interval:     int   = 10
    eval_eps:         int   = 20
    plot:             str   = "training_curves.png"
    checkpoint_every: int   = 0
    checkpoint_dir:   str   = "/Data/roman.lendormy/rl_checkpoints"


def make_envs(n: int):
    return [BlockBlast3PEnv() for _ in range(n)]


def smooth(x, w: int = 5):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_history(history: dict, save_path: str = "training_curves.png", title_suffix: str = ""):
    steps = np.array(history["steps"])
    if len(steps) == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("PPO -- BlockBlast3P" + title_suffix, fontsize=15, fontweight="bold")

    panels = [
        (axes[0, 0], "Mean Episode Return",  "mean_return",  "steelblue"),
        (axes[0, 1], "Median Episode Return", "median_return","darkorange"),
        (axes[0, 2], "Mean Episode Length",   "mean_length",  "seagreen"),
        (axes[1, 0], "Policy Loss",           "loss_policy",  "crimson"),
        (axes[1, 1], "Value Loss",            "loss_value",   "purple"),
        (axes[1, 2], "Entropy",               "entropy",      "saddlebrown"),
    ]

    for ax, label, key, color in panels:
        y = np.array(history[key])
        ax.plot(steps, y, alpha=0.25, color=color, linewidth=1)
        s = smooth(y)
        ax.plot(steps[len(steps) - len(s):], s, color=color, linewidth=2)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Environment steps")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"  Plot saved -> {save_path}")
    plt.close(fig)


def evaluate(trainer: PPOTrainer, n_episodes: int = 20) -> dict:
    env = BlockBlast3PEnv()
    returns, lengths = [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_r = 0.0
        n_steps = 0
        while True:
            batch  = {k: v[None] for k, v in obs.items()}
            obs_t  = obs_to_tensors(batch, trainer.device)
            mask_t = torch.as_tensor(
                valid_to_mask(batch["valid_placements"]),
                device=trainer.device,
            )
            actions, *_ = trainer.model.get_action(obs_t, mask_t, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(actions[0]))
            total_r += r
            n_steps += 1
            if term or trunc:
                break
        returns.append(total_r)
        lengths.append(n_steps)

    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return":  float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
    }


class CheckpointCallback:
    def __init__(self, cfg: TrainConfig):
        self.cfg             = cfg
        self.every           = cfg.checkpoint_every
        self.directory       = cfg.checkpoint_dir
        self._last_ckpt_step = 0

        if self.every > 0:
            os.makedirs(self.directory, exist_ok=True)
            print(f"Checkpointing every {self.every:,} steps -> {self.directory}/")

    def maybe_checkpoint(self, trainer: PPOTrainer, history: dict):
        if self.every <= 0:
            return

        current = trainer.total_steps
        if current - self._last_ckpt_step >= self.every:
            self._last_ckpt_step = (current // self.every) * self.every
            self._save(trainer, history, current)

    def _save(self, trainer: PPOTrainer, history: dict, step: int):
        tag  = f"{step // 1000}k"
        stem = os.path.join(self.directory, f"ckpt_{tag}")

        ckpt_path = stem + ".pt"
        trainer.save(ckpt_path)

        plot_path = stem + "_curves.png"
        plot_history(history, save_path=plot_path, title_suffix=f" @ {tag} steps")

        print(f"[checkpoint @ {step:,}]  model -> {ckpt_path}  |  plot -> {plot_path}")


def run(cfg: TrainConfig = None) -> tuple:
    if cfg is None:
        cfg = TrainConfig()

    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        cfg.device = "cpu"
    if cfg.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU.")
        cfg.device = "cpu"
    device = torch.device(cfg.device)
    print(f"Device : {device}")

    envs    = make_envs(cfg.n_envs)
    ckpt_cb = CheckpointCallback(cfg)

    trainer = PPOTrainer(
        envs          = envs,
        device        = device,
        lr            = cfg.lr,
        n_steps       = cfg.n_steps,
        n_epochs      = cfg.epochs,
        batch_size    = cfg.batch,
        gamma         = cfg.gamma,
        gae_lambda    = cfg.gae_lambda,
        clip_eps      = cfg.clip_eps,
        vf_coef       = cfg.vf_coef,
        ent_coef      = cfg.ent_coef,
        max_grad_norm = cfg.grad_norm,
    )

    if cfg.load:
        trainer.load(cfg.load)

    n_updates     = cfg.steps // (cfg.n_steps * cfg.n_envs)
    steps_per_upd = cfg.n_steps * cfg.n_envs

    trainer._reset_all_envs()

    hdr = (
        f"{'Upd':>6} | {'Steps':>10} | {'MeanRet':>9} | {'MedRet':>9} | "
        f"{'MeanLen':>8} | {'PgLoss':>8} | {'VfLoss':>8} | {'Entropy':>8} | {'ClipFr':>7}"
    )
    print(f"\nPPO -- {cfg.steps:,} steps  {n_updates} updates  {cfg.n_envs} envs\n")
    print(hdr)
    print("-" * len(hdr))

    t0 = time.time()

    for upd in range(1, n_updates + 1):
        trainer._collect_rollout()
        m = trainer._ppo_update()

        if upd % cfg.log_interval == 0 or upd == 1:
            mr  = float(np.mean(trainer.ep_returns))   if trainer.ep_returns else 0.0
            mdr = float(np.median(trainer.ep_returns))  if trainer.ep_returns else 0.0
            ml  = float(np.mean(trainer.ep_lengths))   if trainer.ep_lengths else 0.0

            trainer.history["steps"].append(trainer.total_steps)
            trainer.history["mean_return"].append(mr)
            trainer.history["median_return"].append(mdr)
            trainer.history["mean_length"].append(ml)
            trainer.history["loss_policy"].append(m["loss_policy"])
            trainer.history["loss_value"].append(m["loss_value"])
            trainer.history["entropy"].append(m["entropy"])
            trainer.history["clip_frac"].append(m["clip_frac"])

            print(
                f"{upd:>6} | {trainer.total_steps:>10,} | {mr:>9.2f} | {mdr:>9.2f} | "
                f"{ml:>8.1f} | {m['loss_policy']:>8.4f} | {m['loss_value']:>8.4f} | "
                f"{m['entropy']:>8.4f} | {m['clip_frac']:>7.3f}"
            )

            ckpt_cb.maybe_checkpoint(trainer, trainer.history)

    history = trainer.history
    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed:.1f}s  ({trainer.total_steps / elapsed:,.0f} steps/s)")

    if cfg.save:
        trainer.save(cfg.save)

    if history["steps"] and cfg.plot:
        plot_history(history, save_path=cfg.plot)

    if cfg.eval_eps > 0:
        print(f"\nEvaluating greedy policy over {cfg.eval_eps} episodes ...")
        stats = evaluate(trainer, n_episodes=cfg.eval_eps)
        print(f"  Mean return : {stats['mean_return']:.2f} +/- {stats['std_return']:.2f}")
        print(f"  Mean length : {stats['mean_length']:.1f} steps")

    for env in envs:
        env.close()

    return trainer, history


def _parse_cli() -> TrainConfig:
    import argparse
    p = argparse.ArgumentParser(description="PPO training for BlockBlast3P")
    p.add_argument("--steps",             type=int,   default=50_000)
    p.add_argument("--n_envs",            type=int,   default=4)
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--n_steps",           type=int,   default=128)
    p.add_argument("--epochs",            type=int,   default=4)
    p.add_argument("--batch",             type=int,   default=256)
    p.add_argument("--gamma",             type=float, default=0.99)
    p.add_argument("--gae_lambda",        type=float, default=0.95)
    p.add_argument("--clip_eps",          type=float, default=0.2)
    p.add_argument("--vf_coef",           type=float, default=0.5)
    p.add_argument("--ent_coef",          type=float, default=0.01)
    p.add_argument("--grad_norm",         type=float, default=0.5)
    p.add_argument("--device",            type=str,   default="cpu")
    p.add_argument("--save",              type=str,   default="ppo_blockblast.pt")
    p.add_argument("--load",              type=str,   default=None)
    p.add_argument("--log_interval",      type=int,   default=10)
    p.add_argument("--eval_eps",          type=int,   default=20)
    p.add_argument("--plot",              type=str,   default="training_curves.png")
    p.add_argument("--checkpoint_every",  type=int,   default=0)
    p.add_argument("--checkpoint_dir",    type=str,   default="checkpoints")
    a = p.parse_args()
    return TrainConfig(**vars(a))


if __name__ == "__main__":
    run(_parse_cli())
