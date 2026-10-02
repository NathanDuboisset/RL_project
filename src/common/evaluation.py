"""Evaluation loop and statistics shared by all benchmarks.

A *policy* is any callable `policy(obs, env) -> action | None`
(None = the policy has no move, the episode stops). If the policy has a
`reset()` method, it is called at the start of every episode (used by the
planners that keep a queue of actions for the current round).
"""
import time
from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm


@dataclass
class EpisodeStats:
    returns: np.ndarray
    lengths: np.ndarray
    step_rewards: np.ndarray
    decision_ms: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> dict:
        return {
            "return": describe(self.returns),
            "length": describe(self.lengths),
            "ms_per_decision": float(self.decision_ms.mean()) if self.decision_ms.size else 0.0,
        }


def run_episodes(
    env,
    policy,
    n_episodes: int,
    seed: int = 0,
    max_steps: int | None = None,
    desc: str | None = None,
    progress: bool = True,
) -> EpisodeStats:
    """Play `n_episodes` with `policy`; episode i is reset with seed `seed + i`.

    Piece sequences only depend on the seed, so two policies evaluated with
    the same `seed` see the same pieces (paired comparison).
    """
    returns, lengths, step_rewards, decision_ms = [], [], [], []

    for ep in tqdm(range(n_episodes), desc=desc, disable=not progress):
        obs, _ = env.reset(seed=seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        total, n_steps = 0.0, 0

        while max_steps is None or n_steps < max_steps:
            t0 = time.perf_counter()
            action = policy(obs, env)
            decision_ms.append((time.perf_counter() - t0) * 1000)
            if action is None:
                break

            obs, reward, terminated, truncated, _ = env.step(int(action))
            total += float(reward)
            step_rewards.append(float(reward))
            n_steps += 1
            if terminated or truncated:
                break

        returns.append(total)
        lengths.append(n_steps)

    return EpisodeStats(
        returns=np.asarray(returns, dtype=np.float64),
        lengths=np.asarray(lengths, dtype=np.int64),
        step_rewards=np.asarray(step_rewards, dtype=np.float64),
        decision_ms=np.asarray(decision_ms, dtype=np.float64),
    )


def describe(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    q5, q25, q50, q75, q95 = np.percentile(x, [5, 25, 50, 75, 95])
    mean, std = float(x.mean()), float(x.std())
    return {
        "n": int(x.size), "mean": mean, "std": std, "min": float(x.min()),
        "p5": float(q5), "p25": float(q25), "median": float(q50), "p75": float(q75),
        "p95": float(q95), "max": float(x.max()), "iqr": float(q75 - q25),
        # standard error of the mean: the mean of heavy-tailed returns is noisy
        "sem": std / np.sqrt(x.size) if x.size > 1 else float("nan"),
    }


def print_table(results: dict[str, EpisodeStats], title: str = "") -> None:
    if title:
        print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")
    header = f"{'policy':<22}{'n':>5}{'mean ret':>12}{'± sem':>10}{'median':>11}{'mean len':>10}{'ms/dec':>8}"
    print(header)
    print("-" * len(header))
    for name, st in results.items():
        s = st.summary()
        r, l = s["return"], s["length"]
        print(f"{name:<22}{r['n']:>5}{r['mean']:>12.1f}{r['sem']:>10.1f}{r['median']:>11.1f}"
              f"{l['mean']:>10.1f}{s['ms_per_decision']:>8.1f}")


def ecdf(x):
    xs = np.sort(np.asarray(x))
    return xs, np.arange(1, len(xs) + 1) / len(xs)


def save_npz(path, results: dict[str, EpisodeStats], **meta) -> None:
    arrays = {}
    for name, st in results.items():
        key = name.replace(" ", "_").replace("-", "_")
        arrays[f"{key}_returns"] = st.returns
        arrays[f"{key}_lengths"] = st.lengths
    np.savez_compressed(path, **arrays, **{k: np.asarray(v) for k, v in meta.items()})
