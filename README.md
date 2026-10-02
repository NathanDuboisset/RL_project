# Reinforcement Learning for Block Blast

DQN, Deep Value Network (DVN) and PPO agents for the puzzle game Block Blast
(8×8 grid). The report is in [`report/report.pdf`](report/report.pdf).

## Setup

```bash
uv sync                      # creates .venv and installs the project (editable)
python tests/smoke_test.py   # every method runs a few steps on CPU (~1 min)
```

The packages under `src/` (`blockblast`, `common`, `dqn`, `dvn`, `ppo`) are then
importable from anywhere, e.g. `from dvn.agent import load_dvn_agent`.

## Where each part of the report lives

| Report | Code |
|---|---|
| §2 Environments 1P / 3P, reward shaping | `src/blockblast/block_blast_env.py`, `src/blockblast/block_blast_3p_env.py` |
| §2.3 `get_t_plus_3_candidates` (all 3-placement sequences of a round) | `BlockBlast3PEnv.iter_t_plus_3_sequences` / `get_t_plus_3_candidates` |
| §3 DDQN, Rainbow (PER, n-step, noisy, dueling, C51), `BlockBlastCNNNet1P` | `src/dqn/agent.py`, `src/dqn/models.py`, `src/dqn/train.py`, `notebooks/ddqn 1P.ipynb`, `notebooks/rainbow_1P.ipynb` |
| §4.1–4.3 DVN on afterstates, multi-kernel CNN, training | `src/dvn/agent.py`, `src/dvn/models.py`, `src/dvn/train.py` |
| §4.4 1P benchmark (DVN / greedy / random) | `src/dvn/benchmark_1p.py` |
| §4.5 `RoundPlanner3P` | `src/dvn/planner.py` |
| §4.5 Table 2, Fig. 3–5: DVN depth-2/3, greedy depth-2/3 | `src/dvn/lookahead.py`, `src/common/policies.py`, `src/dvn/benchmark_3p.py`; the reported run is `notebooks/benchmark_3p_clean.ipynb` |
| §5.1–5.2 PPO (Board/Pieces/Scalar encoders, symlog value) | `src/ppo/ppo_agent.py`, `src/ppo/train.py`, `notebooks/mct.ipynb` |
| §5.3 MCTS First Only / Full Triplet, Fig. 6 (value weight α) | `src/ppo/mcts_agent.py`, `src/ppo/value_weight_sweep.py` |
| §5.4 Fig. 8 (PPO greedy vs MCTS) | `src/ppo/benchmark_3p.py` |
| Shared: base agent, evaluation loop, baselines, plots | `src/common/` |

Experiments not used in the report were removed from the tree; they are in the git history (tag `pre-refactor`).

## Running the benchmarks

```bash
python -m dvn.benchmark_1p --checkpoint final_weights/dvn_final_20260313_020137.pt
python -m dvn.benchmark_3p --checkpoint final_weights/dvn_final_20260313_020137.pt \
    --policies dvn-d2 greedy-d2 random --episodes 250
python -m dvn.benchmark_3p --checkpoint final_weights/dvn_final_20260313_020137.pt \
    --policies dvn-d3 greedy-d3 random --episodes 25      # slow
```

Episode `i` is reset with seed `seed + i`; the pieces only depend on the seed, so
all policies are compared on the same piece sequences. Plots and raw returns
(`.npz`) are written to `plots/`.

The PPO benchmarks need a trained PPO checkpoint, which is **not versioned**
(put it in `checkpoints/`, ignored by git):

```python
from blockblast import BlockBlast3PEnv
from ppo.ppo_agent import PPOTrainer
from ppo.benchmark_3p import run_benchmark

trainer = PPOTrainer([BlockBlast3PEnv()])
trainer.load("checkpoints/ppo/<checkpoint>.pt")
run_benchmark(trainer.model, lambda: BlockBlast3PEnv(), n_episodes=100, value_weight=0.3)
```

## Weights

| File | Network | Used for |
|---|---|---|
| `final_weights/dvn_final_20260313_020137.pt` | `BlockBlastValueNet1PmultikernelFlattenned` (54,529 parameters) | 3P benchmarks (Table 2, Fig. 3–5) |
| `final_weights/dvn_1P_60avg.pt` | `BlockBlastValueNet1P`, plain CNN (1,345,121 parameters) | probably the 1P result of Table 1 (to be confirmed) |

`load_dvn_agent(path)` picks the right network from the weights.

## Known differences between the report and the code

- §4.2 gives the DVN as 42,461 parameters with filters (1, 8, 16, 16, 16, 64).
  42,461 is the size of the older `BlockBlastValueNet1Pmultikernel`; the trained
  multi-kernel network has filters (1, 6, 8, 8, 16, 32) and 54,529 parameters.
- §4.3 says the DVN target network is updated every 200 episodes with batch
  size 128; the current `dvn/train.py` updates it every 800 environment steps
  (`target_update_freq=800`, one gradient step every 4 environment steps) with
  batch size 512. It is unknown which configuration produced the final weights.
- §4.5 describes `RoundPlanner3P` (commits to the whole round), but Table 2 was
  produced with the DVN depth-k lookahead of `dvn/lookahead.py`, which replans at
  every step. The depth-2 and depth-3 versions also score end-of-round and
  dead-end positions differently (see the module docstring).
- Despite the name, the "MCTS" planners of §5.3 do an exhaustive search, not a
  Monte Carlo Tree Search.
