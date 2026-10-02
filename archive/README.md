# Archive

Code and files that are **not used in the final report** (`report/report.pdf`),
kept for reference. Nothing here is imported by `src/`, and this code is not
maintained: imports may be out of date (e.g. `from ppo...` modules that moved).

| Path | What it is |
|---|---|
| `src/ppo/bc_trainer.py` | Behaviour cloning of the PPO policy on MCTS decisions (experiment) |
| `src/ppo/mcts_collect.py` | Collects a dataset of MCTS decisions for the above |
| `src/ppo/mcts_ppo_trainer.py` | PPO trained with MCTS-guided rollouts (experiment) |
| `src/ppo/ppo_finetune.py` | PPO fine-tuning on the MCTS dataset (experiment) |
| `src/dqn/models_3p.py` | `BlockBlastCNNNet`, DQN network for the 3-piece env (the report only covers DQN in 1P) |
| `notebooks/ddqn_3p.ipynb` | DDQN on the 3-piece env; uses a `DDQNAgent` class that no longer exists |
| `plots/` | Intermediate benchmark plots not included in the report |
| `report_2026-03-16_old.pdf` | Earlier version of the report |

The `src/ppo/*` experiments are still called from some cells of
`notebooks/mct.ipynb`; those cells will fail unless the files are copied back.
