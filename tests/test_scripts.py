"""End-to-end runs of the command-line scripts with tiny budgets."""
import os
import subprocess
import sys

import pytest

from conftest import DVN_WEIGHTS, ROOT

pytestmark = pytest.mark.slow


def run_module(module, *args, cwd):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "MPLBACKEND": "Agg"}
    result = subprocess.run([sys.executable, "-m", module, *map(str, args)],
                            cwd=cwd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]


def test_benchmark_1p(tmp_path):
    run_module("dvn.benchmark_1p", "--checkpoint", DVN_WEIGHTS, "--episodes", 3, "--max-steps", 10,
               "--device", "cpu", "--output-dir", tmp_path, cwd=tmp_path)
    assert {p.suffix for p in tmp_path.iterdir()} == {".png", ".npz"}


def test_benchmark_3p(tmp_path):
    run_module("dvn.benchmark_3p", "--checkpoint", DVN_WEIGHTS, "--episodes", 1, "--max-steps", 6,
               "--policies", "dvn-d2", "round-planner", "greedy-d2", "random",
               "--device", "cpu", "--output-dir", tmp_path, cwd=tmp_path)
    assert {p.suffix for p in tmp_path.iterdir()} == {".png", ".npz"}


@pytest.mark.parametrize("agent", ["ddqn", "rainbow"])
def test_train_dqn(tmp_path, agent):
    run_module("dqn.train", "--agent", agent, "--episodes", 5, "--device", "cpu", "--output-dir", tmp_path, cwd=tmp_path)
    assert (tmp_path / f"{agent}_seed0" / "returns.csv").exists()


def test_train_dvn_and_resume(tmp_path):
    common = ["--batch-size", 16, "--checkpoint-freq", 5, "--device", "cpu", "--output-dir", tmp_path]
    run_module("dvn.train", "--episodes", 5, *common, cwd=tmp_path)
    run_module("dvn.train", "--episodes", 7, "--resume", tmp_path / "dvn_ep_5.pt", *common, cwd=tmp_path)
    episodes = (tmp_path / "log.csv").read_text().splitlines()[1:]
    assert [line.split(",")[0] for line in episodes] == [str(i) for i in range(1, 8)]


def test_train_ppo(tmp_path):
    run_module("ppo.train", "--steps", 512, "--n_envs", 2, "--device", "cpu", "--eval_eps", 1,
               "--save", tmp_path / "ppo.pt", "--plot", tmp_path / "curves.png", cwd=tmp_path)
    assert (tmp_path / "ppo.pt").exists()


def test_demo_gif(tmp_path):
    run_module("dvn.demo", "--checkpoint", DVN_WEIGHTS, "--max-steps", 3, "--out", tmp_path / "demo.gif", cwd=tmp_path)
    assert (tmp_path / "demo.gif").stat().st_size > 0
