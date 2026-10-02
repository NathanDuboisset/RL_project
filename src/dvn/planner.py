import numpy as np
import torch

from blockblast import BlockBlast3PEnv
from dvn.agent import DVNAgent1P


class RoundPlanner3P:
    """3P planner of the report (Section 4.5): at the start of each round,
    enumerate every valid 3-placement sequence, score it with

        score = r_t + gamma * r_t+1 + gamma^2 * r_t+2 + gamma^3 * V(s_t+3)

    and commit to the best sequence; replan if a queued action became invalid.
    Leaf boards are evaluated by batches of `eval_batch_size`.

    When no full 3-placement sequence exists (usually right before game over),
    the planner plays a random valid action drawn from `seed`'s generator.
    """

    def __init__(self, gamma: float, agent: DVNAgent1P, eval_batch_size: int = 4096, seed: int | None = None) -> None:
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.agent = agent
        self.plan_actions: list[int] = []
        self.eval_batch_size = max(1, int(eval_batch_size))

    def reset(self) -> None:
        self.plan_actions = []

    reset_round_plan = reset  # backward-compatible name

    def _fallback_random_action(self, env: BlockBlast3PEnv) -> list[int] | None:
        valid_placements = env.valid_placements
        if valid_placements is None:
            return None
        valid_actions = np.flatnonzero(valid_placements.reshape(-1))
        if valid_actions.size == 0:
            return None
        return [int(self.rng.choice(valid_actions))]

    def _build_new_round_plan(self, env: BlockBlast3PEnv) -> list[int] | None:
        if env.pieces_used is None or int(np.sum(env.pieces_used == 0)) < 3:
            # Mid-round (a queued action became invalid): no full sequence to plan.
            return self._fallback_random_action(env)

        gamma3 = self.gamma * self.gamma * self.gamma  # same float ops as the original code
        best_score = -np.inf
        best_actions = None
        batch_boards, batch_cum3, batch_actions = [], [], []

        def flush_batch() -> None:
            nonlocal best_score, best_actions
            if not batch_boards:
                return
            with torch.inference_mode():
                x = torch.from_numpy(np.asarray(batch_boards, dtype=np.float32)).to(self.agent.device)
                v = self.agent.policy_net(x).squeeze(-1).detach().cpu().numpy().astype(np.float32, copy=False)
            scores = np.asarray(batch_cum3, dtype=np.float32) + gamma3 * v
            local_best = int(np.argmax(scores))
            if float(scores[local_best]) > best_score:
                best_score = float(scores[local_best])
                best_actions = batch_actions[local_best]
            batch_boards.clear()
            batch_cum3.clear()
            batch_actions.clear()

        for actions, cum3, board3 in env.iter_t_plus_3_sequences(self.gamma):
            batch_boards.append(board3)
            batch_cum3.append(cum3)
            batch_actions.append(actions)
            if len(batch_boards) >= self.eval_batch_size:
                flush_batch()
        flush_batch()

        if best_actions is None:
            return self._fallback_random_action(env)
        return [env.encode_action(p, r, c) for (p, r, c) in best_actions]

    def select_action(self, env: BlockBlast3PEnv) -> int | None:
        if not self.plan_actions:
            plan = self._build_new_round_plan(env)
            if plan is None:
                return None
            self.plan_actions = plan

        action = self.plan_actions.pop(0)

        valid_placements = env.valid_placements
        if valid_placements is None or not valid_placements.reshape(-1)[action]:
            plan = self._build_new_round_plan(env)
            if plan is None:
                return None
            self.plan_actions = plan
            action = self.plan_actions.pop(0)

        return action

    def __call__(self, obs, env) -> int | None:
        return self.select_action(env)
