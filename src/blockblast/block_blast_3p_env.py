import numpy as np
import itertools
from numpy.lib.stride_tricks import sliding_window_view
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

# base shapes, rotated randomly when sampled
SHAPES = {
    "dot":     np.array([[1]], dtype=np.int8),
    "2x2":     np.ones((2, 2), dtype=np.int8),
    "3x3":     np.ones((3, 3), dtype=np.int8),
    "line-3":  np.ones((1, 3), dtype=np.int8),
    "line-4":  np.ones((1, 4), dtype=np.int8),
    "L-small": np.array([[1, 0], [1, 1]], dtype=np.int8),
    "L-large": np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int8),
}

class BlockBlast3PEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi", "rgb_array"]}

    def __init__(
            self,
            render_mode=None,
            base_points=10,
            lookahead_gamma=0.99,
            shape_probs=None
    ):
        super().__init__()

        self.render_mode = render_mode
        self.base_points = base_points
        self.lookahead_gamma = lookahead_gamma
        self.grid_size = 8
        self.piece_box_size = 5
        self.n_pieces = 3

        self.action_space = spaces.Discrete(self.n_pieces * self.grid_size * self.grid_size)

        self.observation_space = spaces.Dict({
            "board":       spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.int8),
            "pieces":      spaces.Box(low=0, high=1, shape=(self.n_pieces, self.piece_box_size, self.piece_box_size), dtype=np.int8),
            "pieces_used": spaces.MultiBinary(self.n_pieces),
            "combo":       spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            "valid_placements": spaces.Box(low=0, high=1, shape=(self.n_pieces, self.grid_size, self.grid_size), dtype=np.int8),
            "placements_result": spaces.Box(
                low=0,
                high=1,
                shape=(self.n_pieces, self.grid_size, self.grid_size, self.grid_size, self.grid_size),
                dtype=np.int8,
            ),
        })

        self.shapes_keys = list(SHAPES.keys())

        if shape_probs is None:
            self.shape_probs = np.ones(len(self.shapes_keys)) / len(self.shapes_keys)
        else:
            assert len(shape_probs) == len(self.shapes_keys), "Probabilities must match number of shapes"
            probs = np.array(shape_probs)
            self.shape_probs = probs / np.sum(probs)

        self.fig = None
        self.board = None
        self.pieces_grids = None
        self.pieces_padded = None
        self.pieces_used = None
        self.combo = 0
        self.round_had_clear = False
        self.valid_placements = None
        self.placements_result = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.combo = 0
        self.round_had_clear = False
        self.pieces_used = np.zeros(self.n_pieces, dtype=np.int8)
        self._sample_new_pieces()
        self._update_all_valid_placements()
        self.placements_result = self._get_all_placements_result()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def step(self, action):
        piece_idx = action // (self.grid_size * self.grid_size)
        pos = action % (self.grid_size * self.grid_size)
        row = pos // self.grid_size
        col = pos % self.grid_size

        if not self.valid_placements[piece_idx, row, col]:
            return self._get_obs(), -1.0, True, False, self._get_info()

        self._place_piece(self.pieces_grids[piece_idx], row, col)
        self.pieces_used[piece_idx] = 1
        n = self._clear_lines()

        reward = 0.1
        if n > 0:
            reward = self.base_points * (n ** 2 + self.combo)
            self.combo += n
            self.round_had_clear = True

        if np.all(self.pieces_used):
            if not self.round_had_clear:
                self.combo = 0
            self.round_had_clear = False
            self.pieces_used = np.zeros(self.n_pieces, dtype=np.int8)
            self._sample_new_pieces()
            self._update_all_valid_placements()
            self.placements_result = self._get_all_placements_result()
            terminated = not np.any(self.valid_placements)
        else:
            self._update_all_valid_placements()
            self.placements_result = self._get_all_placements_result()
            terminated = not np.any(self.valid_placements)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_obs(self):
        return {
            "board":       self.board.copy(),
            "pieces":      self.pieces_padded.copy(),
            "pieces_used": self.pieces_used.copy(),
            "combo":       np.array([self.combo], dtype=np.int32),
            "valid_placements": self.valid_placements.copy(),
            "placements_result": self.placements_result.copy(),
        }

    def _get_info(self):
        return {}

    def _valid_positions_for_piece_on_board(self, board, piece_idx):
        shape_grid = self.pieces_grids[piece_idx]
        h, w = shape_grid.shape
        valid_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        windowed_board = sliding_window_view(board, window_shape=shape_grid.shape)
        overlaps = (windowed_board & shape_grid).any(axis=(-2, -1))
        valid_mask[: self.grid_size - h + 1, : self.grid_size - w + 1] = (~overlaps).astype(np.int8)
        return valid_mask

    def _simulate_one_hyp_step(self, board, combo, piece_idx, row, col):
        shape_grid = self.pieces_grids[piece_idx]
        h, w = shape_grid.shape

        next_board = board.copy()
        next_board[row : row + h, col : col + w] += shape_grid

        full_rows = np.all(next_board == 1, axis=1)
        full_cols = np.all(next_board == 1, axis=0)
        n_cleared = int(np.sum(full_rows) + np.sum(full_cols))
        next_board[full_rows, :] = 0
        next_board[:, full_cols] = 0

        reward = 0.1
        next_combo = combo
        if n_cleared > 0:
            reward = self.base_points * (n_cleared ** 2 + combo)
            next_combo = combo + n_cleared

        return next_board, float(reward), int(next_combo)

    def get_t_plus_3_candidates(self, gamma):
        """Enumerate all valid 3-step sequences and return discounted cumulative rewards."""
        if self.board is None or self.pieces_grids is None or self.pieces_used is None:
            return []

        available = [i for i in range(self.n_pieces) if not self.pieces_used[i]]
        if len(available) < 3:
            return []

        candidates = []
        board0 = self.board.copy()
        combo0 = int(self.combo)

        for order in itertools.permutations(available, 3):
            p0, p1, p2 = order

            valid0 = self._valid_positions_for_piece_on_board(board0, p0)
            rows0, cols0 = np.nonzero(valid0)
            for r0, c0 in zip(rows0.tolist(), cols0.tolist()):
                board1, r_t, combo1 = self._simulate_one_hyp_step(board0, combo0, p0, r0, c0)

                valid1 = self._valid_positions_for_piece_on_board(board1, p1)
                rows1, cols1 = np.nonzero(valid1)
                for r1, c1 in zip(rows1.tolist(), cols1.tolist()):
                    board2, r_t1, combo2 = self._simulate_one_hyp_step(board1, combo1, p1, r1, c1)

                    valid2 = self._valid_positions_for_piece_on_board(board2, p2)
                    rows2, cols2 = np.nonzero(valid2)
                    for r2, c2 in zip(rows2.tolist(), cols2.tolist()):
                        board3, r_t2, _ = self._simulate_one_hyp_step(board2, combo2, p2, r2, c2)
                        cum_reward = r_t + gamma * r_t1 + (gamma ** 2) * r_t2

                        candidates.append(
                            {
                                "state_t_plus_3": board3.copy(),
                                "cumulative_reward_3steps": float(cum_reward),
                                "order": order,
                                "actions": ((p0, r0, c0), (p1, r1, c1), (p2, r2, c2)),
                            }
                        )

        return candidates

    def _sample_new_pieces(self):
        self.pieces_grids = []
        self.pieces_padded = np.zeros((self.n_pieces, self.piece_box_size, self.piece_box_size), dtype=np.int8)

        for i in range(self.n_pieces):
            shape_idx = self.np_random.choice(len(self.shapes_keys), p=self.shape_probs)
            shape_name = self.shapes_keys[shape_idx]
            k = self.np_random.integers(0, 4)
            grid = np.rot90(SHAPES[shape_name], k=k).copy()
            self.pieces_grids.append(grid)
            h, w = grid.shape
            self.pieces_padded[i, :h, :w] = grid

    def _update_all_valid_placements(self):
        self.valid_placements = np.zeros((self.n_pieces, self.grid_size, self.grid_size), dtype=np.int8)
        for i in range(self.n_pieces):
            if self.pieces_used[i]:
                continue
            
            shape_grid = self.pieces_grids[i]
            h, w = shape_grid.shape
            
            windowed_board = sliding_window_view(self.board, window_shape=shape_grid.shape)
            overlaps = (windowed_board & shape_grid).any(axis=(-2, -1))
            self.valid_placements[i, :self.grid_size - h + 1, :self.grid_size - w + 1] = (~overlaps).astype(np.int8)

    def _get_all_placements_result(self):
        results = np.ones(
            (self.n_pieces, self.grid_size, self.grid_size, self.grid_size, self.grid_size),
            dtype=np.int8,
        )

        for i in range(self.n_pieces):
            if self.pieces_used[i]:
                continue

            shape_grid = self.pieces_grids[i]
            h, w = shape_grid.shape

            max_row = self.grid_size - h + 1
            max_col = self.grid_size - w + 1

            for row in range(max_row):
                for col in range(max_col):
                    if not self.valid_placements[i, row, col]:
                        continue

                    board_copy = self.board.copy()
                    board_copy[row : row + h, col : col + w] += shape_grid

                    full_rows = np.all(board_copy == 1, axis=1)
                    full_cols = np.all(board_copy == 1, axis=0)
                    board_copy[full_rows, :] = 0
                    board_copy[:, full_cols] = 0

                    results[i, row, col] = board_copy

        return results

    def _place_piece(self, shape_grid, row, col):
        h, w = shape_grid.shape
        self.board[row:row+h, col:col+w] += shape_grid

    def _clear_lines(self):
        full_rows = np.all(self.board == 1, axis=1)
        full_cols = np.all(self.board == 1, axis=0)
        num_cleared = np.sum(full_rows) + np.sum(full_cols)
        self.board[full_rows, :] = 0
        self.board[:, full_cols] = 0
        return num_cleared



    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        elif self.render_mode == "human":
            self._render_human()

    def _render_ansi(self):
        output = "Board:\n"
        for r in range(self.grid_size):
            row_str = " ".join(["[#]" if val == 1 else "[ ]" for val in self.board[r]])
            output += row_str + "\n"

        output += f"\nCombo: {self.combo}\n"
        for i in range(self.n_pieces):
            status = " (used)" if self.pieces_used[i] else ""
            output += f"\nPiece {i+1}{status}:\n"
            if not self.pieces_used[i]:
                h, w = self.pieces_grids[i].shape
                for r in range(h):
                    row_str = " ".join(["[#]" if val == 1 else "   " for val in self.pieces_grids[i][r]])
                    output += row_str + "\n"

        return output

    def _draw_figure(self):
        if self.fig is None:
            self.fig, self.axes = plt.subplots(1, 4, figsize=(14, 4))
            self.fig.canvas.manager.set_window_title('Block Blast 3P RL')
            if self.render_mode == "human":
                plt.ion()

        for ax in self.axes:
            ax.clear()

        ax_board = self.axes[0]
        ax_board.imshow(self.board, cmap='Blues', vmin=0, vmax=1)
        ax_board.set_title(f"Board  |  combo: {self.combo}")
        ax_board.set_xticks(np.arange(-.5, self.grid_size, 1), minor=True)
        ax_board.set_yticks(np.arange(-.5, self.grid_size, 1), minor=True)
        ax_board.grid(which='minor', color='black', linestyle='-', linewidth=2)
        ax_board.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        for i in range(self.n_pieces):
            ax = self.axes[i + 1]
            used = bool(self.pieces_used[i])

            padded = np.zeros((self.piece_box_size, self.piece_box_size))
            h, w = self.pieces_grids[i].shape
            padded[:h, :w] = self.pieces_grids[i]
            masked = np.ma.masked_where(padded == 0, padded)

            cmap = 'Greys' if used else 'Oranges'
            ax.imshow(masked, cmap=cmap, vmin=0, vmax=1)

            title = f"Piece {i+1}" + (" (used)" if used else "")
            ax.set_title(title)
            ax.set_xlim(-0.5, self.piece_box_size - 0.5)
            ax.set_ylim(self.piece_box_size - 0.5, -0.5)
            ax.set_xticks(np.arange(-.5, self.piece_box_size, 1), minor=True)
            ax.set_yticks(np.arange(-.5, self.piece_box_size, 1), minor=True)
            ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
            ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    def _render_human(self):
        self._draw_figure()
        plt.draw()
        plt.pause(0.001)

    def _render_rgb_array(self):
        self._draw_figure()
        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        rgb_array = rgba[..., :3]
        plt.close(self.fig)  # avoids duplicate plots in jupyter
        self.fig = None
        return rgb_array

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
