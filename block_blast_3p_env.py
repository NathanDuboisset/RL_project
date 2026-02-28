import numpy as np
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
            shape_probs=None
    ):
        super().__init__()

        self.render_mode = render_mode
        self.base_points = base_points
        self.grid_size = 8
        self.piece_box_size = 5
        self.n_pieces = 3

        # piece_idx * 64 + position
        self.action_space = spaces.Discrete(self.n_pieces * self.grid_size * self.grid_size)

        self.observation_space = spaces.Dict({
            "board":       spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.int8),
            "pieces":      spaces.Box(low=0, high=1, shape=(self.n_pieces, self.piece_box_size, self.piece_box_size), dtype=np.int8),
            "pieces_used": spaces.MultiBinary(self.n_pieces),
            "combo":       spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.combo = 0
        self.round_had_clear = False
        self.pieces_used = np.zeros(self.n_pieces, dtype=np.int8)
        self._sample_new_pieces()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def step(self, action):
        piece_idx = action // (self.grid_size * self.grid_size)
        pos = action % (self.grid_size * self.grid_size)
        row = pos // self.grid_size
        col = pos % self.grid_size

        if self.pieces_used[piece_idx] or not self._can_place(self.pieces_grids[piece_idx], row, col):
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
            terminated = any(
                not self._has_valid_moves(self.pieces_grids[i])
                for i in range(self.n_pieces)
            )
        else:
            terminated = not any(
                not self.pieces_used[i] and self._has_valid_moves(self.pieces_grids[i])
                for i in range(self.n_pieces)
            )

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_obs(self):
        return {
            "board":       self.board.copy(),
            "pieces":      self.pieces_padded.copy(),
            "pieces_used": self.pieces_used.copy(),
            "combo":       np.array([self.combo], dtype=np.int32),
        }

    def _get_info(self):
        return {}

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

    def _can_place(self, shape_grid, row, col):
        h, w = shape_grid.shape
        if row + h > self.grid_size or col + w > self.grid_size:
            return False
        board_crop = self.board[row:row+h, col:col+w]
        if np.any((board_crop + shape_grid) > 1):
            return False
        return True

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

    def _has_valid_moves(self, shape_grid):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self._can_place(shape_grid, r, c):
                    return True
        return False

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
        plt.close(self.fig)  # avoids duplicate plots in Jupyter
        self.fig = None
        return rgb_array

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
