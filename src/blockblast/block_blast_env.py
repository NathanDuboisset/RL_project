import numpy as np
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

class BlockBlastEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi", "rgb_array","no_render"]}

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
        
        self.action_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.int8),
                "piece": spaces.Box(low=0, high=1, shape=(self.piece_box_size, self.piece_box_size), dtype=np.int8),
                "valid_placements": spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.int8),
                "placements_result": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.grid_size, self.grid_size, self.grid_size, self.grid_size),
                    dtype=np.int8,
                ),
            }
        )

        self.shapes_keys = list(SHAPES.keys())
        
        if shape_probs is None:
            self.shape_probs = np.ones(len(self.shapes_keys)) / len(self.shapes_keys)
        else:
            assert len(shape_probs) == len(self.shapes_keys), "Probabilities must match number of shapes"
            probs = np.array(shape_probs)
            self.shape_probs = probs / np.sum(probs)

        self.fig = None
        self.ax = None
        self.board = None
        self.current_piece = None
        self.current_shape_grid = None
        self.valid_placements = None
        self.placements_result = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self._sample_new_piece()
        self.valid_placements = self._get_valid_placements(self.current_shape_grid)
        self.placements_result = self._get_placements_result(self.current_shape_grid)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def step(self, action):
        row = action // self.grid_size
        col = action % self.grid_size

        if not self.valid_placements[row, col]:
            return self._get_obs(), -1.0, True, False, self._get_info()

        self._place_piece(self.current_shape_grid, row, col)
        lines_cleared = self._clear_lines()

        # small reward for surviving, bigger reward for clearing lines
        reward = 0.1
        if lines_cleared > 0:
            reward = self.base_points * (lines_cleared ** 2)

        self._sample_new_piece()
        self.valid_placements = self._get_valid_placements(self.current_shape_grid)
        self.placements_result = self._get_placements_result(self.current_shape_grid)
        terminated = not np.any(self.valid_placements)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_obs(self):
        return {
            "board": self.board.copy(),
            "piece": self.current_piece.copy(),
            "valid_placements": self.valid_placements.copy(),
            "placements_result": (self.placements_result[0].copy(), self.placements_result[1].copy()),
        }

    def _get_info(self):
        return {}

    def _sample_new_piece(self):
        shape_idx = self.np_random.choice(len(self.shapes_keys), p=self.shape_probs)
        shape_name = self.shapes_keys[shape_idx]
        k = self.np_random.integers(0, 4)
        self.current_shape_grid = np.rot90(SHAPES[shape_name], k=k).copy()

        padded_piece = np.zeros((self.piece_box_size, self.piece_box_size), dtype=np.int8)
        h, w = self.current_shape_grid.shape
        padded_piece[:h, :w] = self.current_shape_grid
        self.current_piece = padded_piece

    def _get_valid_placements(self, shape_grid):
        h, w = shape_grid.shape
        valid_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        windowed_board = sliding_window_view(self.board, window_shape=shape_grid.shape)
        overlaps = (windowed_board & shape_grid).any(axis=(-2, -1))
        
        valid_mask[:self.grid_size - h + 1, :self.grid_size - w + 1] = (~overlaps).astype(np.int8)
        return valid_mask

    def _get_placements_result(self, shape_grid):
        """
        For each possible (row, col) placement, return the resulting board
        after placing the current piece and clearing lines. If the placement
        is invalid, the result is an all-blocked grid.
        """
        # default: all blocked grids for invalid placements
        results = np.ones(
            (self.grid_size, self.grid_size, self.grid_size, self.grid_size),
            dtype=np.int8,
        )

        h, w = shape_grid.shape

        # only positions where the top-left of the shape can fit on the board
        max_row = self.grid_size - h + 1
        max_col = self.grid_size - w + 1

        rewards = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        for row in range(max_row):
            for col in range(max_col):
                if not self.valid_placements[row, col]:
                    continue

                board_copy = self.board.copy()
                board_copy[row : row + h, col : col + w] += shape_grid

                # apply the same clearing logic as in the real env step
                full_rows = np.all(board_copy == 1, axis=1)
                full_cols = np.all(board_copy == 1, axis=0)
                board_copy[full_rows, :] = 0
                board_copy[:, full_cols] = 0

                results[row, col] = board_copy
                
                #apply same reward logic as in the real env step for hypothetical reward calculation
                reward = 0.1
                lines_cleared = np.sum(full_rows) + np.sum(full_cols)
                if lines_cleared > 0:
                    reward = self.base_points * (lines_cleared ** 2)
                rewards[row, col] = reward

        return results,rewards

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
        else:
            pass

    def _render_ansi(self):
        output = "Board:\n"
        for r in range(self.grid_size):
            row_str = " ".join(["[#]" if val == 1 else "[ ]" for val in self.board[r]])
            output += row_str + "\n"
        
        output += "\nCurrent Piece:\n"
        h, w = self.current_shape_grid.shape
        for r in range(h):
            row_str = " ".join(["[#]" if val == 1 else "   " for val in self.current_shape_grid[r]])
            output += row_str + "\n"
        
        return output

    def _draw_figure(self):
        if self.fig is None:
            self.fig, (self.ax_board, self.ax_piece) = plt.subplots(1, 2, figsize=(8, 4))
            self.fig.canvas.manager.set_window_title('Block Blast RL')
            if self.render_mode == "human":
                plt.ion()
            
        self.ax_board.clear()
        self.ax_piece.clear()
        
        self.ax_board.imshow(self.board, cmap='Blues', vmin=0, vmax=1)
        self.ax_board.set_title("Board")
        self.ax_board.set_xticks(np.arange(-.5, self.grid_size, 1), minor=True)
        self.ax_board.set_yticks(np.arange(-.5, self.grid_size, 1), minor=True)
        self.ax_board.grid(which='minor', color='black', linestyle='-', linewidth=2)
        self.ax_board.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        padded = np.zeros((self.piece_box_size, self.piece_box_size))
        h, w = self.current_shape_grid.shape
        padded[:h, :w] = self.current_shape_grid
        masked_piece = np.ma.masked_where(padded == 0, padded)  # hide empty cells
        
        self.ax_piece.imshow(masked_piece, cmap='Oranges', vmin=0, vmax=1)
        self.ax_piece.set_title("Current Piece")
        self.ax_piece.set_xlim(-0.5, self.piece_box_size - 0.5)
        self.ax_piece.set_ylim(self.piece_box_size - 0.5, -0.5)
        self.ax_piece.set_xticks(np.arange(-.5, self.piece_box_size, 1), minor=True)
        self.ax_piece.set_yticks(np.arange(-.5, self.piece_box_size, 1), minor=True)
        self.ax_piece.grid(which='minor', color='black', linestyle='-', linewidth=1)
        self.ax_piece.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

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
