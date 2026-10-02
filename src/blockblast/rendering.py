"""Game-like rendering of BlockBlast3PEnv episodes, for figures and GIFs.

Purely cosmetic: cells are coloured by the piece that filled them, which the
environment does not track, so colours are maintained here.

    frames = record_episode(BlockBlast3PEnv(), policy, seed=0, max_steps=150)
    save_gif(frames, "assets/dvn_3p.gif")
"""
import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from blockblast.block_blast_3p_env import SHAPES

BG = "#1b1f3b"
EMPTY = "#2a3060"
TEXT = "#f1f2f6"
GOLD = "#ffd32a"
PALETTE = ["#ff5e57", "#ffa801", "#0be881", "#4bcffa", "#a55eea", "#ff3f9a", "#3c40c6"]


def _canonical(grid: np.ndarray) -> bytes:
    rots = [np.ascontiguousarray(np.rot90(grid, k)) for k in range(4)]
    return min(r.tobytes() + bytes(r.shape) for r in rots)


_SHAPE_COLOR = {_canonical(g): i % len(PALETTE) for i, g in enumerate(SHAPES.values())}


def _cell(ax, x, y, color, size=0.9, alpha=1.0, edge=None, lw=0.0):
    ax.add_patch(FancyBboxPatch(
        (x + (1 - size) / 2, y + (1 - size) / 2), size, size,
        boxstyle="round,pad=0,rounding_size=0.18", facecolor=color, alpha=alpha,
        edgecolor=edge or color, linewidth=lw,
    ))


class EpisodeRenderer:
    """Keeps per-cell colours across steps and draws frames."""

    def __init__(self, grid_size: int = 8, dpi: int = 80):
        self.n = grid_size
        self.dpi = dpi
        self.colors = np.full((grid_size, grid_size), -1, dtype=int)

    @staticmethod
    def piece_color(grid: np.ndarray) -> int:
        """One colour per base shape, whatever the rotation."""
        return _SHAPE_COLOR[_canonical(grid)]

    def frame(self, board, pieces, used, score, step, combo, placed=None, placed_color=None,
              full_rows=(), full_cols=(), caption="") -> Image.Image:
        n = self.n
        fig = plt.figure(figsize=(4.0, 5.6), dpi=self.dpi, facecolor=BG)
        ax = fig.add_axes([0.05, 0.25, 0.90, 0.64])
        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)
        ax.set_aspect("equal")
        ax.axis("off")

        for r in range(n):
            for c in range(n):
                if board[r, c]:
                    col = PALETTE[self.colors[r, c]] if self.colors[r, c] >= 0 else "#808e9b"
                    _cell(ax, c, r, col)
                else:
                    _cell(ax, c, r, EMPTY)
        if placed is not None:
            for r, c in placed:
                _cell(ax, c, r, PALETTE[placed_color], edge="white", lw=2.0)
        for r in full_rows:
            ax.add_patch(plt.Rectangle((0.02, r + 0.02), n - 0.04, 0.96, fill=False, edgecolor=GOLD, linewidth=3))
        for c in full_cols:
            ax.add_patch(plt.Rectangle((c + 0.02, 0.02), 0.96, n - 0.04, fill=False, edgecolor=GOLD, linewidth=3))

        fig.text(0.5, 0.965, f"Score {score:,.0f}", color=TEXT, ha="center", va="center",
                 fontsize=15, fontweight="bold")
        fig.text(0.5, 0.925, f"step {step}   combo {combo}", color="#a4b0be", ha="center", va="center", fontsize=9)

        # tray with the 3 pieces of the round
        for i, grid in enumerate(pieces):
            tray = fig.add_axes([0.03 + i * 0.33, 0.04, 0.28, 0.19])
            tray.set_xlim(0, 5)
            tray.set_ylim(5, 0)
            tray.set_aspect("equal")
            tray.axis("off")
            h, w = grid.shape
            oy, ox = (5 - h) / 2, (5 - w) / 2
            color = PALETTE[self.piece_color(grid)]
            for r in range(h):
                for c in range(w):
                    if grid[r, c]:
                        _cell(tray, ox + c, oy + r, EMPTY if used[i] else color)
        if caption:
            fig.text(0.5, 0.015, caption, color="#a4b0be", ha="center", va="bottom", fontsize=8)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=BG, dpi=self.dpi)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def record_episode(env, policy, seed: int = 0, max_steps: int = 200, caption: str = "") -> list[Image.Image]:
    """Play one episode with `policy(obs, env)`; two frames per step: the
    placement (completed lines outlined) and the board after clearing."""
    backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    try:
        obs, _ = env.reset(seed=seed)
        if hasattr(policy, "reset"):
            policy.reset()
        rend = EpisodeRenderer(env.grid_size)
        score, step = 0.0, 0
        frames = [rend.frame(env.board, env.pieces_grids, env.pieces_used, score, step, env.combo, caption=caption)]

        while step < max_steps:
            action = policy(obs, env)
            if action is None:
                break
            p, row, col = env.decode_action(action)
            grid = env.pieces_grids[p]
            pieces, used = list(env.pieces_grids), env.pieces_used.copy()
            used[p] = 1
            color = rend.piece_color(grid)

            placed_board = env.board.copy()
            placed_board[row:row + grid.shape[0], col:col + grid.shape[1]] += grid
            cells = [(row + r, col + c) for r, c in zip(*np.nonzero(grid))]
            full_rows = np.flatnonzero(placed_board.all(axis=1))
            full_cols = np.flatnonzero(placed_board.all(axis=0))
            combo_before = env.combo

            obs, reward, terminated, truncated, _ = env.step(int(action))
            score += reward
            step += 1
            for r, c in cells:
                rend.colors[r, c] = color
            frames.append(rend.frame(placed_board, pieces, used, score, step, combo_before,
                                     placed=cells, placed_color=color,
                                     full_rows=full_rows, full_cols=full_cols, caption=caption))
            rend.colors[env.board == 0] = -1
            if len(full_rows) or len(full_cols):  # extra frame showing the cleared board
                frames.append(rend.frame(env.board, pieces, used, score, step, env.combo, caption=caption))
            if terminated or truncated:
                break

        frames.append(rend.frame(env.board, env.pieces_grids, env.pieces_used, score, step, env.combo,
                                 caption="game over" if step < max_steps else caption))
        return frames
    finally:
        matplotlib.use(backend)


def save_gif(frames: list[Image.Image], path, ms_per_frame: int = 220, end_pause_ms: int = 2000) -> None:
    durations = [ms_per_frame] * (len(frames) - 1) + [end_pause_ms]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
