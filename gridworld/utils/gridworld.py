from __future__ import annotations

from typing import Literal, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.axes
import dataclasses


@dataclasses.dataclass
class GridWorldEnvConfig:
    grid_size: int = 25
    death_threshold: float = -10.0
    goal_threshold: float = 20.0

    @property
    def death_cells(self) -> set[tuple[int, int]]:
        return set()

    @property
    def goal_cells(self) -> set[tuple[int, int]]:
        return set()

    @property
    def wall_cells(self) -> set[tuple[int, int]]:
        """
        These cells there's no penalty, BUT the agent can't move into them.
        """
        return set()

    @property
    def initial_cells(self) -> set[tuple[int, int]]:
        """
        The cells that the agent can start in.
        """
        return {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}

    @property
    def cx(self) -> int:
        return self.grid_size // 2

    @property
    def cy(self) -> int:
        return self.grid_size // 2

    @property
    def center(self) -> np.ndarray:
        return np.array([self.cx, self.cy], int)

    @property
    def reward_map(self) -> np.ndarray:
        size = self.grid_size
        M = np.zeros((size, size), float)
        for x in range(size):
            for y in range(size):
                if (x, y) in self.death_cells:
                    M[y, x] = self.death_threshold
                elif (x, y) in self.goal_cells:
                    M[y, x] = self.goal_threshold
        return M


@dataclasses.dataclass
class ThreeGoalsConfig(GridWorldEnvConfig):
    custom_triangle_radius: int | None = None
    custom_goal_radius: int = 3

    @property
    def triangle_radius(self) -> float:
        if self.custom_triangle_radius is not None:
            return self.custom_triangle_radius
        return self.grid_size / 3

    @property
    def radius(self) -> int:
        if self.custom_goal_radius is not None:
            r = self.custom_goal_radius
        else:
            r = (self.grid_size // 2) - 2
        assert r % 2 == 1, "Radius must be odd"
        return r

    @property
    def death_cells(self) -> set[tuple[int, int]]:
        return {
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if x in [self.cx - 1, self.cx, self.cx + 1]
            and y
            in [
                self.cy - 1,
                self.cy,
                self.cy + 1,
            ]
        }

    @property
    def goal_cells(self) -> set[tuple[int, int]]:
        centerpoints = [
            (12, 18),
            (7, 7),
            (17, 7),
        ]
        blocks = []
        for cx, cy in centerpoints:
            blocks.extend(
                (cx + x - self.radius // 2, cy + y - self.radius // 2)
                for x in range(self.radius)
                for y in range(self.radius)
            )
        return set(blocks)


class TwoWallsConfig(GridWorldEnvConfig):
    @property
    def death_cells(self) -> set[tuple[int, int]]:
        return set()

    @property
    def goal_cells(self) -> set[tuple[int, int]]:
        return {
            (x, y)
            for x in range(self.grid_size)
            for y in [self.grid_size - 1, self.grid_size - 2] * self.grid_size
        } | {(x, y) for x in range(self.grid_size) for y in [0, 1] * self.grid_size}


class TreeInTheMiddleConfig(GridWorldEnvConfig):
    @property
    def death_cells(self) -> set[tuple[int, int]]:
        cells = {
            (x, y)
            # for x in range(self.cx+2, self.cx + 4)
            # for y in range(self.cy - 5, self.cy + 6)
            for x in range(self.cx + 1, self.cx + 6)
            for y in range(self.cy - 2, self.cy + 3)
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size
        }
        return cells

    @property
    def goal_cells(self) -> set[tuple[int, int]]:
        return {
            (x, y)
            for x in [self.grid_size - 1, self.grid_size - 2] * self.grid_size
            for y in range(self.cx - 2, self.cx + 3)
        }


class TwoSlitsConfig(GridWorldEnvConfig):
    """
    A grid world with a long wall-like obstacle (depth cells), with two openings.
    The goals are at the right edge of the environment.
    """

    @property
    def wall_cells(self) -> set[tuple[int, int]]:
        death_cell_col = self.cx + 3
        opening_rows = [
            *list(range(self.cy - 8, self.cy - 4)),
            *list(range(self.cy + 5, self.cy + 9)),
        ]
        return {
            (death_cell_col, row)
            for row in range(self.grid_size)
            if row not in opening_rows
        }

    @property
    def goal_cells(self) -> set[tuple[int, int]]:
        return {(self.grid_size - 1, y) for y in range(self.grid_size)}


class CShapeConfig(GridWorldEnvConfig):
    """
    A gridworld with a C (but flipped) shaped wall, where the reward is at the center of the C.
    """

    @property
    def wall_cells(self) -> set[tuple[int, int]]:
        wall_rows = [self.cy - 5, self.cy + 5]
        wall_cols = [self.cx - 5, self.cx + 5]
        cells = set()
        cells = (
            cells
            | {
                (x, y)
                for x in range(wall_rows[0], wall_rows[1] + 1)
                for y in [wall_cols[0], wall_cols[1]]
            }
            | {
                (x, y)
                for y in range(wall_rows[0], wall_rows[1] + 1)
                for x in [wall_cols[1]]
            }
        )
        return cells

    @property
    def goal_cells(self) -> set[tuple[int, int]]:
        return {
            (x, y)
            for x in range(self.cx - 2, self.cx + 3)
            for y in range(self.cy - 2, self.cy + 3)
        }

    # NOTE(cmk) I removed this for now, because it was having a hard time converging even with PPO.
    # @property
    # def initial_cells(self) -> set[tuple[int, int]]:
    #     return {
    #         (x, y)
    #         for x in range(self.cx + 8, self.cx + 15)
    #         for y in range(self.cy - 2, self.cy + 3)
    #     }


class GridWorldEnv(gym.Env):
    """
   A grid-based environment with selectable reward map configurations.

    - The grid world is defined by the `mode` parameter, which selects a specific layout
      (e.g. 'three_goals', 'two_walls', etc.). Each mode defines the reward map, death zones,
      goal zones, and walls.
    - The agent state is its 2D position on the grid, normalized to [-1,1]^2 in observations.
    - The action space is continuous in [-1,1]^2, representing intended delta movement.
      Actions are clipped and rounded to discrete grid moves of -1, 0, or 1 per axis.
    - Episodes terminate on entering a death cell, a goal cell, or reaching the max number of steps.
    - The environment enforces walls by blocking movement into wall cells.

    Observation:
        - 2D position normalized to [-1,1].

    Action:
        - 2D continuous delta in [-1,1], interpreted as a direction for grid moves.

    Reward:
        - Defined by the reward map of the selected mode.
        - Typically sparse: high positive in goal cells, large negative in death cells.
    """

    config: GridWorldEnvConfig
    metadata = {"render_modes": ["matplotlib"], "render_fps": 4}

    def __init__(
        self,
        mode: (
            Literal[
                "three_goals",
                "two_walls",
                "tree_in_the_middle",
                "two_slits",
                "cshape",
            ]
            | None
        ) = "two_walls",
        config: Optional[GridWorldEnvConfig] = None,
        max_steps: int = 100,
    ):
        super().__init__()
        if config is None:
            if mode == "three_goals":
                config = ThreeGoalsConfig()
            elif mode == "two_walls":
                config = TwoWallsConfig()
            elif mode == "tree_in_the_middle":
                config = TreeInTheMiddleConfig()
            elif mode == "two_slits":
                config = TwoSlitsConfig()
            elif mode == "cshape":
                config = CShapeConfig()
            else:
                raise ValueError(f"Invalid config name: {mode}")

        self.grid_size = config.grid_size
        self.max_steps = max_steps
        self.center = config.center
        self.config = config

        # build reward map
        self.reward_map = self.config.reward_map

        # spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # plotting setup
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.patch.set_facecolor("#f0f0f0")
        self.ax.set_facecolor("#f8f9fa")
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mask = (
            (self.reward_map != self.config.death_threshold) & 
            (self.reward_map != self.config.goal_threshold)
        )

        # Exclude wall cells from potential starting positions
        for wall_cell in self.config.wall_cells:
            if (
                0 <= wall_cell[0] < self.grid_size
                and 0 <= wall_cell[1] < self.grid_size
            ):
                mask[wall_cell[1], wall_cell[0]] = False

        # Filter mask by initial_cells
        initial_cells_mask = np.zeros_like(mask, dtype=bool)
        for x, y in self.config.initial_cells:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                initial_cells_mask[y, x] = True

        mask &= initial_cells_mask

        ys, xs = np.where(mask)
        if len(xs) == 0:
            raise RuntimeError(
                "No valid starting positions available. Check your wall, death, and goal cell configurations."
            )
        idx = self.np_random.choice(len(xs))
        self.pos = np.array([xs[idx], ys[idx]], int)

        mask = (self.reward_map != self.config.death_threshold) & (self.reward_map != self.config.goal_threshold)
        ys, xs = np.where(mask)
        idx = self.np_random.choice(len(xs))
        self.pos = np.array([xs[idx], ys[idx]], int)
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return ((self.pos - self.center)/(self.grid_size//2)).astype(np.float32)

    def step(self, action):
        move = np.round(np.clip(action, -1, 1)).astype(int)

        new_pos = np.clip(self.pos + move, 0, self.grid_size - 1)
        if tuple(new_pos) in self.config.wall_cells:
            new_pos = self.pos  # Stay in current position
    
        self.pos = new_pos
        grid_pos = self.pos
        
        r = float(self.reward_map[grid_pos[1], grid_pos[0]])
        done = (r == self.config.death_threshold) or (r == self.config.goal_threshold)

        # print(f"action {action}, move {move}, pos {self.pos}")
        self.steps += 1
        return self._get_obs(), r, done, self.steps >= self.max_steps, {}
        
    def _render_grid(self, ax: matplotlib.axes.Axes, x_limits: tuple[float, float], y_limits: tuple[float, float]):
        size = self.grid_size
        img = np.ones((size, size, 3), dtype=np.uint8) * 240
        
        # Color death cells
        for x, y in self.config.death_cells:
            if 0 <= x < size and 0 <= y < size:
                img[y, x] = [229, 57, 70] # Red
        
        # Color goal cells
        for x, y in self.config.goal_cells:
            if 0 <= x < size and 0 <= y < size:
                img[y, x] = [42, 157, 143] # Green
        
        # Color wall cells
        for x, y in self.config.wall_cells:
            if 0 <= x < size and 0 <= y < size:
                img[y, x] = [128, 128, 128]  # Gray
        
        # Color initial cells (conditionally)
        if len(self.config.initial_cells) < size * size:
            for x_init, y_init in self.config.initial_cells:
                if 0 <= x_init < size and 0 <= y_init < size:
                    # Check if the cell is not already a death, goal, or wall cell
                    is_special_cell = False
                    if (x_init, y_init) in self.config.death_cells:
                        is_special_cell = True
                    if not is_special_cell and (x_init, y_init) in self.config.goal_cells:
                        is_special_cell = True
                    if not is_special_cell and (x_init, y_init) in self.config.wall_cells:
                        is_special_cell = True
                    
                    if not is_special_cell:
                        img[y_init, x_init] = [173, 216, 230] # Light blue

        ax.imshow(img, origin="lower", extent=(0, size, 0, size), interpolation="none")
        
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        
        majors = np.arange(0, size + 1, 5)
        minors = np.arange(0, size + 1, 1)
        ax.set_xticks(majors)
        ax.set_yticks(majors)
        ax.set_xticks(minors, minor=True)
        ax.set_yticks(minors, minor=True)
        ax.grid(which="minor", color="#ddd", linestyle="-", linewidth=0.5)
        ax.grid(which="major", color="#bbb", linestyle="--", linewidth=1)

    def render(self, *_args, **_kwargs):
        self.ax.clear()
        self._render_grid(self.ax, x_limits=(1, self.grid_size - 1), y_limits=(1, self.grid_size - 1))

        # add agent cell
        x,y = self.pos
        self.ax.add_patch(Rectangle((x,y),1,1, facecolor='#264653', edgecolor='#1d3557', linewidth=1.5, zorder=5))

        self.fig.canvas.draw()
        plt.pause(0.001)

    def render_into_axes(self, ax: matplotlib.axes.Axes):
        self._render_grid(ax, x_limits=(0, self.grid_size), y_limits=(0, self.grid_size))

    def close(self):
        plt.ioff()
        plt.close(self.fig)
