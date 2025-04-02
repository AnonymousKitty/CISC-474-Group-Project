import numpy as np
import gymnasium as gym
from typing import Optional

def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Defines the observation space matching the environment's requirements.
    """
    grid_shape = env.grid.shape
    num_cells = grid_shape[0] * grid_shape[1] * grid_shape[2]
    
    # Agent position space (flattened grid coordinate)
    agent_pos_space = gym.spaces.Discrete(grid_shape[0] * grid_shape[1])
    
    # Enemy positions space (max 5 enemies)
    max_enemies = 5
    enemy_space = gym.spaces.MultiDiscrete([grid_shape[0] * grid_shape[1]] * max_enemies)
    
    return gym.spaces.Tuple((
        gym.spaces.MultiDiscrete([256] * num_cells),  # Grid values
        agent_pos_space,                              # Agent position
        enemy_space                                   # Enemy positions
    ))

def observation(grid: np.ndarray, agent_pos: Optional[int] = None, enemies: Optional[list] = None):
    """
    Modified to work with both:
    - env.py's call (observation(grid))
    - Your code's calls (observation(grid, agent_pos, enemies))
    """
    flattened_grid = grid.flatten()
    
    # If agent_pos not provided, find it in grid (GREY color)
    if agent_pos is None:
        agent_pos = 0  # Default to (0,0) if not found
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if np.array_equal(grid[y, x], np.array([160, 161, 161])):  # GREY = agent
                    agent_pos = y * grid.shape[1] + x
                    break
    
    # If enemies not provided, find them in grid (GREEN color)
    if enemies is None:
        enemy_positions = []
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if np.array_equal(grid[y, x], np.array([31, 198, 0])):  # GREEN = enemy
                    enemy_positions.append(y * grid.shape[1] + x)
    else:
        # Convert enemy objects to positions if passed
        enemy_positions = [e.x * grid.shape[1] + e.y for e in enemies] if enemies else []
    
    # Pad enemy positions to length 5
    max_enemies = 5
    enemy_positions = enemy_positions[:max_enemies] + [0] * (max_enemies - len(enemy_positions))
    
    return (flattened_grid, agent_pos, enemy_positions)

def reward(info: dict) -> float:
    """
    Reward function remains unchanged.
    """
    new_cell_bonus = 10 if info["new_cell_covered"] else 0
    coverage_bonus = 5 * (info["total_covered_cells"] / info["coverable_cells"])
    steps_penalty = -0.1 * (info["coverable_cells"] - info["total_covered_cells"])
    caught_penalty = -100 if info["game_over"] else 0
    
    return new_cell_bonus + coverage_bonus + steps_penalty + caught_penalty