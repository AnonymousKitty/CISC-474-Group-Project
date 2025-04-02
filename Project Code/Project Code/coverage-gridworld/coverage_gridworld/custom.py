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
    Function to calculate the reward for the current step based on the state information.

    The info dictionary has the following keys:
    - enemies (list): list of `Enemy` objects. Each Enemy has the following attributes:
        - x (int): column index,
        - y (int): row index,
        - orientation (int): orientation of the agent (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3),
        - fov_cells (list): list of integer tuples indicating the coordinates of cells currently observed by the agent,
    - agent_pos (int): agent position considering the flattened grid (e.g. cell `(2, 3)` corresponds to position `23`),
    - total_covered_cells (int): how many cells have been covered by the agent so far,
    - cells_remaining (int): how many cells are left to be visited in the current map layout,
    - coverable_cells (int): how many cells can be covered in the current map layout,
    - steps_remaining (int): steps remaining in the episode.
    - new_cell_covered (bool): if a cell previously uncovered was covered on this step
    - game_over (bool) : if the game was terminated because the player was seen by an enemy or not
    """
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    total_covered_cells = info["total_covered_cells"]
    cells_remaining = info["cells_remaining"]
    coverable_cells = info["coverable_cells"]
    steps_remaining = info["steps_remaining"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]
    grid_size = 10
    
    # Track visited cells (initialize if not present)
    if not hasattr(reward, 'visited_cells'):
        reward.visited_cells = set()
    if not hasattr(reward, 'last_position'):
        reward.last_position = None
    
    # Current cell position
    current_cell = agent_pos
    
    # 1. Exploration incentives
    exploration_bonus = 20 if new_cell_covered else 0
    
    # 2. Coverage progress (scaled by remaining cells)
    coverage_bonus = 15 * (total_covered_cells / coverable_cells)
    
    # 3. Enemy avoidance (more nuanced distance calculation)
    min_enemy_dist = min(
        [abs(agent_pos - (e.y * grid_size + e.x)) for e in enemies],  # Fixed position calculation
        default=grid_size * 2
    )
    danger_penalty = -8 * (1 - min(min_enemy_dist / (grid_size * 1.5), 1))
    
    # 4. Backtracking penalty (new)
    backtrack_penalty = 0
    if reward.last_position is not None:
        if current_cell in reward.visited_cells:
            # Penalize revisiting cells, with increasing penalty for frequent revisits
            revisit_count = info.get('revisit_counts', {}).get(current_cell, 0)
            backtrack_penalty = -2 * (1 + revisit_count * 0.5)  # -2, -3, -4.5, etc.
    
    # 5. Time pressure (adjusted)
    steps_penalty = -0.1 * (coverable_cells - total_covered_cells)
    
    # 6. Catastrophic failure
    failure_penalty = -250 if game_over else 0
    
    # 7. Movement encouragement (adjusted)
    movement_bonus = 1.0 if info.get("agent_moved", True) else 0
    
    # 8. Unique coverage bonus (new)
    unique_coverage_bonus = 0
    if new_cell_covered:
        reward.visited_cells.add(current_cell)
        unique_coverage_bonus = 5 * (1 - len(reward.visited_cells)/coverable_cells)
    
    # Composite reward
    total_reward = (
        exploration_bonus +
        coverage_bonus +
        danger_penalty +
        backtrack_penalty +
        steps_penalty +
        failure_penalty +
        movement_bonus +
        unique_coverage_bonus
    )
    
    # Update last position
    reward.last_position = current_cell
    
    # Clip to reasonable range
    return np.clip(total_reward, -15, 25)