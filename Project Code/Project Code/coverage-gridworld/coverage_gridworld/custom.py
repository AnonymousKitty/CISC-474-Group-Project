import numpy as np
import gymnasium as gym
from typing import Optional

def observation_space(env: gym.Env) -> gym.spaces.Space:

    # The grid has (10, 10, 3) shape and can store values from 0 to 255 (uint8). To use the whole grid as the
     # observation space, we can consider a MultiDiscrete space with values in the range [0, 256).
    max_value = 256
    grid_shape = (10, 10, 3)
    obs_space = gym.spaces.MultiDiscrete([max_value] * np.prod(grid_shape))
     # if MultiDiscrete is used, it's important to flatten() numpy arrays!
    return obs_space

def observation(grid: np.ndarray):

    return grid.flatten()
    

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
    reward = 0
    
    if new_cell_covered:
        reward += 1

    if game_over:
        reward -= 70  # Big punishment for being seen by enemies
        
    penalty_for_slow_progress = 0.1 * (steps_remaining / 100)  # Adjust scale of penalty
    reward -= penalty_for_slow_progress

    
    return reward