import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""


# rendering colors
BLACK = (0, 0, 0)            # unexplored cell
WHITE = (255, 255, 255)      # explored cell
BROWN = (101, 67, 33)        # wall
GREY = (160, 161, 161)       # agent
GREEN = (31, 198, 0)         # enemy
RED = (255, 0, 0)            # unexplored cell being observed by an enemy
LIGHT_RED = (255, 127, 127)  # explored cell being observed by an enemy

# color IDs
COLOR_IDS = {
    0: BLACK,      # unexplored cell
    1: WHITE,      # explored cell
    2: BROWN,      # wall
    3: GREY,       # agent
    4: GREEN,      # enemy
    5: RED,        # unexplored cell being observed by an enemy
    6: LIGHT_RED,  # explored cell being observed by an enemy
}

COLOR_TO_ID = {
    BLACK: 0,
    WHITE: 1,
    BROWN: 2,
    GREY: 3,
    GREEN: 4,
    RED: 5,
    LIGHT_RED: 6
}

USE_CNN = True

def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Observation space from Gymnasium (https://gymnasium.farama.org/api/spaces/)
    """
    if USE_CNN == False:
        # The grid has (10, 10, 3) shape and can store values from 0 to 255 (uint8). To use the whole grid as the
        # observation space, we can consider a MultiDiscrete space with values in the range [0, 256).
        cell_values = np.ones(shape = (env.grid.shape[0], env.grid.shape[1])) * 7

        # if MultiDiscrete is used, it's important to flatten() numpy arrays!
        #print(cell_values.flatten())
        return gym.spaces.MultiDiscrete(cell_values.flatten())
    elif USE_CNN:
        return gym.spaces.Box(low=0, high=255, shape=(env.grid.shape[2], env.grid.shape[0], env.grid.shape[1]), dtype=np.uint8)


def observation(grid: np.ndarray):
    """
    Function that returns the observation for the current state of the environment.
    """
    if USE_CNN == False:
        # If the observation returned is not the same shape as the observation_space, an error will occur!
        # Make sure to make changes to both functions accordingly.
        id_grid = np.zeros(shape = (grid.shape[0], grid.shape[1]))
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                id_grid[x, y] = COLOR_TO_ID[tuple(grid[x, y])]


        return id_grid.flatten()
    elif USE_CNN:
        return np.transpose(grid, (2, 0, 1))


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

    # IMPORTANT: You may design a reward function that uses just some of these values. Experiment with different
    # rewards and find out what works best for the algorithm you chose given the observation space you are using

    # Track visited cells (initialize if not present)
    if not hasattr(reward, 'visited_cells'):
        reward.visited_cells = set()
    if not hasattr(reward, 'last_position'):
        reward.last_position = None
    
    # Current cell position
    current_cell = agent_pos
    
    # 1. Exploration incentives
    exploration_bonus = 5 if new_cell_covered else -1
    
    

    # 5. Time pressure (adjusted)
    #steps_penalty = -2
    
    # 6. Catastrophic failure
    failure_penalty = -50 if game_over else 0
    
    # 7. Movement encouragement (adjusted)
    #movement_bonus = 1.0 if current_cell else 0
    

    
    # Composite reward
    total_reward = (
        exploration_bonus +
        #steps_penalty +
        failure_penalty# +
        #movement_bonus
    )

    
    # Clip to reasonable range
    return total_reward#np.clip(total_reward, -15, 25)
