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

OBSERVATION_CHOICE = 2  # 1 or 2
REWARD_CHOICE = 1   # 1, 2, or 3

def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Observation space from Gymnasium (https://gymnasium.farama.org/api/spaces/)
    """
    if OBSERVATION_CHOICE == 1:
        cell_values = np.ones(shape = (env.grid.shape[0], env.grid.shape[1])) * 7
        return gym.spaces.MultiDiscrete(cell_values.flatten())
    elif OBSERVATION_CHOICE == 2:
        cell_values = np.ones(shape = (5, 5)) * 7
        return gym.spaces.MultiDiscrete(cell_values.flatten())


def observation(grid: np.ndarray):
    """
    Function that returns the observation for the current state of the environment.
    """
    if OBSERVATION_CHOICE == 1:
        # If the observation returned is not the same shape as the observation_space, an error will occur!
        # Make sure to make changes to both functions accordingly.
        id_grid = np.zeros(shape = (grid.shape[0], grid.shape[1]))
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                id_grid[x, y] = COLOR_TO_ID[tuple(grid[x, y])]


        return id_grid.flatten()
    elif OBSERVATION_CHOICE == 2:
        id_grid = np.zeros(shape = (10, 10))
        for x in range(10):
            for y in range(10):
                id_grid[x, y] = COLOR_TO_ID[tuple(grid[x, y])]
        
        agent_pos = np.argwhere(id_grid == 3)
        if len(agent_pos) == 0:
            obs = np.full((5, 5), 2)
            return obs.flatten()
        agentx, agenty = agent_pos[0]
        obs = np.zeros((5, 5), dtype=np.uint8)
        counter = 0
        for r in range(5):
            for c in range(5):
                globx = agentx + (r - 2)
                globy = agenty + (c - 2)
            
                if 0 <= globx < 10 and 0 <= globy < 10:
                    obs[r,c] = id_grid[globx,globy]
                else:
                    obs[r,c] = 2


        
        return obs.flatten()



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

    

        



    if REWARD_CHOICE == 1:
        return 1 if new_cell_covered else 0
    elif REWARD_CHOICE == 2:
        exploration_reward = 10 if new_cell_covered else 0
        time_step_penalty = -1
        failure_penalty = -steps_remaining if game_over else 0
        return exploration_reward + time_step_penalty + failure_penalty
    elif REWARD_CHOICE == 3:
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
            [abs(agent_pos - (e.y * 10 + e.x)) for e in enemies],  # Fixed position calculation
            default=10 * 2
        )
        danger_penalty = -8 * (1 - min(min_enemy_dist / (10 * 1.5), 1))
        
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
        return total_reward

