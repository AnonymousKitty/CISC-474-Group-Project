import numpy as np
import gymnasium as gym
from typing import Optional

def observation_space(env: gym.Env) -> gym.spaces.Space:
    grid_shape = env.grid.shape
    num_cells = grid_shape[0] * grid_shape[1] * grid_shape[2]
    
    agent_pos_space = gym.spaces.Discrete(grid_shape[0] * grid_shape[1])
    max_enemies = 5
    enemy_space = gym.spaces.MultiDiscrete([grid_shape[0] * grid_shape[1]] * max_enemies)
    
    return gym.spaces.Tuple((
        gym.spaces.MultiDiscrete([256] * num_cells),
        agent_pos_space,
        enemy_space
    ))

def observation(grid: np.ndarray, agent_pos: Optional[int] = None, enemies: Optional[list] = None):
    flattened_grid = grid.flatten()
    
    if agent_pos is None:
        agent_pos = 0
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if np.array_equal(grid[y, x], np.array([160, 161, 161])):
                    agent_pos = y * grid.shape[1] + x
                    break
    
    if enemies is None:
        enemy_positions = []
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if np.array_equal(grid[y, x], np.array([31, 198, 0])):
                    enemy_positions.append(y * grid.shape[1] + x)
    else:
        enemy_positions = [e.x * grid.shape[1] + e.y for e in enemies] if enemies else []
    
    max_enemies = 5
    enemy_positions = enemy_positions[:max_enemies] + [0] * (max_enemies - len(enemy_positions))
    
    return (flattened_grid, agent_pos, enemy_positions)

def reward(info: dict) -> float:
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    total_covered = info["total_covered_cells"]
    coverable = info["coverable_cells"]
    new_cell = info["new_cell_covered"]
    game_over = info["game_over"]
    grid_size = 10
    
    if not hasattr(reward, 'visited_cells'):
        reward.visited_cells = set()
    if not hasattr(reward, 'last_position'):
        reward.last_position = None
    if not hasattr(reward, 'last_action'):
        reward.last_action = None
    
    y, x = divmod(agent_pos, grid_size)
    
    exploration_bonus = 25 if new_cell else 0
    coverage_bonus = 50 * (total_covered / coverable)
    failure_penalty = -1000 if game_over else 0
    
    danger_penalty = 0
    predicted_danger = False
    current_danger = False

    if enemies:
        for enemy in enemies:
            current_fov = enemy.get_fov_cells()
            current_fov_positions = [fy * grid_size + fx for (fy, fx) in current_fov]
            
            if agent_pos in current_fov_positions:
                return -1000  # Immediate termination penalty
            
            next_orientation = (enemy.orientation - 1) % 4
            predicted_fov = []
            for i in range(1, 5):
                if next_orientation == 0:
                    fx, fy = enemy.x - i, enemy.y
                elif next_orientation == 1:
                    fx, fy = enemy.x, enemy.y + i
                elif next_orientation == 2:
                    fx, fy = enemy.x + i, enemy.y
                else:
                    fx, fy = enemy.x, enemy.y - i
                
                if 0 <= fx < grid_size and 0 <= fy < grid_size:
                    predicted_fov.append(fy * grid_size + fx)
            
            if agent_pos in predicted_fov:
                predicted_danger = True
                danger_penalty -= 1000  # Large penalty for stepping into predicted danger
            
            nearby_penalty = -30 if any(abs(agent_pos - pos) <= 1 for pos in predicted_fov) else 0
            danger_penalty += nearby_penalty
    
    avoidance_bonus = 10 if predicted_danger and info.get("agent_moved", False) and danger_penalty == 0 else 0
    
    backtrack_penalty = 0
    if reward.last_position is not None and agent_pos in reward.visited_cells:
        revisit_count = info.get('revisit_counts', {}).get(agent_pos, 0)
        if not predicted_danger:
            backtrack_penalty = -5 * (1 + revisit_count * 0.5)
    
    movement_bonus = 2 if info.get("agent_moved", False) else 0
    
    unique_coverage_bonus = 0
    if new_cell:
        reward.visited_cells.add(agent_pos)
        unique_coverage_bonus = 10 * (1 - len(reward.visited_cells)/coverable)
    
    total_reward = (
        exploration_bonus +
        coverage_bonus +
        danger_penalty +
        backtrack_penalty +
        failure_penalty +
        movement_bonus +
        avoidance_bonus +
        unique_coverage_bonus
    )
    
    reward.last_position = agent_pos
    reward.last_action = info.get("last_action", None)
    
    return np.clip(total_reward, -1000, 50)
