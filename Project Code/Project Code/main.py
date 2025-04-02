import random
import time
import gymnasium
import coverage_gridworld  # must be imported, even though it's not directly referenced
import numpy as np
from itertools import cycle  # Add this import
from collections import deque
import math

def human_player():
    # Write the letter for the desired movement in the terminal/console and then press Enter

    input_action = input()
    if input_action.lower() == "w":
        return 3
    elif input_action.lower() == "a":
        return 0
    elif input_action.lower() == "s":
        return 1
    elif input_action.lower() == "d":
        return 2
    elif input_action.isdigit():
        return int(input_action)
    else:
        return 4


def random_player():
    return random.randint(0, 4)


maps = [
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [3, 0, 0, 2, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 2, 2, 2, 2, 2, 2, 0],
        [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 2, 2, 2, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 2, 0, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    ],
    [
        [3, 2, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
        [0, 2, 0, 2, 0, 4, 2, 4, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
    ],
    [
        [3, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 4, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 4, 0, 4, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
    ],
    [
        [3, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0]
    ]
]

env = gymnasium.make("sneaky_enemies", predefined_map_list=maps, activate_game_status=True)
# env = gymnasium.make("sneaky_enemies", render_mode="human", predefined_map_list=None, activate_game_status=True)
num_episodes = 1000

class SmartCoverageAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.q_table = {}
        
        # Dynamic learning parameters
        self.learning_rate = 0.7
        self.min_learning_rate = 0.01
        self.discount_factor = 0.95
        self.exploration_rate = 1.0
        self.min_exploration = 0.01
        self.exploration_decay = 0.9995
        
        # Reward normalization
        self.reward_memory = deque(maxlen=100)
        self.best_avg_reward = -float('inf')
        
    def get_state_key(self, state):
        """Create a compact state representation focusing on critical elements"""
        grid, agent_pos, enemies = state
        # Focus on agent's immediate surroundings (3x3 grid around agent)
        agent_y, agent_x = divmod(agent_pos, 10)  # Assuming 10x10 grid
        nearby_cells = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                y, x = agent_y + dy, agent_x + dx
                if 0 <= y < 10 and 0 <= x < 10:
                    nearby_cells.append(grid[y*10 + x])
                else:
                    nearby_cells.append(0)  # Out of bounds
        
        # Include closest enemy positions
        enemy_distances = [abs(agent_pos - e) for e in enemies if e != 0]
        closest_enemies = sorted(enemy_distances)[:2] if enemy_distances else [0, 0]
        
        return (tuple(nearby_cells), agent_pos, tuple(closest_enemies))
    
    def get_action(self, state):
        state_key = self.get_state_key(state)
        
        # Initialize Q-values for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = np.random.uniform(-0.1, 0.1, size=self.action_space.n)
            
        # Exploration-exploitation tradeoff
        if random.random() < self.exploration_rate:
            return random.choice([a for a in range(self.action_space.n) if a != 4])  # Avoid STAY
        return np.argmax(self.q_table[state_key])
    
    def learn(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize Q-values if needed
        for key in [state_key, next_state_key]:
            if key not in self.q_table:
                self.q_table[key] = np.random.uniform(-0.1, 0.1, size=self.action_space.n)
                
        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key]) if not done else 0
        learned_value = reward + self.discount_factor * max_next_q
        self.q_table[state_key][action] = (1 - self.learning_rate) * current_q + self.learning_rate * learned_value
        
        # Track rewards for adaptive learning
        self.reward_memory.append(reward)
        if len(self.reward_memory) == self.reward_memory.maxlen:
            current_avg = np.mean(self.reward_memory)
            if current_avg > self.best_avg_reward:
                self.best_avg_reward = current_avg
                self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
                
        # Adaptive learning rate
        self.learning_rate = max(self.min_learning_rate, 
                               self.learning_rate * (0.999 if reward > 0 else 0.99))

# Initialize with all 5 maps
env = gymnasium.make("sneaky_enemies", render_mode="human", predefined_map_list=maps, activate_game_status=True)
agent = SmartCoverageAgent(env.action_space)
performance_history = []

# Train for exactly 200 episodes per map (1000 total episodes)
num_episodes_per_map = 200
current_map_index = 0

for episode in range(1, 1001):
    # Switch to next map every 200 episodes
    if episode % num_episodes_per_map == 0:
        current_map_index = (current_map_index + 1) % len(maps)
        env.close()
        env = gymnasium.make("sneaky_enemies", 
                           render_mode="human",
                           predefined_map_list=[maps[current_map_index]],
                           activate_game_status=True)
    
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _, info = env.step(action)
        
        # Add movement information
        info["agent_moved"] = action != 4
        
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        # Optional: Slow down for visualization
        time.sleep(0.05)
        
    performance_history.append(total_reward)
    
    # Progress reporting
    if episode % 50 == 0:
        avg_reward = np.mean(performance_history[-50:])
        print(f"Episode {episode:4d} (Map {current_map_index+1}) | "
              f"Avg Reward: {avg_reward:7.1f} | "
              f"ε: {agent.exploration_rate:.3f} | "
              f"α: {agent.learning_rate:.3f} | "
              f"Coverage: {info['total_covered_cells']}/{info['coverable_cells']}")

env.close()