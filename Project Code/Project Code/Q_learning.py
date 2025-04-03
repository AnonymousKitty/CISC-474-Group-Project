import random
import time
import gymnasium
import coverage_gridworld  # must be imported, even though it's not directly referenced
import numpy as np
from itertools import cycle  # Add this import
from collections import deque
import math

class Q_LearningAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.q_table = {}
        
        # Dynamic learning parameters
        self.learning_rate = 0.7
        self.min_learning_rate = 0.01
        self.max_learning_rate = 0.9  # New maximum learning rate
        self.discount_factor = 0.95
        self.exploration_rate = 1.0
        self.min_exploration = 0.01
        self.exploration_decay = 0.9995
        
        # Tracking variables
        self.reward_history = deque(maxlen=100)
        self.coverage_history = deque(maxlen=100)
        self.best_avg_reward = -float('inf')
        self.best_avg_coverage = 0.0
        self.danger_history = deque(maxlen=100)

    def get_state_key(self, state):
        """Compact state representation focusing on critical features"""
        grid, agent_pos, enemies = state
        agent_y, agent_x = divmod(agent_pos, 10)
        
        # Immediate neighborhood (5x5 grid)
        neighborhood = []
        for dy in [-2, -1, 0, 1, 2]:
            for dx in [-2, -1, 0, 1, 2]:
                y, x = agent_y + dy, agent_x + dx
                neighborhood.append(grid[y*10 + x] if 0 <= y < 10 and 0 <= x < 10 else 0)
        
        # Enemy proximity
        enemy_proximity = [min(abs(agent_pos - e), 10) for e in enemies if e != 0][:3]
        enemy_proximity += [10] * (3 - len(enemy_proximity))  # Pad to 3 enemies
        
        return (tuple(neighborhood), agent_pos, tuple(enemy_proximity))
    
    def get_action(self, state):
        state_key = self.get_state_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.random.uniform(-1, 1, size=self.action_space.n)
            
        # Epsilon-greedy with momentum
        if random.random() < self.exploration_rate:
            return random.choice([a for a in range(self.action_space.n) if a != 4])  # Prefer movement
        return np.argmax(self.q_table[state_key])
    
    def learn(self, state, action, reward, next_state, done, info):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize Q-values if needed
        for key in [state_key, next_state_key]:
            if key not in self.q_table:
                self.q_table[key] = np.random.uniform(-1, 1, size=self.action_space.n)
        
        # Q-learning update with adaptive clipping
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key]) if not done else 0
        td_target = reward + self.discount_factor * max_next_q
        self.q_table[state_key][action] = np.clip(
            current_q + self.learning_rate * (td_target - current_q),
            -10, 10  # Prevent value explosion
        )
        
        # Update performance metrics
        coverage = info['total_covered_cells'] / info['coverable_cells']
        self.reward_history.append(reward)
        self.coverage_history.append(coverage)
        self.danger_history.append(1 if reward < -20 else 0)
        
        # Adaptive parameter adjustment
        if len(self.reward_history) == 100:
            avg_reward = np.mean(self.reward_history)
            avg_coverage = np.mean(self.coverage_history)
            danger_rate = np.mean(self.danger_history)
            
            # Adjust learning rate based on performance
            if avg_reward < -500 or danger_rate > 0.5:
                self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.05)
            elif avg_reward > -100 and danger_rate < 0.2:
                self.learning_rate = max(self.min_learning_rate, self.learning_rate * 0.95)
            
            # Maintain minimum exploration based on coverage
            self.min_exploration = max(0.05, 0.2 * (1 - avg_coverage))
        
        # Update exploration rate (with minimum floor)
        self.exploration_rate = max(self.min_exploration, 
                                   self.exploration_rate * self.exploration_decay)
