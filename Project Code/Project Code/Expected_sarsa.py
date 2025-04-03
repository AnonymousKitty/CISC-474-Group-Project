import random
import numpy as np
from collections import deque

class ExpectedSARSAAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.q_table = {}
        
        self.learning_rate = 0.7
        self.min_learning_rate = 0.01
        self.discount_factor = 0.95
        self.exploration_rate = 1.0
        self.min_exploration = 0.01
        self.exploration_decay = 0.9995
        
        self.reward_memory = deque(maxlen=100)
        self.best_avg_reward = -float('inf')
    
    def get_state_key(self, state):
        grid, agent_pos, enemies = state
        agent_y, agent_x = divmod(agent_pos, 10)
        nearby_cells = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                y, x = agent_y + dy, agent_x + dx
                if 0 <= y < 10 and 0 <= x < 10:
                    nearby_cells.append(grid[y * 10 + x])
                else:
                    nearby_cells.append(0)
        enemy_distances = [abs(agent_pos - e) for e in enemies if e != 0]
        closest_enemies = sorted(enemy_distances)[:2] if enemy_distances else [0, 0]
        return (tuple(nearby_cells), agent_pos, tuple(closest_enemies))
    
    def get_action(self, state):
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.random.uniform(-0.1, 0.1, size=self.action_space.n)
        if random.random() < self.exploration_rate:
            return random.choice([a for a in range(self.action_space.n) if a != 4])
        return np.argmax(self.q_table[state_key])
    
    def learn(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.random.uniform(-0.1, 0.1, size=self.action_space.n)
        
        next_q_values = self.q_table[next_state_key]
        policy = np.ones(self.action_space.n) * (self.exploration_rate / self.action_space.n)
        best_action = np.argmax(next_q_values)
        policy[best_action] += (1 - self.exploration_rate)
        expected_q = np.sum(policy * next_q_values)
        
        target = reward + self.discount_factor * expected_q * (not done)
        self.q_table[state_key][action] += self.learning_rate * (target - self.q_table[state_key][action])
        
        self.reward_memory.append(reward)
        if len(self.reward_memory) == self.reward_memory.maxlen:
            current_avg = np.mean(self.reward_memory)
            if current_avg > self.best_avg_reward:
                self.best_avg_reward = current_avg
                self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
        
        self.learning_rate = max(self.min_learning_rate, self.learning_rate * (0.999 if reward > 0 else 0.99))
