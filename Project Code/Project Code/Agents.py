from collections import deque
import numpy as np
import random
import gymnasium as gym
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


class QLearningAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """
        Initialize the Q-learning agent.
        :param action_space: The action space of the environment.
        :param alpha: Learning rate.
        :param gamma: Discount factor for future rewards.
        :param epsilon: Exploration rate for ε-greedy policy.
        :param epsilon_decay: Decay rate for epsilon.
        :param min_epsilon: Minimum value for epsilon to avoid zero exploration.
        """
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Initialize Q-table with zeros
        self.q_table = {}
    
    def get_q_value(self, state, action):
        """
        Get Q-value for a given state-action pair.
        :param state: The current state.
        :param action: The action.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
        return self.q_table[state][action]
    
    def update_q_value(self, state, action, reward, next_state, done):
        """
        Update Q-value using the Q-learning update rule.
        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received.
        :param next_state: The next state after the action.
        :param done: Whether the episode is done.
        """
        # Convert state and next_state to tuples to make them hashable
        state = tuple(state.flatten())  # Flatten the state and convert it to a tuple
        next_state = tuple(next_state.flatten())  # Flatten next_state and convert it to a tuple
        
        # If the next_state is not in the Q-table, initialize it with zeros
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_space.n)
        
        # Find the best next action
        best_next_action = np.argmax(self.q_table.get(next_state, np.zeros(self.action_space.n)))
        
        # Get current Q-value for the state-action pair
        current_q_value = self.get_q_value(state, action)
        
        # Q-learning update rule
        target = reward + (0 if done else self.gamma * self.q_table.get(next_state, np.zeros(self.action_space.n))[best_next_action])
        self.q_table[state][action] = current_q_value + self.alpha * (target - current_q_value)

    
    def select_action(self, state):
        """
        Select an action using an epsilon-greedy policy.
        :param state: The current state.
        :return: The selected action.
        """
        # Convert state to tuple to make it hashable
        state = tuple(state.flatten())  # Flatten the state and convert it to a tuple
        
        # If state is not in the Q-table, initialize it with zeros
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
        
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            # Explore: Select a random action
            return np.random.randint(self.action_space.n)
        else:
            # Exploit: Select the action with the highest Q-value
            return np.argmax(self.q_table[state])

    
    def decay_epsilon(self):
        """
        Decay epsilon to reduce exploration over time.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)





