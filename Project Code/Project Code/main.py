import random
import time
import gymnasium
import coverage_gridworld  # must be imported, even though it's not directly referenced
import numpy as np


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

env = gymnasium.make("sneaky_enemies", predefined_map_list=None, activate_game_status=True)
# env = gymnasium.make("sneaky_enemies", render_mode="human", predefined_map_list=None, activate_game_status=True)
num_episodes = 1000

class AdaptiveQLearningAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.q_table = {}
        # Initial learning parameters
        self.initial_alpha = 0.5  # Start with high learning rate
        self.min_alpha = 0.01     # Minimum learning rate
        self.alpha = self.initial_alpha
        self.gamma = 0.95         # Discount factor
        self.initial_epsilon = 1.0  # Start with high exploration
        self.min_epsilon = 0.01    # Minimum exploration rate
        self.epsilon = self.initial_epsilon
        self.reward_history = []   # Track rewards for adaptation
        self.window_size = 100     # Number of episodes to consider for adaptation
        
    def get_action(self, state):
        state_key = str(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
            
        if random.random() < self.epsilon:  # Explore
            return self.action_space.sample()
        else:  # Exploit
            return np.argmax(self.q_table[state_key])
            
    def learn(self, state, action, reward, next_state):
        state_key = str(state)
        next_state_key = str(next_state)
        
        # Initialize Q-values if not present
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space.n)
            
        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
        
    def update_parameters(self, episode_reward):
        # Update reward history
        self.reward_history.append(episode_reward)
        if len(self.reward_history) > self.window_size:
            self.reward_history.pop(0)
            
        # Calculate average reward over the window
        if len(self.reward_history) == self.window_size:
            avg_reward = np.mean(self.reward_history)
            # Normalize between 0 and 1 based on expected reward range
            normalized_reward = (avg_reward + 100) / 200  # Assuming reward range [-100, 100]
            normalized_reward = np.clip(normalized_reward, 0, 1)
            
            # Adjust learning rate - decrease as performance improves
            self.alpha = self.min_alpha + (self.initial_alpha - self.min_alpha) * (1 - normalized_reward)
            
            # Adjust exploration rate - decrease as we learn more
            self.epsilon = self.min_epsilon + (self.initial_epsilon - self.min_epsilon) * (1 - normalized_reward)

# Initialize the improved agent
agent = AdaptiveQLearningAgent(env.action_space)

# Training loop
for i in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        
    # Update learning parameters after each episode
    agent.update_parameters(total_reward)
    
    # Print progress
    if (i+1) % 50 == 0:
        avg_reward = np.mean(agent.reward_history[-50:]) if len(agent.reward_history) >= 50 else total_reward
        print(f"Episode {i+1}, Total Reward: {total_reward:.1f}, "
              f"Avg Reward (last 50): {avg_reward:.1f}, "
              f"Alpha: {agent.alpha:.3f}, Epsilon: {agent.epsilon:.3f}")