import random
import time
import gymnasium
import coverage_gridworld  # must be imported, even though it's not directly referenced
import numpy as np
from itertools import cycle  # Add this import
from collections import deque
import math
from Q_learning import Q_LearningAgent
from Expected_sarsa import ExpectedSARSAAgent
from Double_Q import DoubleQLearningAgent
import sys

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
    elif input_action.lower() == "q":
        env.close()
        sys.exit(0) 
    else:
        return 4


def random_player():
    return random.randint(0, 4)


maps = [
    # [
    #     [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # ],
    # [
    #     [3, 0, 0, 2, 0, 2, 0, 0, 0, 0],
    #     [0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
    #     [0, 2, 0, 2, 2, 2, 2, 2, 2, 0],
    #     [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
    #     [0, 2, 0, 2, 0, 2, 0, 0, 2, 0],
    #     [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
    #     [0, 2, 2, 2, 0, 0, 0, 2, 0, 0],
    #     [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
    #     [0, 2, 0, 2, 0, 2, 2, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    # ],
    # [
    #     [3, 2, 0, 0, 0, 0, 2, 0, 0, 0],
    #     [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
    #     [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    #     [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
    #     [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    #     [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
    #     [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    #     [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
    #     [0, 2, 0, 2, 0, 4, 2, 4, 0, 0],
    #     [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
    # ],
    # [
    #     [3, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    #     [0, 0, 2, 0, 0, 0, 0, 0, 0, 4],
    #     [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    #     [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    #     [0, 4, 2, 0, 0, 0, 0, 2, 0, 0],
    #     [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    #     [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    #     [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    #     [0, 0, 0, 0, 4, 0, 4, 2, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
    # ],
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
# Initialize with all 5 maps
env = gymnasium.make("sneaky_enemies", predefined_map_list=maps, activate_game_status=True)
agent = Q_LearningAgent(env.action_space)
# agent = DoubleQLearningAgent(env.action_space)  # Use Double Q-Learning agent
# agent = ExpectedSARSAAgent(env.action_space)  # Use Expected SARSA agent
performance_history = []
coverage_history = []

for episode in range(1, 1001):
    try:
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_coverage = 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, info = env.step(action)
            
            # Enhance info dict
            info["agent_moved"] = action != 4
            info["last_action"] = action
            
            agent.learn(state, action, reward, next_state, done, info)
            state = next_state
            episode_reward += reward
            episode_coverage = info['total_covered_cells'] / info['coverable_cells']
            
            # Early termination if stuck
            if info['steps_remaining'] < 100 and episode_coverage < 0.1:
                break
                
        # Progress reporting
        if episode % 10 == 0:
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:7.1f} | "
                  f"Coverage: {episode_coverage*100:5.1f}% | "
                  f"ε: {agent.exploration_rate:.3f} | "
                  f"α: {agent.learning_rate:.3f}")
            
    except KeyboardInterrupt:
        print("\nTraining stopped by user")
        break

env.close()