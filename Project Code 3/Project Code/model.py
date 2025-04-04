from stable_baselines3 import DQN
import gymnasium as gym
import time
import coverage_gridworld
from stable_baselines3.common.env_util import make_vec_env
from coverage_gridworld.custom import observation_space, observation, reward



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

#env = gym.make("standard", predefined_map_list=maps)

#env = gym.make("sneaky_enemies", render_mode="human", predefined_map_list=None, activate_game_status=True)
# print("Env class chain:")
# while hasattr(env, "env"):
#     print("→", type(env))
#     env = env.env
# print("→", type(env)) 
# print(type(env.unwrapped))
# print((env.unwrapped.grid + 255).flatten())
# print("one: ", gym.spaces.MultiDiscrete((env.unwrapped.grid + 255).flatten()))
# print("two: ", observation_space(env.unwrapped))


# Parallel environments for training
USE_CNN = True
do_training = True
if USE_CNN:
    import torch as th
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class TinyCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 32):
            super().__init__(observation_space, features_dim)
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),  # (3, 10, 10) → (16, 10, 10)
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (16, 10, 10) → (32, 10, 10)
                nn.ReLU(),
                nn.Flatten()  # 32*10*10 = 3200
            )

            # Final feature dimension
            self.linear = nn.Sequential(
                nn.Linear(32 * 10 * 10, features_dim),
                nn.ReLU()
            )

        def forward(self, obs: th.Tensor) -> th.Tensor:
            return self.linear(self.cnn(obs))
    policy_kwargs = dict(
        features_extractor_class=TinyCNN,
        features_extractor_kwargs=dict(features_dim=32),
    )


    if do_training:
        vec_env = make_vec_env("all_maps", n_envs=32)
        #print(vec_env.observation_space)
        model = DQN("CnnPolicy", vec_env, policy_kwargs=policy_kwargs, learning_rate=0.0005, learning_starts=10000, gamma=0.95, exploration_fraction=0.8, exploration_final_eps=0.1, verbose=1, tensorboard_log="./tensorboard/")
        model.learn(total_timesteps=50000000, tb_log_name="dqn_run")
        model.save("CnnTest1")
elif USE_CNN == False:
    if do_training:
        vec_env = make_vec_env("all_maps", n_envs=32)
        #print(vec_env.observation_space)
        model = DQN("MlpPolicy", vec_env, learning_rate=0.0005, learning_starts=10000, gamma=0.95, exploration_fraction=0.8, exploration_final_eps=0.1, verbose=1, tensorboard_log="./tensorboard/")
        model.learn(total_timesteps=50000000, tb_log_name="dqn_run")
        model.save("MlpTest1")



# Single environment for testing
env = gym.make("chokepoint", render_mode="human")
if USE_CNN:
    model = DQN.load("CnnTest1")
else:
    model = DQN.load("MlpTest1")
obs, _ = env.reset()
terminated = False
total_reward = 0
while not terminated:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    total_reward += rewards
    time.sleep(0.1)

print(f"Total reward: {total_reward}")
env.close()