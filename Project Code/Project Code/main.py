from stable_baselines3 import DQN
import gymnasium as gym
import time
import coverage_gridworld
from stable_baselines3.common.env_util import make_vec_env
from coverage_gridworld.custom import observation_space, observation, reward
import json, os


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

test_path = "./test_2_3"
os.makedirs(test_path, exist_ok=True)



def evaluate(file):
    trials = 10
    map_names = [
        "just_go",
        "safe",
        "maze",
        "chokepoint",
        "sneaky_enemies"
    ]
    envs = {
        "just_go": gym.make("just_go"),
        "safe": gym.make("safe"),
        "maze": gym.make("maze"),
        "chokepoint": gym.make("chokepoint"),
        "sneaky_enemies": gym.make("sneaky_enemies")
    }
    scores = {
        "just_go": 0,
        "safe": 0,
        "maze": 0,
        "chokepoint": 0,
        "sneaky_enemies": 0,
        "total": 0
    }
    model = DQN.load(file)
    for map in map_names:
        for trial in range(trials):
            env = envs[map]
            obs, _ = env.reset()
            terminated = False
            coverage = 0
            while not terminated:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, terminated, truncated, info = env.step(action)
                coverage = info["total_covered_cells"]/info["coverable_cells"]
            scores[map] += coverage/trials
            scores["total"] += coverage/trials
    return scores

from stable_baselines3.common.callbacks import BaseCallback

class PauseAndEvalCallback(BaseCallback):
    def __init__(self, test_path, eval_freq=10_000, verbose=0):
        super().__init__(verbose)
        self.eval_envs = {
            "just_go": gym.make("just_go"),
            "safe": gym.make("safe"),
            "maze": gym.make("maze"),
            "chokepoint": gym.make("chokepoint"),
            "sneaky_enemies": gym.make("sneaky_enemies")
        }
        self.eval_freq = eval_freq
        self.score_hist = {
            "just_go": [],
            "safe": [],
            "maze": [],
            "chokepoint": [],
            "sneaky_enemies": [],
            "total": []
        }
        self.test_path = test_path

    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            scores = self.evaluate_policy()
            for key in self.score_hist:
                self.score_hist[key].append(scores[key])
            with open(self.test_path+f'/scores.json', "w+") as f:
                json.dump(self.score_hist, f, indent=4)
            self.model.save(self.test_path + "/model")
        return True  # Don't stop training

    def evaluate_policy(self):
        trials = 10
        map_names = [
            "just_go",
            "safe",
            "maze",
            "chokepoint",
            "sneaky_enemies"
        ]
        envs = self.eval_envs
        scores = {
            "just_go": 0,
            "safe": 0,
            "maze": 0,
            "chokepoint": 0,
            "sneaky_enemies": 0,
            "total": 0
        }
        model = self.model
        for map in map_names:
            for trial in range(trials):
                env = envs[map]
                obs, _ = env.reset()
                terminated = False
                coverage = 0
                while not terminated:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, rewards, terminated, truncated, info = env.step(action)
                    coverage = info["total_covered_cells"]/info["coverable_cells"]
                scores[map] += coverage/trials
                scores["total"] += coverage/trials
        return scores

# Parallel environments for training
do_training = True
if do_training:
    vec_env = make_vec_env("standard", n_envs=32)
    #print(vec_env.observation_space)
    model = DQN("MlpPolicy", vec_env, learning_rate=0.0005, learning_starts=10000, gamma=0.95, exploration_fraction=0.9, exploration_final_eps=0.1, verbose=1, tensorboard_log="./tensorboard/")
    total_timesteps = 10000000
    pause_eval_callback = PauseAndEvalCallback(test_path, eval_freq=total_timesteps//1000)
    model.learn(total_timesteps=total_timesteps, callback=pause_eval_callback, tb_log_name="dqn_run")
    model.save(test_path + "/model")

scores = evaluate(test_path + "/model")
with open(test_path+f'/scores_final.json', "w+") as f:
    json.dump(scores, f, indent=4)


# Single environment for testing
def demo():
    env = gym.make("standard", render_mode="human")

    model = DQN.load("MlpTest3")
    obs, _ = env.reset()
    time.sleep(1)
    obs, _ = env.reset()
    time.sleep(1)
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
