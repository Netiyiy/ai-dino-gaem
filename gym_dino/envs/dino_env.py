import game
import pygame
import time
import json
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import spaces


class DinoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):

        print("Hi! I'm dino and I am running")

        self.world = game.Game()
        #self.b = game.Game()
        #self.b.play(False, False, False)

        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, np.inf, shape=(2,), dtype=int),
                "bird": spaces.Box(0, np.inf, shape=(1,), dtype=int),
                "cactus": spaces.Box(0, np.inf, shape=(1,), dtype=int),
            }
        )

        self.action_space = spaces.Discrete(4)

        self._action_to_keyboard = {
            0: "nothing",
            1: "jump",
            2: "crouch",
            3: "uncrouch"
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        obs = self.world.gc.getEnvironment()
        #cactus_dist = obs.get("cactus")
        #print(cactus_dist)
        return obs

    def _get_info(self):
        return {"score": self.world.gc.score}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        #self.world = game.Game()
        self.world.reset()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            # self._render_frame()
            self.world.play(False, False, False)

        return observation, info

    def step(self, action):

        doPrint = False

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        key = self._action_to_keyboard[action]

        penalty = 0

        KEY_SPACE = False
        KEY_DOWN = False
        KEY_UP = False

        if key == "jump":
            KEY_SPACE = True
            if doPrint:
                print("Jumped")

            # reasonable = False
            # for cactus in game.gc.cacti:
            #    if cactus.getX() > game.gc.dino.getX():
            # print(abs(cactus.getX() - game.gc.dino.getX()))
            # if abs(cactus.getX() - game.gc.dino.getX()) < 200:
            #    reasonable = True
            #    penalty += -2
            # if not reasonable:
            # penalty += 2
            # print(reasonable)
        elif key == "crouch":
            KEY_DOWN = True
            if doPrint:
                print("Crouched")
        elif key == "uncrouch":
            KEY_UP = True
            if doPrint:
                print("Uncrouched")
            # penalty += 0.1
        elif key == "nothing":
            KEY_UP = True
            # penalty += -0.5
            # game.simulate_key_release(pygame.K_DOWN)
            if doPrint:
                print("Nothing")

        # constant

        # self.world.currentScore
        reward = 9 * self.world.gc.successfulJumps + 1
        self.world.gc.successfulJumps = 0
        terminated = self.world.gc.hasGameEnded
        if terminated:
            reward = -10
        # reward = (1 + game.gc.successfulJumps) if terminated else -1  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            # self._render_frame()
            self.world.play(KEY_SPACE, KEY_DOWN, KEY_UP)
            #self.b.play(KEY_SPACE, KEY_DOWN, KEY_UP)

        return observation, reward, terminated, False, info


def plot_metrics(avg_lengths, avg_rewards):

    print(sum(avg_lengths))

    avg_lengths = [sum(avg_lengths[max(i - 50, 0):min(i + 51, len(avg_lengths))]) / len(
        avg_lengths[max(i - 50, 0):min(i + 51, len(avg_lengths))]) for i in range(len(avg_lengths))]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(avg_lengths[:15000], marker='o', markersize=3)
    plt.title('Average Episode Lengths')
    plt.xlabel('Timestamp')
    plt.ylabel('Average Length')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    plt.plot(avg_rewards, marker='o')
    plt.title('Average Episode Rewards')
    plt.xlabel('Timestamp')
    plt.ylabel('Average Reward')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    print("Showing graph")


class DinoCheckpointCallback(CheckpointCallback):

    def __init__(self, save_freq, save_path, name_prefix='model', verbose=0):
        super(DinoCheckpointCallback, self).__init__(save_freq, save_path, name_prefix, verbose)
        self.episode_lengths = []
        self.episode_rewards = []
        self.average_rewards = -1
        self.average_lengths = -1

    def _save_metrics(self):

        if self.average_rewards == -1 or self.average_lengths == -1:
            return

        json_path = os.path.join(self.save_path, 'metrics.json')

        time_stamp = time.strftime("%Y%m%d%H%M%S")

        new_data = {
            f'{time_stamp}-average_length': self.average_lengths,
            f'{time_stamp}-average_reward': self.average_rewards
        }

        data = {}

        if os.path.exists(json_path):
            with open(json_path, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = {}

        data.update(new_data)

        with open(json_path, 'w') as file:
            json.dump(data, file, indent=4)

        self.average_lengths = -1
        self.average_rewards = -1
        self.episode_lengths = []
        self.episode_rewards = []

    def load_metrics(self):

        json_path = os.path.join(self.save_path, 'metrics.json')

        if os.path.exists(json_path):
            with open(json_path, 'r') as file:
                try:
                    data = json.load(file)
                    return data
                except json.JSONDecodeError:
                    print("Error reading JSON file.")
                    return {}
        else:
            print("File not found.")
            return {}

    def parse_metrics(self, data):
        avg_lengths = []
        avg_rewards = []

        for key, value in data.items():
            timestamp, metric = key.split('-')
            if metric == 'average_length':
                avg_lengths.append(value)
            elif metric == 'average_reward':
                avg_rewards.append(value)

        return avg_lengths, avg_rewards

    def _on_training_end(self):
        self.saveData()

    def _on_rollout_end(self):
        self.saveData()

    def saveData(self):
        if len(self.episode_rewards) > 0 and len(self.episode_lengths) > 0:
            self.average_rewards = np.average(self.episode_rewards)
            self.average_lengths = np.average(self.episode_lengths)
        self._save_metrics()

    def _on_step(self) -> bool:
        if self.locals['dones']:
            episode_rewards = self.locals['infos'][0].get('episode', {}).get('r', 0)
            episode_lengths = self.locals['infos'][0].get('episode', {}).get('l', 0)
            self.episode_rewards.append(episode_rewards)
            self.episode_lengths.append(episode_lengths)
        return super(DinoCheckpointCallback, self)._on_step()
