import os
import json
import time

import pygame
import stable_baselines3.common.monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import game
import gym_dino
import random
import gymnasium as gym
from multiprocessing import Process, Manager
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import wrapper_data
from gym_dino.envs.dino_env import DinoCheckpointCallback, plot_metrics

seed = 12343214

# 0
# 1
# 203853699

# random.seed(seed)


env = gym.make('DinoEnv-v0', render_mode='human')


def showData(model_name):
    checkpoint_callback = DinoCheckpointCallback(save_freq=100000, save_path='./checkpoints/', name_prefix=model_name)
    data = checkpoint_callback.load_metrics()
    avg_lengths, avg_rewards = checkpoint_callback.parse_metrics(data)
    plot_metrics(avg_lengths, avg_rewards)


def createNewModel(model_name):
    # model = PPO("MultiInputPolicy", env, verbose=1, seed=seed)

    model = PPO(
        policy="MultiInputPolicy",  # or "MultiInputPolicy" if you're using multiple input types
        env=env,
        verbose=10_000
    )

    """
        seed=seed,
        n_steps=32,
        batch_size=256,
        gae_lambda=0.8,
        gamma=0.98,
        n_epochs=20,
        ent_coef=0.01,
    """

    # model.set_random_seed(seed)
    model.save(model_name)


def trainModel(model_name):
    checkpoint_callback = DinoCheckpointCallback(save_freq=100000, save_path='./checkpoints/', name_prefix=model_name)

    model = PPO.load(model_name, env)

    while True:
        model.learn(total_timesteps=10_000, callback=checkpoint_callback)

        # model.save(model_name)


def testModel(model_name):
    checkpoint_callback = DinoCheckpointCallback(save_freq=100000, save_path='./checkpoints/', name_prefix=model_name)

    model = PPO.load(model_name, env)

    model.learn(total_timesteps=10_000, callback=checkpoint_callback)


def saveMetrics(currentHighest):

    json_path = os.path.join('./checkpoints/', 'metrics.json')

    time_stamp = time.strftime("%Y%m%d%H%M%S")

    new_data = {
        f'{time_stamp}-average_length': currentHighest
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


def reload_shared_metrics():

    json_path = os.path.join('./checkpoints/', 'shared_metrics.json')

    data = {
        "ppo_dino19-average_length": 0,
        "ppo_dino19-average_reward": 0,
        "ppo_dino16-average_length": 64,
        "ppo_dino16-average_reward": 64,
        "ppo_dino20-average_length": 0,
        "ppo_dino20-average_reward": 0,
        "ppo_dino18-average_length": 0,
        "ppo_dino18-average_reward": 0,
        "ppo_dino17-average_length": 0,
        "ppo_dino17-average_reward": 0,
        "ppo_dino15-average_length": 0,
        "ppo_dino15-average_reward": 0,
    }

    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)

### EVOLUTION PLAN ###

def createSpecies():
    createNewModel("ppo_dino15")
    createNewModel("ppo_dino16")
    createNewModel("ppo_dino17")
    createNewModel("ppo_dino18")
    createNewModel("ppo_dino19")
    createNewModel("ppo_dino20")


def addEvolution(chosen_model, model_name):
    checkpoint_callback = DinoCheckpointCallback(save_freq=int(1e10), save_path='./checkpoints/', name_prefix=model_name)

    model = PPO.load(chosen_model, env)

    model.learn(total_timesteps=10_000, callback=checkpoint_callback)

    model.save(model_name)

def graphData():
    pass

def main():

    latestHighestScore = 1

    processes = []

    while True:

        model_names = ["ppo_dino15", "ppo_dino16", "ppo_dino17", "ppo_dino18", "ppo_dino19", "ppo_dino20"]

        values = {}

        with open('checkpoints/shared_metrics.json', 'r') as file:
            data = json.load(file)

        def get_average_length(m):
            key = f"{m}-average_length"
            return data.get(key, "Model not found")

        for model_name in model_names:
            values[model_name] = get_average_length(model_name)

        # use random model for now
        chosen_model = model_names[-1]

        try:
            chosen_model = max(values, key=values.get)
        except TypeError as e:
            reload_shared_metrics()

        saveMetrics(values[chosen_model])


        # make sure that the model is safe to use (20%)
        if ((latestHighestScore-values[chosen_model])/latestHighestScore) > 0.5:
            # not safe to use, use latest save
            chosen_model = "ppo_dino_highestSaved"
        else:
            latestHighestScore = values[chosen_model]
            model = PPO.load(chosen_model, env)
            model.save("ppo_dino_highestSaved")

        print("Chosen model is " + chosen_model)

        for model_name in model_names:
            p = Process(target=addEvolution,
                        args=(chosen_model, model_name))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()


if __name__ == '__main__':
    main()

"""
    model.policy = d["test"]
    model = PPO.load(chosen_model, env)
    d["test"] = model.policy
    manager = Manager()
    d = manager.dict()
"""
# Can't load and save models too many times

env.close()

# it stopped jumping
# stops when reaches 42
# episode reward vs episode length
# cnn
# specific reward improves training
# also give game speed


# smooth data OUT
# 1. retrain with set seed and see performance
# train for two days to see if the bird distance is a issue
# 2. if it is a issue, retrain model with x, y bird locations
# 3. add more locations
# 4. analyze model to tell what it is struggling with
# - could be calculating distance incorrectly
# - print out inputs to see if it makes sense
# 5. compare score of hard coded dino vs ai dino


# how good performance
# diff factors performance

# vectorize environment
# optimize values
