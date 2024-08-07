import pygame
import stable_baselines3.common.monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime, timedelta
import game
import gym_dino
import random
import gymnasium as gym
from stable_baselines3 import PPO
from gym_dino.envs.dino_env import DinoCheckpointCallback, plot_metrics

seed = None

random.seed(seed)

env = gym.make('DinoEnv-v0', render_mode='human')

checkpoint_callback = DinoCheckpointCallback(save_freq=100000, save_path='./checkpoints/', name_prefix='dino_model')


def showData():
    data = checkpoint_callback.load_metrics()
    avg_lengths, avg_rewards = checkpoint_callback.parse_metrics(data)
    plot_metrics(avg_lengths, avg_rewards)


def createNewModel(model_name):
    model = PPO("MultiInputPolicy", env, verbose=1, seed=seed)
    model.set_random_seed(seed)
    model.save(model_name)


def trainModel(model_name):
    i = 0

    model = PPO.load(model_name, env)
    model.ent_coef = 0.01

    while True:
        print(f"<< run {i} >>")

        # model = PPO.load(model_name, env)

        model.learn(total_timesteps=10_000, callback=checkpoint_callback)

        model.save(model_name)

        i += 1


def testModel2(model_name):

    model = PPO.load(model_name, env)

    model.learn(total_timesteps=10_000, callback=checkpoint_callback)


def testModel(model_name):
    model = PPO.load(model_name, env)
    obs, info = env.reset()
    for _ in range(1000):
        print(obs)
        action, _states = model.predict(obs)
        obs, reward, done, reachedEnd, info = env.step(action)

        env.render()
        if done:
            obs, info2 = env.reset()


def runHardCoded(min_to_run):

    world = game.Game()

    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=min_to_run)

    while datetime.now() < end_time:

        data = game.gc.getEnvironment()

        bird_distance = data["bird"][0]
        cactus_distance = data["cactus"][0]

        if cactus_distance < 90:
            game.simulate_key_press(pygame.K_SPACE)
            game.simulate_key_release(pygame.K_SPACE)

        if bird_distance < 90:
            game.simulate_key_press(pygame.K_SPACE)
            game.simulate_key_release(pygame.K_SPACE)

        if (game.gc.hasGameEnded):
            game.reset()
            world = game.Game()


        world.play()


def runTillDeath():

    world = game.Game()

    while not game.gc.hasGameEnded:

        world.play()


def runHardCodedTillDeath():

    world = game.Game()

    dist_jump = 92

    while not game.gc.hasGameEnded:

        data = game.gc.getEnvironment()

        bird_distance = data["bird"][0]
        cactus_distance = data["cactus"][0]

        if cactus_distance < dist_jump:
            game.simulate_key_press(pygame.K_SPACE)
            game.simulate_key_release(pygame.K_SPACE)

        if bird_distance < dist_jump:
            game.simulate_key_press(pygame.K_SPACE)
            game.simulate_key_release(pygame.K_SPACE)

        if (game.gc.hasGameEnded):
            game.reset()
            world = game.Game()

        game.gc.getCactusDistances()
        world.play()

#trainModel("ppo_dino7")
#testModel2("checkpoints/dino_model_10240_steps")
#runHardCoded()
#showData()
#trainModel("ppo_dino8")
#createNewModel("ppo_dino8")

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
