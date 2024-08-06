from gymnasium.envs.registration import register

register(
     id="DinoEnv-v0",
     entry_point="gym_dino.envs:DinoEnv",
     max_episode_steps=None,
)