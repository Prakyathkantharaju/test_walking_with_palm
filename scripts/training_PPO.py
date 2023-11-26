import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


from gymnasium.envs.registration import register


register(
    id="QuadWalker-V0",
    entry_point="envs.walk_straight:WalkerStraight",
)


# env = gym.make("QuadWalker-V0")
# print(env.reset())

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("QuadWalker-V0", n_envs=10)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=250000000)

model.save("save/ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("save/ppo_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    print(info['com'])
    #vec_env.render("")
