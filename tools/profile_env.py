import gymnasium as gym
import gym_usv
import numpy as np

env = gym.make('usv-simple')
env.reset()
for _ in range(10000):
    env.step(np.zeros(2))