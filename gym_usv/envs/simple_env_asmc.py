import gymnasium as gym
from gymnasium import spaces
from gym_usv.envs import UsvSimpleEnv
from gym_usv.control import UsvAsmc
import numpy as np

class UsvSimpleASMCEnv(UsvSimpleEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.asmc = UsvAsmc()

    def reset(self, seed=None, options=None):
        self.asmc = UsvAsmc()
        return super().reset(seed=seed)

    def step(self, action):
        self.position, self.velocity, _ = self.asmc.compute(
            action,
            self.position,
            self.velocity,
            False
        )

        return super().step(np.zeros(2))

    def render(self):
        return super().render()

    def close(self):
        super().close()