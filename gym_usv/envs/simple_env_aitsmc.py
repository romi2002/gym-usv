import gymnasium as gym
from gymnasium import spaces
from gym_usv.envs import UsvSimpleEnv
import usv_libs_py as usv
import numpy as np

class UsvSimpleAITSMCEnv(UsvSimpleEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.model = usv.model.DynamicModel()
        self.aitsmc = usv.controller.AITSMC(usv.controller.AITSMC.defaultParams())

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        self.model = usv.model.DynamicModel(self.position[0], self.position[1], self.position[2])
        self.aitsmc = usv.controller.AITSMC(usv.controller.AITSMC.defaultParams())
        return obs, info

    def step(self, action):
        left_thruster = 0
        right_thruster = 0
        for _ in range(2):
            state = usv.utils.from_model(self.model)
            setpoint = usv.controller.AITSMCSetpoint()
            setpoint.u = action[0] * 0.2
            setpoint.r = action[1] * 0.5

            # TODO
            setpoint.dot_u = 0
            setpoint.dot_r = 0

            out = self.aitsmc.update(state, setpoint)
            model_out = self.model.update(out.left_thruster, out.right_thruster)
            left_thruster, right_thruster = out.left_thruster, out.right_thruster
            self.position = np.hstack((
                model_out.pose_x,
                model_out.pose_y,
                model_out.pose_psi
            ))

            self.velocity = np.hstack((
                model_out.u,
                model_out.v,
                model_out.r
            ))

        obs, reward, terminated, truncated, info = super().step(np.zeros(2))
        debug_data = self.aitsmc.getDebugData()
        info['left_thruster'] = left_thruster
        info['right_thruster'] = right_thruster
        info['e_u'] = debug_data.e_u
        info['e_r'] = debug_data.e_r
        info['Ka_u'] = debug_data.Ka_u
        info['Ka_r'] = debug_data.Ka_r

        info['action0'] = action[0]
        info['action1'] = action[1]
        return obs, reward, terminated, truncated, info

    def render(self):
        return super().render()

    def close(self):
        super().close()