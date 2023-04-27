import gymnasium as gym
from gymnasium import spaces
from gym_usv.envs import UsvSimpleEnv
import usv_libs_py as usv
import numpy as np
from scipy import signal
from gym_usv.utils.live_filter import LiveLFilter

class UsvSimpleAITSMCEnv(UsvSimpleEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.model = usv.model.DynamicModel()
        self.aitsmc = usv.controller.AITSMC(usv.controller.AITSMC.defaultParams())

        # Filter setup
        # self.window_size = 10
        # self.window = np.zeros((2, self.window_size))
        # self.window_pos = 0
        r_b, r_a = signal.iirfilter(4, Wn=0.25, fs=100, btype="low", ftype="butter")
        self.filter_r = LiveLFilter(r_b, r_a)

        u_b, u_a = signal.iirfilter(4, Wn=0.25, fs=100, btype="low", ftype="butter")
        self.filter_u = LiveLFilter(u_b, u_a)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        self.model = usv.model.DynamicModel(self.position[0], self.position[1], self.position[2])
        self.aitsmc = usv.controller.AITSMC(usv.controller.AITSMC.defaultParams())
        return obs, info

    def filter_action(self, action):
        # Update window and window_pos
        # self.window[:, self.window_pos] = action
        # self.window_pos += 1
        # if self.window_pos > self.window_size:
        #     self.window_pos = 0

        # Compute new action with filter
        setpoint = usv.controller.AITSMCSetpoint()
        setpoint.r = self.filter_r(action[1])
        setpoint.u = self.filter_u(action[0])
        return setpoint

    def step(self, action):
        left_thruster = 0
        right_thruster = 0
        last_u, last_r = 0, 0

        for _ in range(10):
            state = usv.utils.from_model(self.model)

            # TODO
            setpoint = self.filter_action(action)
            last_u = setpoint.u
            last_r = setpoint.r
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
        info['setpoint_u'] = last_u
        info['setpoint_r'] = last_r

        return obs, reward, terminated, truncated, info

    def render(self):
        return super().render()

    def close(self):
        super().close()