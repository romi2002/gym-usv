"""
@author: Alejandro Gonzalez, Ivana Collado, Sebastian
        Perez

Environment of an Unmanned Surface Vehicle with an
Adaptive Sliding Mode Controller to train collision
avoidance on the OpenAI Gym library.
"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from collections import defaultdict
from functools import lru_cache


class UsvAsmcCaEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self,config=None):
        # Integral step (or derivative) for 100 Hz
        self.integral_step = 0.01

        # USV model coefficients
        self.X_u_dot = -2.25
        self.Y_v_dot = -23.13
        self.Y_r_dot = -1.31
        self.N_v_dot = -16.41
        self.N_r_dot = -2.79
        self.Xuu = 0
        self.Yvv = -99.99
        self.Yvr = -5.49
        self.Yrv = -5.49
        self.Yrr = -8.8
        self.Nvv = -5.49
        self.Nvr = -8.8
        self.Nrv = -8.8
        self.Nrr = -3.49
        self.m = 30
        self.Iz = 4.1
        self.B = 0.41
        self.c = 0.78

        # ASMC gains
        self.k_u = 0.1
        self.k_psi = 0.2
        self.kmin_u = 0.05
        self.kmin_psi = 0.2
        self.k2_u = 0.02
        self.k2_psi = 0.1
        self.mu_u = 0.05
        self.mu_psi = 0.1
        self.lambda_u = 0.001
        self.lambda_psi = 1

        # Second order filter gains (for r_d)
        self.f1 = 2.
        self.f2 = 2.
        self.f3 = 2.

        # Overall vector variables
        self.state = None
        self.position = None
        self.aux_vars = None
        self.last = None
        self.target = None
        self.so_filter = None

        # Obstacle variables
        self.num_obs = None
        self.posx = None  # array
        self.posy = None  # array
        self.radius = None  # array

        # Sensor vector column 0 = senor angle column 1 = distance mesured
        self.sensor_num = np.int(225)
        self.sensors = np.zeros((self.sensor_num, 2))
        self.sensor_span = (2 / 3) * (2 * np.pi)
        self.lidar_resolution = self.sensor_span / self.sensor_num  # angle resolution in radians
        self.sector_num = 25  # number of sectors
        self.sector_size = np.int(self.sensor_num / self.sector_num)  # number of points per sector
        self.sensor_max_range = 10.0  # m
        self.last_reward = 0

        # Boat radius
        self.boat_radius = 0.5
        self.safety_radius = 0.3
        self.safety_distance = 0.1

        # Map limits in meters
        self.max_y = 10
        self.min_y = -10
        self.max_x = 30
        self.min_x = -10

        # Variable for the visualizer
        self.viewer = None

        # Min and max actions
        # velocity 
        self.min_action0 = 0.25
        self.max_action0 = 1.0
        # angle (change to -pi and pi if necessary)
        self.min_action1 = -np.pi / 2
        self.max_action1 = np.pi / 2

        # Reward associated functions anf gains
        self.w_chi = 2.60 # Course direction error
        self.w_ye = 1.35
        self.k_ye = 0.25 # Crosstracking reward

        self.k_uu = 2.0 # Velocity Reward
        self.w_u = 1 # Velocity reward

        self.gamma_theta = 4.0  # 4.0
        self.gamma_x = 0.05  # 0.005
        self.epsilon = 3.0
        self.lambda_reward = 0.85

        self.w_action0 = 0.2
        self.w_action1 = 0.2
        # Action gradual change reward
        self.c_action0 = 1. / np.power((self.max_action0 / 2 - self.min_action0 / 2) / self.integral_step, 2)
        self.c_action1 = 1. / np.power((self.max_action1 / 2 - self.min_action1 / 2) / self.integral_step, 2)
        self.k_action0 = 0
        self.k_action1 = 0

        # Min and max values of the state
        self.min_u = -1.5
        self.max_u = 1.5
        self.min_v = -1.0
        self.max_v = 1.0
        self.min_r = -4.
        self.max_r = 4.
        self.min_ye = -20.
        self.max_ye = 20.
        self.min_ye_dot = -1.5
        self.max_ye_dot = 1.5
        self.min_chi_ak = -np.pi
        self.max_chi_ak = np.pi
        self.min_u_ref = 0.4
        self.max_u_ref = 1.4
        self.min_sectors = np.zeros((self.sector_num))
        self.max_sectors = np.full((self.sector_num), 1)
        self.sectors = np.zeros((self.sector_num))

        # Min and max state vectors
        self.low_state = np.hstack((np.full(34, -1.0), -20))
        self.high_state = np.hstack((np.full(34, 1.0), 0))

        self.min_action = np.array([-1.0, -1.0])
        self.max_action = np.array([1.0, 1.0])

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

        self.screen = None
        self.clock = None
        self.isopen = True
        self.total_reward = 0

    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _normalize_val(self, x, in_min, in_max):
        return self._map(x, in_min, in_max, -1, 1)

    def _denormalize_val(self, x, out_min, out_max):
        return self._map(x, -1, 1, out_min, out_max)

    def _map(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def _denormalize_state(self, state):
        u, v, r, ye, ye_dot, chi_ak, u_ref, sectors, action0_last, action1_last = state[0], state[1], state[2], state[
            3], state[4], state[5], state[6], state[7:7 + self.sector_num], state[7 + self.sector_num], state[
                                                                                      7 + self.sector_num + 1]

        u = self._denormalize_val(u, self.min_u, self.max_u)
        v = self._denormalize_val(v, self.min_v, self.max_v)
        r = self._denormalize_val(r, self.min_r, self.max_r)
        ye = self._denormalize_val(ye, self.min_ye, self.max_ye)
        ye_dot = self._denormalize_val(ye_dot, self.min_ye_dot, self.max_ye_dot)
        chi_ak = self._denormalize_val(chi_ak, self.min_chi_ak, self.max_chi_ak)
        u_ref = self._denormalize_val(u_ref, self.min_u, self.max_u)

        # sectors are ignored
        action0_last = self._denormalize_val(action0_last, self.min_action0, self.max_action0)
        action1_last = self._denormalize_val(action1_last, self.min_action1, self.max_action1)

        state = np.hstack(
            (u, v, r, ye, ye_dot, chi_ak, u_ref, sectors, action0_last, action1_last, state[7 + self.sector_num + 2:]))

        return state

    def _normalize_state(self, state):
        u, v, r, ye, ye_dot, chi_ak, u_ref, sectors, action0_last, action1_last = state[0], state[1], state[2], state[
            3], state[4], state[5], state[6], state[7:7 + self.sector_num], state[7 + self.sector_num], state[
                                                                                      7 + self.sector_num + 1]

        u = self._normalize_val(u, self.min_u, self.max_u)
        v = self._normalize_val(v, self.min_v, self.max_v)
        r = self._normalize_val(r, self.min_r, self.max_r)
        ye = self._normalize_val(ye, self.min_ye, self.max_ye)
        ye_dot = self._normalize_val(ye_dot, self.min_ye_dot, self.max_ye_dot)
        chi_ak = self._normalize_val(chi_ak, self.min_chi_ak, self.max_chi_ak)
        u_ref = self._normalize_val(u_ref, self.min_u, self.max_u)

        # sectors are ignored
        action0_last = self._normalize_val(action0_last, self.min_action0, self.max_action0)
        action1_last = self._normalize_val(action1_last, self.min_action1, self.max_action1)

        state = np.hstack(
            (u, v, r, ye, ye_dot, chi_ak, u_ref, sectors, action0_last, action1_last, state[7 + self.sector_num + 2:]))

        return state

    def step(self, action):
        '''
        @name: step
        @brief: ASMC and USV step, add obstacles and sensors.
        @param: action: vector of actions
        @return: state: state vector
                 reward: reward from the current action
                 done: if finished
        '''
        # Read overall vector variables
        state = self._denormalize_state(self.state)
        self.state = state
        position = self.position

        # Change from vectors to scalars
        u, v, r, ye, ye_dot, chi_ak, u_ref, sectors, action0_last, action1_last = state[0], state[1], state[2], state[
            3], state[4], state[5], state[6], state[7:7+self.sector_num - 1], state[7 + self.sector_num], state[7 + self.sector_num + 1]
        x, y, psi = position

        action[0] = self._denormalize_val(action[0], self.min_action0, self.max_action0)
        action[1] = self._denormalize_val(action[1], self.min_action1, self.max_action1)

        eta, upsilon, psi, tport, tstbd = self._compute_asmc(action)
        u, v, r = upsilon
        self.position = np.array([eta[0], eta[1], psi])

        # Calculate action derivative for reward
        action_dot0 = (action[0] - action0_last) / self.integral_step
        action_dot1 = (action[1] - action1_last) / self.integral_step
        action0_last = action[0]
        action1_last = action[1]

        x_0, y_0, u_ref, ak, x_d, y_d = self.target
        ak = np.math.atan2(y_d - y_0, x_d - x_0)
        ak = np.float32(ak)

        beta = np.math.asin(upsilon[1] / (0.001 + np.sqrt(upsilon[0] * upsilon[0] + upsilon[1] * upsilon[1])))
        chi = psi + beta
        chi = self._wrap_angle(chi)
        # Compute angle between USV and path
        chi_ak = self._wrap_angle(chi - ak)
        psi_ak = self._wrap_angle(psi - ak)

        # Compute cross-track error
        ye = -(eta[0] - x_0) * np.math.sin(ak) + (eta[1] - y_0) * np.math.cos(ak)
        ye_abs = np.abs(ye)

        # Compute collision
        done = False

        distance = np.hypot(self.posx - eta[0], self.posy - eta[1]) - self.radius - self.boat_radius - self.safety_radius
        distance = distance.reshape(-1)
        if distance.size == 0:
            collision = False
        else:
            collision = np.min(distance) < 0
            done = collision

        # Compute sensor readings
        self._compute_sensor_measurments(distance)

        # Feasability pooling: compute sectors
        sectors = self._compute_feasability_pooling(self.sensors)
        self.sectors = sectors
        sectors = np.clip((1 - sectors / self.sensor_max_range), -1, 1)

        # Compute reward
        reward, info = self.compute_reward(ye_abs, chi_ak, action_dot0, action_dot1, collision, u_ref, u, v)
        self.total_reward += reward

        if self.total_reward < -5000:
            done = True

        # Compute velocities relative to path (for ye derivative as ye_dot = v_ak)
        xe_dot, ye_dot = self.body_to_path(upsilon[0], upsilon[1], psi_ak)

        # If USV collides, abort
        # if collision == True:
        #
        # else:
        #     done = False


        #Clamp ye and finish ep
        if abs(ye) > self.max_ye:
            ye = np.copysign(self.max_ye, ye)
            reward = (1 - self.lambda_reward) * -2000
            done = True

        if x > self.max_x:
            done = True

        # Fill overall vector variables
        self.state = np.hstack(
            (upsilon[0], upsilon[1], upsilon[2], ye, ye_dot, chi_ak, u_ref, sectors, action0_last, action1_last, np.log10(self.lambda_reward)))
        self.state = self._normalize_state(self.state)

        # Reshape state
        state = self.state.reshape(self.observation_space.shape[0]).astype(np.float32)

        info.update({"position": position, "sensors": self.sensors, "sectors": sectors, "thrusters": (tport, tstbd)})
        return state, reward, done, info

    def reset(self):
        x = np.random.uniform(low=-2.5, high=2.5)
        y = np.random.uniform(low=-5.0, high=5.0)
        psi = np.random.uniform(low=-np.pi, high=np.pi)
        eta = np.array([x, y])
        upsilon = np.array([0., 0., 0.])
        eta_dot_last = np.array([0., 0., 0.])
        upsilon_dot_last = np.array([0., 0., 0.])
        action0_last = 0.0
        action1_last = 0.0
        e_u_int = 0.
        Ka_u = 0.
        Ka_psi = 0.
        e_u_last = 0.
        Ka_dot_u_last = 0.
        Ka_dot_psi_last = 0.
        psi_d_last = psi
        o_dot_dot_last = 0.
        o_dot_last = 0.
        o_last = 0.
        o_dot_dot = 0.
        o_dot = 0.
        o = 0.
        # Start and Final position
        x_0 = np.random.uniform(low=-2.5, high=2.5)
        y_0 = np.random.uniform(low=-5.0, high=5.0)
        x_d = np.random.uniform(low=15, high=30)
        y_d = y_0
        # Desired speed
        u_ref = np.random.uniform(low=self.min_u_ref, high=self.max_u_ref)
        # number of obstacles 
        self.num_obs = np.random.random_integers(low=15, high=25)
        # array of positions in x and y and radius
        self.posx = np.random.normal(15, 10, size=(self.num_obs, 1))
        self.posy = np.random.uniform(-10, 10, size=(self.num_obs, 1))
        self.radius = np.random.normal(1.1, 0.65, size=(self.num_obs, 1))

        distance = np.hypot(self.posx - eta[0],
                            self.posy - eta[1]) - self.radius - self.boat_radius - (self.safety_radius + 0.35)
        distance = distance.reshape(-1)

        # Delete all obstacles within boat radius
        elems_to_delete = np.flatnonzero(distance < 0)
        self.posx = np.delete(self.posx, elems_to_delete).reshape(-1,1)
        self.posy = np.delete(self.posy, elems_to_delete).reshape(-1,1)
        self.radius = np.delete(self.radius, elems_to_delete).reshape(-1,1)
        self.num_obs -= elems_to_delete.size

        ak = np.math.atan2(y_d - y_0, x_d - x_0)
        ak = np.float32(ak)

        psi_ak = psi - ak
        psi_ak = np.where(np.greater(np.abs(psi_ak), np.pi), np.sign(psi_ak) * (np.abs(psi_ak) - 2 * np.pi), psi_ak)
        psi_ak = np.float32(psi_ak)

        ye = -(x - x_0) * np.math.sin(ak) + (y - y_0) * np.math.cos(ak)
        xe_dot, ye_dot = self.body_to_path(upsilon[0], upsilon[1], psi_ak)

        self.state = np.hstack(
            (upsilon[0], upsilon[1], upsilon[2], ye, ye_dot, psi_ak, u_ref, self.sectors, action0_last, action1_last))
        self.state = self._normalize_state(self.state)
        self.aux_vars = np.array([e_u_int, Ka_u, Ka_psi])
        self.last = np.array(
            [eta_dot_last[0], eta_dot_last[1], eta_dot_last[2], upsilon_dot_last[0], upsilon_dot_last[1],
             upsilon_dot_last[2], e_u_last, Ka_dot_u_last, Ka_dot_psi_last])
        self.target = np.array([x_0, y_0, u_ref, ak, x_d, y_d])
        self.so_filter = np.array([psi_d_last, o_dot_dot_last, o_dot_last, o_last, o, o_dot, o_dot_dot])

        self.position = np.array([eta[0], eta[1], psi])

        self.lambda_reward = np.random.beta(5, 1.65)

        state, _, _, _ = self.step([0,0])
        return state

    def _compute_asmc(self, action):
        x_dot_last, y_dot_last, psi_dot_last, u_dot_last, v_dot_last, r_dot_last, e_u_last, Ka_dot_u_last, Ka_dot_psi_last = self.last
        psi_d_last, o_dot_dot_last, o_dot_last, o_last, o, o_dot, o_dot_dot = self.so_filter

        state = self.state
        u, v, r, ye, ye_dot, chi_ak, u_ref, sectors, action0_last, action1_last = state[0], state[1], state[2], state[
            3], state[4], state[5], state[6], state[7:7+self.sector_num - 1], state[7 + self.sector_num], state[7 + self.sector_num + 1]
        x, y, psi = self.position
        e_u_int, Ka_u, Ka_psi = self.aux_vars

        # Create model related vectors
        eta = np.array([x, y, psi])
        upsilon = np.array([u, v, r])
        eta_dot_last = np.array([x_dot_last, y_dot_last, psi_dot_last])
        upsilon_dot_last = np.array([u_dot_last, v_dot_last, r_dot_last])

        for i in range(10):
            beta = np.math.asin(upsilon[1] / (0.001 + np.hypot(upsilon[0], upsilon[1])))
            chi = psi + beta
            #chi = np.where(np.greater(np.abs(chi), np.pi), (np.sign(chi)) * (np.abs(chi) - 2 * np.pi), chi)

            # Compute the desired heading
            psi_d = chi + action[1]
            # psi_d = ak + action[1]
            #psi_d = np.where(np.greater(np.abs(psi_d), np.pi), (np.sign(psi_d)) * (np.abs(psi_d) - 2 * np.pi), psi_d)

            # Second order filter to compute desired yaw rate
            r_d = (psi_d - psi_d_last) / self.integral_step
            psi_d_last = psi_d
            o_dot_dot = (((r_d - o_last) * self.f1) - (self.f3 * o_dot_last)) * self.f2
            o_dot = (self.integral_step) * (o_dot_dot + o_dot_dot_last) / 2 + o_dot
            o = (self.integral_step) * (o_dot + o_dot_last) / 2 + o
            r_d = o
            o_last = o
            o_dot_last = o_dot
            o_dot_dot_last = o_dot_dot

            # Compute variable hydrodynamic coefficients
            Xu = -25
            Xuu = 0
            if (abs(upsilon[0]) > 1.2):
                Xu = 64.55
                Xuu = -70.92

            Yv = 0.5 * (-40 * 1000 * abs(upsilon[1])) * \
                 (1.1 + 0.0045 * (1.01 / 0.09) - 0.1 * (0.27 / 0.09) + 0.016 * (np.power((0.27 / 0.09), 2)))
            Yr = 6 * (-3.141592 * 1000) * \
                 np.sqrt(np.power(upsilon[0], 2) + np.power(upsilon[1], 2)) * 0.09 * 0.09 * 1.01
            Nv = 0.06 * (-3.141592 * 1000) * \
                 np.sqrt(np.power(upsilon[0], 2) + np.power(upsilon[1], 2)) * 0.09 * 0.09 * 1.01
            Nr = 0.02 * (-3.141592 * 1000) * \
                 np.sqrt(np.power(upsilon[0], 2) + np.power(upsilon[1], 2)) * 0.09 * 0.09 * 1.01 * 1.01

            # Rewrite USV model in simplified components f and g
            g_u = 1 / (self.m - self.X_u_dot)
            g_psi = 1 / (self.Iz - self.N_r_dot)
            f_u = (((self.m - self.Y_v_dot) * upsilon[1] * upsilon[2] + (
                        Xuu * np.abs(upsilon[0]) + Xu * upsilon[0])) / (self.m - self.X_u_dot))
            f_psi = (((-self.X_u_dot + self.Y_v_dot) * upsilon[0] * upsilon[1] + (Nr * upsilon[2])) / (
                        self.Iz - self.N_r_dot))

            # Compute heading error
            e_psi = psi_d - eta[2]
            e_psi = np.where(np.greater(np.abs(e_psi), np.pi), (np.sign(e_psi)) * (np.abs(e_psi) - 2 * np.pi), e_psi)
            e_psi_dot = r_d - upsilon[2]

            # Compute desired speed (unnecessary if DNN gives it)
            u_d = action[0]

            # Compute speed error
            e_u = u_d - upsilon[0]
            e_u_int = self.integral_step * (e_u + e_u_last) / 2 + e_u_int

            # Create sliding surfaces for speed and heading
            sigma_u = e_u + self.lambda_u * e_u_int
            sigma_psi = e_psi_dot + self.lambda_psi * e_psi

            # Compute ASMC gain derivatives
            Ka_dot_u = np.where(np.greater(Ka_u, self.kmin_u), self.k_u * np.sign(np.abs(sigma_u) - self.mu_u),
                                self.kmin_u)
            Ka_dot_psi = np.where(np.greater(Ka_psi, self.kmin_psi),
                                  self.k_psi * np.sign(np.abs(sigma_psi) - self.mu_psi), self.kmin_psi)

            # Compute gains
            Ka_u = self.integral_step * (Ka_dot_u + Ka_dot_u_last) / 2 + Ka_u
            Ka_dot_u_last = Ka_dot_u

            Ka_psi = self.integral_step * (Ka_dot_psi + Ka_dot_psi_last) / 2 + Ka_psi
            Ka_dot_psi_last = Ka_dot_psi

            # Compute ASMC for speed and heading
            ua_u = (-Ka_u * np.power(np.abs(sigma_u), 0.5) * np.sign(sigma_u)) - (self.k2_u * sigma_u)
            ua_psi = (-Ka_psi * np.power(np.abs(sigma_psi), 0.5) * np.sign(sigma_psi)) - (self.k2_psi * sigma_psi)

            # Compute control inputs for speed and heading
            Tx = ((self.lambda_u * e_u) - f_u - ua_u) / g_u
            Tz = ((self.lambda_psi * e_psi) - f_psi - ua_psi) / g_psi

            # Compute both thrusters and saturate their values
            Tport = (Tx / 2) + (Tz / self.B)
            Tstbd = (Tx / (2 * self.c)) - (Tz / (self.B * self.c))

            Tport = np.where(np.greater(Tport, 36.5), 36.5, Tport)
            Tport = np.where(np.less(Tport, -30), -30, Tport)
            Tstbd = np.where(np.greater(Tstbd, 36.5), 36.5, Tstbd)
            Tstbd = np.where(np.less(Tstbd, -30), -30, Tstbd)

            # Compute USV model matrices
            M = np.array([[self.m - self.X_u_dot, 0, 0],
                          [0, self.m - self.Y_v_dot, 0 - self.Y_r_dot],
                          [0, 0 - self.N_v_dot, self.Iz - self.N_r_dot]])

            T = np.array([Tport + self.c * Tstbd, 0, 0.5 * self.B * (Tport - self.c * Tstbd)])

            CRB = np.array([[0, 0, 0 - self.m * upsilon[1]],
                            [0, 0, self.m * upsilon[0]],
                            [self.m * upsilon[1], 0 - self.m * upsilon[0], 0]])

            CA = np.array([[0, 0, 2 * ((self.Y_v_dot * upsilon[1]) + ((self.Y_r_dot + self.N_v_dot) / 2) * upsilon[2])],
                           [0, 0, 0 - self.X_u_dot * self.m * upsilon[0]],
                           [2 * (((0 - self.Y_v_dot) * upsilon[1]) - ((self.Y_r_dot + self.N_v_dot) / 2) * upsilon[2]),
                            self.X_u_dot * self.m * upsilon[0], 0]])

            C = CRB + CA

            Dl = np.array([[0 - Xu, 0, 0],
                           [0, 0 - Yv, 0 - Yr],
                           [0, 0 - Nv, 0 - Nr]])

            Dn = np.array([[Xuu * abs(upsilon[0]), 0, 0],
                           [0, self.Yvv * abs(upsilon[1]) + self.Yvr * abs(upsilon[2]), self.Yrv *
                            abs(upsilon[1]) + self.Yrr * abs(upsilon[2])],
                           [0, self.Nvv * abs(upsilon[1]) + self.Nvr * abs(upsilon[2]),
                            self.Nrv * abs(upsilon[1]) + self.Nrr * abs(upsilon[2])]])

            D = Dl - Dn

            # Compute acceleration and velocity in body
            upsilon_dot = np.matmul(np.linalg.inv(
                M), (T - np.matmul(C, upsilon) - np.matmul(D, upsilon)))
            upsilon = (self.integral_step) * (upsilon_dot +
                                              upsilon_dot_last) / 2 + upsilon  # integral
            upsilon_dot_last = upsilon_dot

            # Rotation matrix
            J = np.array([[np.cos(eta[2]), -np.sin(eta[2]), 0],
                          [np.sin(eta[2]), np.cos(eta[2]), 0],
                          [0, 0, 1]])

            # Compute NED position
            eta_dot = np.matmul(J, upsilon)  # transformation into local reference frame
            eta = (self.integral_step) * (eta_dot + eta_dot_last) / 2 + eta  # integral
            eta_dot_last = eta_dot

            psi = eta[2]

        self.last = np.array(
            [eta_dot_last[0], eta_dot_last[1], eta_dot_last[2], upsilon_dot_last[0], upsilon_dot_last[1],
             upsilon_dot_last[2], e_u_last, Ka_dot_u_last, Ka_dot_psi_last])

        self.so_filter = np.array([psi_d_last, o_dot_dot_last, o_dot_last, o_last, o, o_dot, o_dot_dot])
        self.aux_vars = np.array([e_u_int, Ka_u, Ka_psi])

        return eta, upsilon, psi, Tport, Tstbd


    def _compute_sensor_measurments(self, distance):
        x = self.position[0]
        y = self.position[1]
        psi = self.position[2]

        obs_order = np.argsort(distance)  # order obstacles in closest to furthest

        sensor_len = len(self.sensors)
        self.sensors = np.vstack((-np.pi * 2 / 3 + np.arange(sensor_len) * self.lidar_resolution,
                                  np.ones(sensor_len) * self.sensor_max_range)).T

        sensor_angles = self.sensors[:, 0] + psi
        #sensor_angles = np.where(np.greater(np.abs(sensor_angles), np.pi),
        #                         np.sign(sensor_angles) * (np.abs(sensor_angles) - 2 * np.pi), sensor_angles)

        obstacle_positions = np.hstack((self.posx[:], self.posy[:]))[obs_order]

        boat_position = np.array([x,y])
        ned_obstacle_positions = self.compute_obstacle_positions(sensor_angles, obstacle_positions,
                                                                 boat_position)

        for i in range(sensor_len):
            if self.sensors[i][1] != self.sensor_max_range:
                continue
            for j in range(self.num_obs):
                (obs_x, obs_y) = ned_obstacle_positions[i][j]
                obs_idx = obs_order[j]

                if obs_x < 0:
                    # Obstacle is behind sensor
                    # self.sensors[i][1] = self.sensor_max_range
                    continue

                delta = (self.radius[obs_idx] * self.radius[obs_idx]) - (obs_y * obs_y)
                if delta < 0:
                    continue

                new_distance = obs_x - np.sqrt(delta)
                if new_distance < self.sensor_max_range:
                    self.sensors[i][1] = min(self.sensors[i][1], new_distance)

    def _compute_feasability_pooling(self, sensors):
        sectors = np.full((self.sector_num), self.sensor_max_range)
        for i in range(self.sector_num):  # loop through sectors
            x = sensors[i * self.sector_size:(i + 1) * self.sector_size, 1]
            x_ordered = np.argsort(x)
            for j in range(self.sector_size):  # loop through
                x_index = x_ordered[j]
                arc_length = self.lidar_resolution * x[x_index]
                opening_width = arc_length / 2
                opening_found = False
                for k in range(self.sector_size):
                    if x[k] > x[x_index]:
                        opening_width = opening_width + arc_length
                        if opening_width > (2 * (self.boat_radius + self.safety_radius)):
                            opening_found = True
                            break
                    else:
                        opening_width = opening_width + arc_length / 2
                        if opening_width > (self.boat_radius + self.safety_radius):
                            opening_found = True
                            break
                        opening_width = 0
                if not opening_found:
                    sectors[i] = x[x_index]
        return sectors

    def _transform_points(self, points, x, y, angle):
        if angle is not None:
            s,c = (np.sin(angle), np.cos(angle))
            points = [(px * c - py * s, px * s + py * c) for (px, py) in points]
        points = [(px + x, py + y) for (px, py) in points]
        return points

    def _draw_sectors(self, scale, position, sensors, sectors):
        import pygame
        x = position[0]
        y = position[1]
        psi = position[2]

        for i in range(len(self.sensors)):
            angle = sensors[i][0] + psi
            #angle = np.where(np.greater(np.abs(angle), np.pi), np.sign(angle) * (np.abs(angle) - 2 * np.pi), angle)
            initial = ((y - self.min_y) * scale, (x - self.min_x) * scale)
            m = np.math.tan(angle)
            x_f = sensors[i][1] * np.math.cos(angle) + x - self.min_x
            y_f = sensors[i][1] * np.math.sin(angle) + y - self.min_y
            final = (y_f * scale, x_f * scale)
            section = np.int(np.floor(i / self.sector_size))

            color = (0, 255, 0)

            if sectors[section] < self.sensor_max_range:
                color = (255, 0, 0)
                if(i % 10 == 0):
                    color = (255,0,255)
            elif(i % 10 == 0):
                color = (0,0,255)

            pygame.draw.line(self.surf, color, initial, final)

    def _draw_boat(self, scale, position):
        import pygame
        boat_width = 15
        boat_height = 20

        l, r, t, b, c, m = -boat_width / 2, boat_width / 2, boat_height, 0, 0, boat_height / 2
        boat_points = [(l, b), (l, m), (c, t), (r, m), (r, b)]
        boat_points = [(x,y - boat_height / 2) for (x,y) in boat_points]
        boat_points = self._transform_points(boat_points, (position[1] - self.min_y) * scale, (position[0] - self.min_x) * scale, -position[2])

        pygame.draw.polygon(self.surf, (3,94,252), boat_points)

    def _draw_obstacles(self, scale):
        import pygame
        for i in range(self.num_obs):
            obs_points = [((self.posy[i][0] - self.min_y) * scale, (self.posx[i][0] - self.min_x) * scale)]
            pygame.draw.circle(self.surf, (0,0,255), obs_points[0], self.radius[i][0] * scale)

    def _draw_highlighted_sectors(self, scale):
        import pygame
        x = self.position[0]
        y = self.position[1]
        psi = self.position[2]

        angle = -(2 / 3) * np.pi + psi
        angle = np.where(np.greater(np.abs(angle), np.pi), np.sign(angle) * (np.abs(angle) - 2 * np.pi), angle)
        for i in range(self.sector_num + 1):
            initial = ((y - self.min_y) * scale, (x - self.min_x) * scale)
            m = np.math.tan(angle)
            x_f = self.sensor_max_range * np.math.cos(angle) + x - self.min_x
            y_f = self.sensor_max_range * np.math.sin(angle) + y - self.min_y
            final = (y_f * scale, x_f * scale)
            pygame.draw.line(self.surf, (0,0,0), initial, final, width=2)
            angle = angle + self.sensor_span / self.sector_num
            angle = np.where(np.greater(np.abs(angle), np.pi), np.sign(angle) * (np.abs(angle) - 2 * np.pi), angle)

    def render(self, mode='human', info=None, draw_obstacles=True):
        screen_width = 400
        screen_height = 800

        world_width = self.max_y - self.min_y
        world_width = 20
        scale = screen_width / world_width

        if info is not None:
            position = info['position']
            sensors = info['sensors']
            sectors = info['sectors']
        else:
            position = self.position
            sensors = self.sensors
            sectors = self.sectors

        x = position[0]
        y = position[1]
        psi = position[2]

        import pygame
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        self._draw_sectors(scale, position, sensors, sectors)

        if(draw_obstacles):
            self._draw_obstacles(scale)

        self._draw_boat(scale, position)

        clearance = -10

        x_0 = (self.min_x - self.min_x) * scale
        y_0 = (self.target[1] - self.min_y) * scale
        x_d = (self.max_x - self.min_x) * scale
        y_d = (self.target[5] - self.min_y) * scale
        start = (y_0, x_0)
        end = (y_d, x_d)

        #Draw path
        pygame.draw.line(self.surf, (0, 255, 0), start, end, width=2)

        #Draw safety radius
        safety_radius = (self.boat_radius + self.safety_radius) * scale
        safety = ((y - self.min_y) * scale, (x - self.min_x) * scale)
        pygame.draw.circle(self.surf, (255,0,0), safety, safety_radius, width=3)

        self.surf = pygame.transform.flip(self.surf, False, True)

        if mode == "human":
            self.screen.blit(self.surf, (0,0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        if mode == "rgb_array":
            return self._create_image_array(self.surf, (screen_width, screen_height))
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def _create_image_array(self, screen, size):
        import pygame

        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            self.isopen = False
            pygame.quit()


        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _crosstrack_reward(self, ye):
        return np.maximum(np.exp(-k_ye * np.power(ye,2)),  np.exp(-k_ye * np.abs(ye))) + 1

    def _coursedirection_reward(self, chi_ak, u, v):
        reward = -np.exp(1 * np.abs(chi_ak) - np.pi) + 1
        return reward


    def _oa_reward(self, sensor):
        gammainv = (1 + np.abs(sensor[0] * self.gamma_theta))
        denominator = gammainv

        distelem = (self.gamma_x * np.power(np.maximum(sensor[1], self.epsilon), 2))
        numerator = 1 / distelem
        return -numerator / denominator

    def compute_reward(self, ye, chi_ak, action_dot0, action_dot1, collision, u_ref, u, v):
        info = {}
        if (collision == False):
            # Velocity reward
            reward_u = np.clip(np.exp(-self.k_uu * np.abs(u_ref - np.hypot(u, v))), -10, 10) + 1
            # Action velocity gradual change reward
            reward_a0 = np.math.tanh(-self.c_action0 * np.power(action_dot0, 2)) * self.k_action0
            # Action angle gradual change reward
            reward_a1 = np.math.tanh(-self.c_action1 * np.power(action_dot1, 2)) * self.k_action1

            # Path following reward
            reward_coursedirection = self._coursedirection_reward(chi_ak, u, v)
            reward_crosstrack = self._crosstrack_reward(ye)
            #reward_pf = -1 + reward_coursedirection * reward_crosstrack
            reward_pf = -1 + reward_coursedirection * reward_crosstrack * reward_u
            # Obstacle avoidance reward
            numerator = np.sum(np.power(self.gamma_x * np.power(np.maximum(self.sensors[:,1], self.epsilon), 2), -1))
            denominator = np.sum(1 + np.abs(self.sensors[:, 0] * self.gamma_theta))
            reward_oa = -(np.log(numerator / denominator))

            #Exists reward
            reward_exists = -self.lambda_reward * 0.015

            # Total non-collision reward
            reward = self.lambda_reward * reward_pf + (1 - self.lambda_reward) * reward_oa + reward_exists + reward_a0 + reward_a1

            info['reward_velocity'] = np.hypot(u, v)
            info['reward_u'] = reward_u
            info['reward_a0'] = reward_a0
            info['reward_a1'] = reward_a1
            info['reward_coursedirection'] = reward_coursedirection
            info['reward_crosstrack'] = reward_crosstrack
            info['reward_oa'] = reward_oa
            info['reward_pf'] = reward_pf
            info['reward_ye'] = ye
            info['reward_exists'] = reward_exists
            info['reward_chi_ak'] = chi_ak
            info['reward_u_ref'] = u_ref
            #print(info)

            if (np.abs(reward) > 100000 and not collision):
                print("PANIK")

        else:
            # Collision Reward
            reward = (1 - self.lambda_reward) * -2000

        #print(reward)
        info['reward'] = reward
        return reward, info

    def body_to_path(self, x2, y2, alpha):
        '''
        @name: body_to_path
        @brief: Coordinate transformation between body and path reference frames.
        @param: x2: target x coordinate in body reference frame
                y2: target y coordinate in body reference frame
        @return: path_x2: target x coordinate in path reference frame
                 path_y2: target y coordinate in path reference frame
        '''
        p = np.array([x2, y2])
        J = self.rotation_matrix(alpha)
        n = J.dot(p)
        path_x2 = n[0]
        path_y2 = n[1]
        return (path_x2, path_y2)

    def rotation_matrix(self, angle):
        '''
        @name: rotation_matrix
        @brief: Transformation matrix template.
        @param: angle: angle of rotation
        @return: J: transformation matrix
        '''
        J = np.array([[np.math.cos(angle), -np.math.sin(angle)],
                      [np.math.sin(angle), np.math.cos(angle)]])
        return (J)

    def compute_obstacle_positions(self, sensor_angles, obstacle_pos, boat_pos):
        sensor_angle_len = len(sensor_angles)

        # Generate rotation matrix
        c, s = np.cos(sensor_angles), np.sin(sensor_angles)
        rm = np.linalg.inv(np.array([c, -s, s, c]).T.reshape(len(sensor_angles), 2, 2))

        n = obstacle_pos - boat_pos

        # obs_pos = np.zeros((sensor_angle_len, obstacle_len, 2))

        # for i in range(sensor_angle_len):
        #     for j in range(obstacle_len):
        #         obs_pos[i][j] = (rm[i].dot(n[j]))

        obs_pos = np.inner(rm,n).transpose(0,2,1)

        obs_pos[:, :, 1] *= -1  # y *= -1
        return obs_pos

    def ned_to_body(self, ned_x2, ned_y2, ned_xboat, ned_yboat, psi):
        '''
        @name: ned_to_ned
        @brief: Coordinate transformation between NED and body reference frames.
        @param: ned_x2: target x coordinate in ned reference frame
                ned_y2: target y coordinate in ned reference frame
                ned_xboat: robot x regarding NED
                ned_yboat: robot y regarding NED
                psi: robot angle regarding NED
        @return: body_x2: target x coordinate in body reference frame
                body_y2: target y coordinate in body reference frame
        '''
        n = np.array([ned_x2 - ned_xboat, ned_y2 - ned_yboat], dtype=np.float32)
        J = self.compute_rot_matrix(psi.item())
        b = J.dot(n)
        return b
