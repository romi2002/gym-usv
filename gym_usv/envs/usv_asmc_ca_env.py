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
from numba import njit
from gym_usv.control import UsvAsmc
from gym_usv.utils import generate_path, place_obstacles, simplified_lookahead
from .usv_ca_renderer import UsvCaRenderer

class UsvAsmcCaEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, config=None):
        # Integral step (or derivative) for 100 Hz
        self.integral_step = 0.01

        self.place_obstacles = True
        self.use_kinematic_model = True

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
        self.sensor_num = 225
        self.sensors = np.zeros((self.sensor_num, 2))
        self.sensor_span = (2 / 3) * (2 * np.pi)
        self.lidar_resolution = self.sensor_span / self.sensor_num  # angle resolution in radians
        self.sector_num = 25  # number of sectors
        self.sector_size = np.floor(self.sensor_num / self.sector_num).astype(int)  # number of points per sector
        self.sensor_max_range = 40.0  # m
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
        self.min_action0 = 0.0
        self.max_action0 = 1.4
        # angle (change to -pi and pi if necessary)
        self.min_action1 = -np.pi
        self.max_action1 = np.pi

        # Reward associated functions anf gains
        self.k_ye = 0.25  # Crosstracking reward

        self.k_uu = 3.0  # Velocity Reward

        self.gamma_theta = 4.0  # 4.0
        self.gamma_x = 0.0005  # 0.005
        self.epsilon = 1
        self.lambda_reward = 0.85

        self.w_action0 = 0.2
        self.w_action1 = 0.2
        # Action gradual change reward
        self.c_action0 = 1. / np.power((self.max_action0 / 2 - self.min_action0 / 2) / self.integral_step, 2)
        self.c_action1 = 1. / np.power((self.max_action1 / 2 - self.min_action1 / 2) / self.integral_step, 2)
        self.k_action0 = 0.0
        self.k_action1 = 0.0

        # Min and max values of the state
        self.min_u = -2.5/2
        self.max_u = 2.5/2
        self.min_v = -1.75/2
        self.max_v = 1.75/2
        self.min_r = -2.
        self.max_r = 2.

        self.min_lookahead_error = -np.pi
        self.max_lookahead_error = np.pi
        self.min_courseerror = -10
        self.max_courseerror = 10

        self.min_u_ref = 0.7
        self.max_u_ref = 1.4
        self.u_ref = 0

        self.action0_last = 0
        self.action1_last = 0

        self.min_sectors = np.zeros((self.sector_num))
        self.max_sectors = np.full((self.sector_num), 1)
        self.sectors = np.zeros((self.sector_num))
        self.debug_vars = {}

        self.state_length = 9 + self.sector_num

        # Min and max state vectors
        self.low_state = np.full(self.state_length, -1.0)
        self.high_state = np.full(self.state_length, 1.0)

        self.min_action = np.array([-1.0, -1.0])
        self.max_action = np.array([1.0, 1.0])

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

        self.lookahead_distance = 1.0
        self.courseangle_error = 0

        self.renderer = None
        self.isopen = True
        self.total_reward = 0

        self.path = None
        self.path_deriv = None
        self.waypoints = None

        self.asmc = UsvAsmc()
        self.reset()

    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _normalize_val(self, x, in_min, in_max):
        return self._map(x, in_min, in_max, -1, 1)

    def _denormalize_val(self, x, out_min, out_max):
        return self._map(x, -1, 1, out_min, out_max)

    def _map(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def _denormalize_state(self, state):
        u, v, r, ylp, lookahead_error, courseerror = state[0], state[1], state[2], state[3], state[4], state[5]

        u = self._denormalize_val(u, self.min_u, self.max_u)
        v = self._denormalize_val(v, self.min_v, self.max_v)
        r = self._denormalize_val(r, self.min_r, self.max_r)
        ylp = self._denormalize_val(ylp, -np.pi, np.pi)
        lookahead_error = self._denormalize_val(lookahead_error, self.min_lookahead_error, self.max_lookahead_error)
        courseerror = self._denormalize_val(lookahead_error, self.min_courseerror, self.max_courseerror)

        state = np.hstack(
            (u, v, r, ylp, lookahead_error, courseerror))

        return state

    def _normalize_state(self, state):
        u, v, r, ylp, lookahead_error, courseerror = state[0], state[1], state[2], state[3], state[4], state[5]

        self.debug_vars['vel'] = np.hypot(u, v)
        #self.debug_vars['v'] = v
        #self.debug_vars['r'] = r

        u = self._normalize_val(u, self.min_u, self.max_u)
        v = self._normalize_val(v, self.min_v, self.max_v)
        r = self._normalize_val(r, self.min_r, self.max_r)
        ylp = self._normalize_val(ylp, -np.pi, np.pi)
        lookahead_error = self._normalize_val(lookahead_error, self.min_lookahead_error, self.max_lookahead_error)
        courseerror = self._normalize_val(lookahead_error, self.min_courseerror, self.max_courseerror)

        state = np.hstack(
            (u, v, r, ylp, lookahead_error, courseerror, state[6:]))

        return state

    def step(self, action_in):
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
        u, v, r, courseangle_error, crosstrack_error, sectors = state[0], state[1], state[2], state[3], state[4], state[5:]
        x, y, psi = position

        #self.debug_vars['raw_action_0'] = action_in[0]
        #self.debug_vars['raw_action_1'] = action_in[1]

        action = [
            self._denormalize_val(action_in[0], self.min_action0, self.max_action0),
            self._denormalize_val(action_in[1], self.min_action1, self.max_action1)
        ]

        if self.use_kinematic_model:
            # Update rotational vel
            T = 1/10
            dvr = T * (psi - action[1])
            r += dvr
            r = np.clip(r, self.min_r, self.max_r)
            psi += r * self.integral_step

            du, dv = (T * (u - action[0])), 0
            u = np.clip(u + du, self.min_u, self.max_u)
            v = np.clip(v + dv, self.min_v, self.max_v)

            x += u * self.integral_step * -np.cos(psi)
            y += u * self.integral_step * -np.sin(psi)

            upsilon = u,v,r
            self.debug_vars['u'] = u
            self.debug_vars['r'] = r
            eta = x,y,psi
        else:
            eta, upsilon = self.asmc.compute(action, np.array([x, y, psi]), np.array([u, v, r]))

        psi = eta[2]
        u, v, r = upsilon
        self.position = eta

        # Calculate action derivative for reward
        self.debug_vars['action_0'] = action[0]
        self.debug_vars['action_1'] = action[1]
        action_dot0 = (action[0] - self.action0_last) / self.integral_step
        action_dot1 = (action[1] - self.action1_last) / self.integral_step
        self.action0_last = action[0]
        self.action1_last = action[1]

        # Update path target with lookahead
        x_0, y_0 = position[0], position[1]
        x_d, y_d = simplified_lookahead(self.path, self.waypoints, x_0, self.lookahead_distance)
        self.target = [x_0, y_0, x_d, y_d, self.u_ref]

        # Compute collision
        done = False

        distance = np.hypot(self.posx - eta[0],
                            self.posy - eta[1]) - self.radius - self.boat_radius - self.safety_radius
        distance = distance.reshape(-1)
        if distance.size == 0:
            collision = False
        else:
            collision = (np.min(distance) < 0)

        # Compute sensor readings
        self.sensors = self._compute_sensor_measurments(
            self.position,
            self.sensor_num,
            self.sensor_max_range,
            self.radius,
            self.lidar_resolution,
            self.posx,
            self.posy,
            self.num_obs,
            distance)

        # Feasability pooling: compute sectors
        sectors = np.full(self.sector_num, 1)

        if self.place_obstacles:
            sectors = self._compute_feasability_pooling(self.sector_num, self.sector_size, self.sensor_max_range,
                                                        self.lidar_resolution, self.boat_radius + self.safety_radius,
                                                        self.sensors)
            sectors = np.clip((sectors / self.sensor_max_range), 0, 1)

        self.sectors = sectors

        # Compute lookahead course error
        # compute angle from north to tangent line at lookahead point
        gamma_p = np.arctan2(self.path_deriv(x_d) - self.path_deriv(x_d + 0.1), 0.1).item()
        courseangle_error = self._wrap_angle(gamma_p - psi)
        self.courseangle_error = courseangle_error

        # Compute course error
        # the closest point on path to boat
        chi_error = self._wrap_angle(np.arctan2(y_d - y_0, x_d - x_0) - psi)
        self.debug_vars['chi_error'] = chi_error

        self.debug_vars['ca_error'] = courseangle_error

        # Compute reward
        reward, info = self.compute_reward(courseangle_error, chi_error, action_dot0, action_dot1, collision, self.u_ref, u, v)
        self.total_reward += reward
        self.debug_vars['reward'] = reward

        # If USV collides, abort
        # if collision:
        #     done = True

        if x > self.max_x:
            done = True

        distance_to_final = np.hypot(self.waypoints[-1][0] - position[0], self.waypoints[-1][1] - position[1])
        #self.debug_vars['dtf'] = distance_to_final
        #if distance_to_final < 1:
            #reward = (1 - self.lambda_reward) * 200
            #done = True

        # Fill overall vector variables
        #surge, sway, yaw velocity, LA course error, course error, cross-track, lambda, sectors...
        self.state = np.hstack((
            upsilon[0], upsilon[1], upsilon[2], chi_error, courseangle_error, crosstrack_error, action[0] / self.max_action0, action[1] / self.max_action1, -np.log10(self.lambda_reward) / 30.0, sectors))
        self.state = self._normalize_state(self.state)

        # Reshape state
        self.state = self.state.reshape(self.observation_space.shape[0]).astype(np.float32)

        info.update({"position": position, "sensors": self.sensors, "sectors": sectors, "max_x":self.max_x, "collision": collision})
        return self.state, reward, done, info

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
        self.u_ref = np.random.uniform(low=self.min_u_ref, high=self.max_u_ref)
        self.debug_vars['u_ref'] = self.u_ref
        # number of obstacles 
        self.num_obs = np.random.randint(low=20, high=50)
        if not self.place_obstacles:
            self.num_obs = 0

        self.path, self.waypoints = generate_path(np.array([x_0, y_0]), np.random.randint(10, 20))
        self.path_deriv = self.path.derivative()
        obstacles = place_obstacles(self.path, self.waypoints, self.num_obs, obs_pos_std=15)
        self.num_obs = len(obstacles)  # Update after removing obstacles
        self.posx = obstacles[:, 0]
        self.posy = obstacles[:, 1]
        self.radius = obstacles[:, 2]

        self.total_reward = 0

        distance = np.hypot(self.posx - eta[0],
                            self.posy - eta[1]) - self.radius - self.boat_radius - (self.safety_radius + 0.35)
        distance = distance.reshape(-1)

        self.asmc = UsvAsmc()

        # Delete all obstacles within boat radius
        elems_to_delete = np.flatnonzero(distance < 0)
        self.posx = np.delete(self.posx, elems_to_delete).reshape(-1, 1)
        self.posy = np.delete(self.posy, elems_to_delete).reshape(-1, 1)
        self.radius = np.delete(self.radius, elems_to_delete).reshape(-1, 1)
        self.num_obs -= elems_to_delete.size

        ak = np.math.atan2(y_d - y_0, x_d - x_0)
        ak = np.float32(ak)

        psi_ak = psi - ak
        psi_ak = np.where(np.greater(np.abs(psi_ak), np.pi), np.sign(psi_ak) * (np.abs(psi_ak) - 2 * np.pi), psi_ak)
        psi_ak = np.float32(psi_ak)

        ye = -(x - x_0) * np.math.sin(ak) + (y - y_0) * np.math.cos(ak)
        xe_dot, ye_dot = self.body_to_path(upsilon[0], upsilon[1], psi_ak)

        self.state = np.hstack(
            (upsilon[0], upsilon[1], upsilon[2], 0, 0, 0, 0, self.sectors))
        self.state = self._normalize_state(self.state)
        self.aux_vars = np.array([e_u_int, Ka_u, Ka_psi])
        self.last = np.array(
            [eta_dot_last[0], eta_dot_last[1], eta_dot_last[2], upsilon_dot_last[0], upsilon_dot_last[1],
             upsilon_dot_last[2], e_u_last, Ka_dot_u_last, Ka_dot_psi_last])
        self.target = np.array([x_0, y_0, x_d, y_d, self.u_ref])
        self.so_filter = np.array([psi_d_last, o_dot_dot_last, o_dot_last, o_last, o, o_dot, o_dot_dot])

        self.position = np.array([eta[0], eta[1], psi])

        self.lambda_reward = 1 - np.power(10, -np.random.beta(8, 0.65)) + 0.1
        self.debug_vars['lambda'] = self.lambda_reward

        state, _, _, _ = self.step([0, 0])
        return state

    @staticmethod
    def _compute_sensor_measurments(position, sensor_count, sensor_max_range, radius, lidar_resolution, posx, posy,
                                    num_obs, distance):
        x = position[0]
        y = position[1]
        psi = position[2]

        obs_order = np.argsort(distance)  # order obstacles in closest to furthest

        sensors = np.vstack((-np.pi * 2 / 3 + np.arange(sensor_count) * lidar_resolution,
                             np.ones(sensor_count) * sensor_max_range)).T

        sensor_angles = sensors[:, 0] + psi
        # sensor_angles = np.where(np.greater(np.abs(sensor_angles), np.pi),
        #                         np.sign(sensor_angles) * (np.abs(sensor_angles) - 2 * np.pi), sensor_angles)

        obstacle_positions = np.hstack((posx[:], posy[:]))[obs_order]

        boat_position = np.array([x, y])
        ned_obstacle_positions = UsvAsmcCaEnv.compute_obstacle_positions(sensor_angles, obstacle_positions,
                                                                         boat_position)

        new_dist = UsvAsmcCaEnv._compute_sensor_distances(sensor_max_range, num_obs, sensors, radius,
                                                          ned_obstacle_positions, obs_order)
        sensors[:, 1] = new_dist
        return sensors

    @staticmethod
    @njit
    def _compute_sensor_distances(sensor_max_range, num_obs, sensors, radius, ned_obstacle_positions, obs_order):
        new_distances = np.full(len(sensors), sensor_max_range)
        for i in range(len(sensors)):
            if sensors[i][1] != sensor_max_range:
                continue
            for j in range(num_obs):
                (obs_x, obs_y) = ned_obstacle_positions[i][j]
                obs_idx = obs_order[j]

                if obs_x < 0:
                    # Obstacle is behind sensor
                    # self.sensors[i][1] = self.sensor_max_range
                    continue

                delta = (radius[obs_idx] * radius[obs_idx]) - (obs_y * obs_y)
                if delta < 0:
                    continue

                new_distance = obs_x - np.sqrt(delta)
                if new_distance < sensor_max_range:
                    new_distances[i] = min(new_distance[0], sensors[i][1])
        return new_distances

    @staticmethod
    @njit
    def _compute_feasability_pooling(sector_num, sector_size, sensor_max_range, lidar_resolution, boat_radius, sensors):
        sectors = np.full(sector_num, sensor_max_range)
        for i in range(sector_num):  # loop through sectors
            x = sensors[i * sector_size:(i + 1) * sector_size, 1]
            x_ordered = np.argsort(x)
            for j in range(sector_size):  # loop through
                x_index = x_ordered[j]
                arc_length = lidar_resolution * x[x_index]
                opening_width = arc_length / 2
                opening_found = False
                for k in range(sector_size):
                    if x[k] > x[x_index]:
                        opening_width += arc_length
                        if opening_width > (2 * boat_radius):
                            opening_found = True
                            break
                    else:
                        opening_width += arc_length / 2
                        if opening_width > boat_radius:
                            opening_found = True
                            break
                        opening_width = 0
                if not opening_found:
                    sectors[i] = x[x_index]
        return sectors

    ## TODO Add screen space transformation functions
    def render(self, mode='human', info=None, draw_obstacles=True, show_debug_vars=True):
        if self.renderer is None:
            self.renderer = UsvCaRenderer()

        return self.renderer.render(
            self.position,
            self.sensors,
            self.sectors,
            self.sector_size,
            self.posx,
            self.posy,
            self.radius,
            self.waypoints,
            self.courseangle_error,
            self.path,
            self.target,
            self.debug_vars,
            show_debug_vars,
            mode
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def _crosstrack_reward(self, ye):
        return np.maximum(np.exp(-self.k_ye * np.power(ye, 2)), np.exp(-self.k_ye * np.abs(ye)))

    def _coursedirection_reward(self, chi_ak, u, v):
        k_cd = 1
        k_cd = 0.35
        return np.exp(-k_cd * np.power(chi_ak, 2))
        return np.maximum(np.exp(-k_cd * np.power(chi_ak, 2)), np.exp(-k_cd * np.abs(chi_ak)))

    def _oa_reward(self, sensors):
        gammainv = np.power(1 + np.abs(sensors[:, 0] * self.gamma_theta), -1)
        distelem = np.power(self.gamma_x * np.power(np.maximum(sensors[:, 1], self.epsilon), 2), -1)

        numerator = np.sum(distelem)
        denominator = np.sum(gammainv)
        return -(numerator / denominator)

        #numerator = np.sum(np.power()
        denominator = np.sum(1 + np.abs(sensors[:, 0] * self.gamma_theta))
        return -(numerator / denominator)

    def compute_reward(self, courseangle_error, crosstrack_error, action_dot0, action_dot1, collision, u_ref, u, v):
        info = {}
        if (collision == False):
            # Velocity reward
            reward_u = np.clip(np.exp(-self.k_uu * np.abs(u_ref - np.hypot(u, v))), 0, 1)
            # Action velocity gradual change reward
            reward_a0 = np.math.tanh(-self.c_action0 * np.power(action_dot0, 2)) * self.k_action0
            # Action angle gradual change reward
            reward_a1 = np.math.tanh(-self.c_action1 * np.power(action_dot1, 2)) * self.k_action1
            #self.debug_vars['reward_a0'] = reward_a0
            #self.debug_vars['reward_a1'] = reward_a1

            # Path following reward
            reward_coursedirection = self._coursedirection_reward(courseangle_error, u, v)
            reward_crosstrack = self._crosstrack_reward(crosstrack_error)
            reward_pf = ((reward_coursedirection) * (reward_crosstrack) * (reward_u))
            # reward_pf = -1 + reward_coursedirection * reward_crosstrack + reward_u
            # Obstacle avoidance reward
            reward_oa = self._oa_reward(self.sensors) / 10.0

            # Exists reward
            reward_exists = -self.lambda_reward * 1.25

            # Total non-collision reward
            reward = self.lambda_reward * reward_pf + (
                        1 - self.lambda_reward) * reward_oa + reward_exists + reward_a0 + reward_a1
            info['reward_velocity'] = np.hypot(u, v)
            info['reward_u'] = reward_u
            info['reward_a0'] = reward_a0
            info['reward_a1'] = reward_a1
            info['reward_coursedirection'] = reward_coursedirection
            info['reward_crosstrack'] = reward_crosstrack
            info['reward_oa'] = reward_oa
            info['reward_pf'] = reward_pf
            info['reward_exists'] = reward_exists
            info['reward_u_ref'] = u_ref
            self.debug_vars['reward_oa'] = reward_oa
            # print(info)

            if (np.abs(reward) > 100000 and not collision):
                print("PANIK")

        else:
            # Collision Reward
            reward = (1 - self.lambda_reward) * -2000
            info['reward_oa'] = 0
            info['reward_pf'] = 0

        # print(reward)
        info['reward'] = reward
        self.last_reward = reward
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

    @staticmethod
    def compute_obstacle_positions(sensor_angles, obstacle_pos, boat_pos):
        sensor_angle_len = len(sensor_angles)

        # Generate rotation matrix
        c, s = np.cos(sensor_angles), np.sin(sensor_angles)
        rm = np.linalg.inv(np.array([c, -s, s, c]).T.reshape(sensor_angle_len, 2, 2))

        n = obstacle_pos - boat_pos

        # obs_pos = np.zeros((sensor_angle_len, obstacle_len, 2))

        # for i in range(sensor_angle_len):
        #     for j in range(obstacle_len):
        #         obs_pos[i][j] = (rm[i].dot(n[j]))

        obs_pos = np.inner(rm, n).transpose(0, 2, 1)

        obs_pos[:, :, 1] *= -1  # y *= -1
        return obs_pos
