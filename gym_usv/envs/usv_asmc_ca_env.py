"""
@author: Alejandro Gonzalez, Ivana Collado, Sebastian
        Perez

Environment of an Unmanned Surface Vehicle with an
Adaptive Sliding Mode Controller to train collision
avoidance on the OpenAI Gym library.
"""

import gymnasium
from gymnasium import spaces
import numpy as np
from numba import njit
from scipy.spatial import distance
from collections import deque
from gym_usv.control import UsvAsmc
from .usv_ca_renderer import UsvCaRenderer
from scipy.stats import linregress

class UsvAsmcCaEnv(gymnasium.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode="rgb_array"):
        self.render_mode = render_mode
        # Integral step (or derivative) for 100 Hz
        self.integral_step = 0.1

        self.place_obstacles = True
        self.use_kinematic_model = False

        # Overall vector variables
        self.state = None

        # Obstacle variables
        self.num_obs = None
        self.obs_x = None  # array
        self.obs_y = None  # array
        self.obs_r = None  # array

        # current state variable
        self.velocity = np.zeros(3)
        self.last_velocity = np.zeros(3)
        self.position = np.zeros(3)

        self.sensor_num = 32
        # angle, distance
        self.sensors = np.zeros((self.sensor_num, 2))
        self.sensor_span = (2 / 3) * (2 * np.pi)
        self.lidar_resolution = self.sensor_span / self.sensor_num  # angle resolution in radians
        self.sector_num = 25  # number of sectors
        self.sector_size = np.floor(self.sensor_num / self.sector_num).astype(int)  # number of points per sector
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
        self.min_action0 = 1.0
        self.max_action0 = 1.2
        # angle (change to -pi and pi if necessary)
        self.min_action1 = -np.pi / 3
        self.max_action1 = np.pi / 3

        # Min and max values of the state
        self.min_u = -2.5 / 2
        self.max_u = 2.5 / 2
        self.min_v = -1.75 / 2
        self.max_v = 1.75 / 2
        self.min_r = -2.
        self.max_r = 3.5

        self.debug_vars = {}
        self.plot_vars = {}

        self.action_history_len = 8
        self.action_history = None

        # Distance to target, angle between real and target, long velocity, normal vel, ang accel, sensor data
        self.state_length = 10 + self.sensor_num

        # Min and max state vectors
        self.low_state = np.full(self.state_length, -1.0)
        self.high_state = np.full(self.state_length, 1.0)

        self.min_action = np.array([-1.0, -1.0])
        self.max_action = np.array([1.0, 1.0])

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

        self.renderer = None
        self.isopen = True

        self.renderplots = True

        self.target_point = np.zeros(2)

        self.pos_history_pos = np.zeros(3)
        self.pos_history_next = np.zeros(3)
        self.pos_history_last = np.zeros(3)
        self.pos_history_last_last = np.zeros(3)
        self.last_action = np.zeros(2)
        self.start_position = np.zeros(3)
        self.action_vel_accel = np.zeros((2, 2)) # a', a''

        self.asmc = UsvAsmc()
        self.reset()

    @staticmethod
    def _wrap_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    @staticmethod
    def _normalize_val(x, in_min, in_max):
        return UsvAsmcCaEnv._map(x, in_min, in_max, -1, 1)

    @staticmethod
    def _denormalize_val(x, out_min, out_max):
        return UsvAsmcCaEnv._map(x, -1, 1, out_min, out_max)

    @staticmethod
    def _map(x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def _normalize_state(self, state):

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
        # Change from vectors to scalars
        u, v, r = self.velocity
        x, y, psi = self.position

        self.debug_vars['x'], self.debug_vars['y'] = x, y

        action = [
            self._denormalize_val(action_in[0], self.min_action0, self.max_action0),
            self._denormalize_val(action_in[1], self.min_action1, self.max_action1)
        ]
        self.debug_vars['action0_in'] = action_in[0]
        self.debug_vars['action1_in'] = action_in[1]

        self.debug_vars['action0'] = action[0]
        self.debug_vars['action1'] = action[1]

        if self.use_kinematic_model:
            # Update rotational vel
            T = 1 / 2
            dvr = T * (action[1] - r)
            r += dvr
            r = np.clip(r, self.min_r, self.max_r)
            psi += r * self.integral_step

            du = (T * (action[0] - u))
            u = np.clip(u + du, self.min_u, self.max_u)

            x += -u * self.integral_step * -np.cos(psi)
            y += -u * self.integral_step * -np.sin(psi)

            upsilon = u, v, r
            self.debug_vars['u'] = u
            self.debug_vars['r'] = r
            eta = x, y, psi
        else:
            eta = self.position
            upsilon = self.velocity
            for _ in range(5):
                eta, upsilon = self.asmc.compute(action, np.array(eta), np.array(upsilon))

        u, v, r = upsilon
        self.velocity = upsilon
        self.position = eta
        x, y, psi = self.position
        self.pos_history_next = np.array(self.position)

        # Compute collision
        done = False
        truncated = False

        distance = np.hypot(self.obs_x - eta[0],
                            self.obs_y - eta[1]) - self.obs_r - self.boat_radius - self.safety_radius
        distance_to_obstacle = distance + self.obs_r
        distance = distance.reshape(-1)
        nearest_obstacle_distance = 0
        nearest_obstacle_angle = 0
        nearest_obstacle_radius = 0
        if distance.size == 0:
            collision = False
        else:
            nearest_obstacle = np.argmin(distance_to_obstacle)
            nearest_obstacle_distance = distance_to_obstacle[nearest_obstacle]
            nearest_obstacle_radius = self.obs_r[nearest_obstacle]

            obs_pos = [self.obs_x[nearest_obstacle], self.obs_y[nearest_obstacle]]
            nearest_obstacle_angle = self._compute_angle_to_point(self.position, obs_pos, psi)

            collision = (nearest_obstacle_distance < nearest_obstacle_radius)

        # Compute sensor readings
        self.sensors = self._compute_sensor_measurments(
            self.position,
            self.sensor_num,
            self.sensor_max_range,
            self.obs_r,
            self.lidar_resolution,
            self.obs_x,
            self.obs_y,
            self.num_obs,
            distance)
        sensors = self.sensors / [1, self.sensor_max_range]

        distance_to_target = np.hypot(self.position[0] - self.target_point[0], self.position[1] - self.target_point[1])
        angle_to_target = self._compute_angle_to_point(self.position, self.target_point, psi)
        # Compute reward
        arrived = distance_to_target < 1
        tracking_error = np.array([[np.cos(psi), np.sin(psi), 0],
                                  [-np.sin(psi), np.cos(psi), 0],
                                  [0, 0, 1]]) @ (self.target_point - self.position)
        tracking_error[2] = self._wrap_angle(tracking_error[2])
        div_fac = self.max_x ** 2 + self.max_y ** 2
        normalized_tracking_error = tracking_error / np.array([div_fac, div_fac, np.pi])

        # Nearest obstacle distance according to lidar
        nearest_obstacle_distance = np.min(sensors[:,1]) * self.sensor_max_range

        reward, info = self.compute_reward(normalized_tracking_error,
                                           angle_to_target,
                                           arrived,
                                           action,
                                           self.action_history,
                                           distance_to_obs=nearest_obstacle_distance,
                                           obs_radius=nearest_obstacle_radius)

        self.debug_vars['reward'] = reward
        self.last_velocity = self.velocity
        self.last_action = action_in.copy()

        self.state = np.hstack((
            u / self.max_u, r / self.max_r,
            normalized_tracking_error,
            np.hypot(tracking_error[0], tracking_error[1]) / 45,
            angle_to_target / np.pi,
            np.average(self.action_history, axis=0) / np.maximum(self.max_action0, self.max_action1),
            nearest_obstacle_distance,
            sensors[:,1]
        ))

        #if np.max(np.abs(self.state)) > 1.0:
            #print(self.state)

        self.action_history.append(action)

        self.debug_vars['p'] = ", ".join([str(np.round(p, 3)) for p in self.position])

        if arrived:
            done = True

        #if collision:
        #    done = True

        if np.hypot(tracking_error[0], tracking_error[1]) > 40:
            done = True
            reward -= 100

        # Reshape state
        self.state = self.state.reshape(self.observation_space.shape[0]).astype(np.float32)
        info['completed'] = arrived

        if np.max(np.abs(self.position)) > 100:
            done = True
            truncated = True

        self.plot_vars['action0'] = action_in[0]
        self.plot_vars['action1'] = action_in[1]
        self.plot_vars['u'] = u
        self.plot_vars['r'] = r

        return self.state, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.renderer:
            self.renderer.reset()
        x = np.random.uniform(low=self.min_x, high=self.max_x)
        y = np.random.uniform(low=self.min_y, high=self.min_y + 5)
        theta = np.random.uniform(low=-np.pi / 4, high=np.pi / 4)
        self.position = [x, y, theta]
        self.start_position = self.position
        self.last_pos = np.array([x, y, theta])
        self.action_vel_accel = np.zeros((2, 2))  # a', a''

        self.action_history = deque([np.zeros(2)] * self.action_history_len, maxlen=self.action_history_len)

        # number of obstacles 
        self.num_obs = np.clip(int(np.random.normal(loc=15, scale=6)), 0, 30)
        if not self.place_obstacles:
            self.num_obs = 0

        self.target_point = np.random.uniform(
            low=(self.min_x, self.max_y - 5),
            high=(self.max_x - 10, self.max_y - 1),
            size=2)
        self.target_point = np.hstack([self.target_point, np.zeros(1)]) # Target angle is always 0

        # TODO maybe change distribution
        min_radius = 0.75
        max_radius = 0.77
        # row_n = int(np.random.uniform(4, 10))
        # col_n = int(np.random.uniform(4, 5))
        # if options:
        #     row_n = options['row_n']
        #     col_n = options['col_n']
        #obs_x = np.linspace(self.min_x, self.max_x, row_n)
        #obs_y = np.linspace(self.min_y + 6, self.max_y - 6, col_n)
        #m = np.meshgrid(obs_x, obs_y)
        # obstacles = np.random.uniform(
        #     low=np.tile((self.min_x, self.min_y, min_radius), (self.num_obs, 1)).T,
        #     high=np.tile((self.max_x, self.max_y, max_radius), (self.num_obs, 1)).T,
        #     size=(3, self.num_obs)).T
        #
        #obs_pos = np.random.normal(loc=(np.array(self.target_point[:2]) + np.array(self.start_position[:2])) / 2, scale=5.0, size=(self.num_obs, 2))
        #
        #self.num_obs = len(obstacles)
        #self.obs_x = obstacles[:, 0]
        #self.obs_y = obstacles[:, 1]
        # self.obs_r = obstacles[:, 2]
        #
        #self.obs_x, self.obs_y = obs_pos[:,0], obs_pos[:,1]
        #self.obs_x = np.random.uniform(low=self.min_x, high=self.max_x, size=self.num_obs)
        #self.obs_y = np.random.uniform(low=self.min_y + 6, high=self.max_y - 6, size=self.num_obs)
        self.obs_x = np.array(np.average([self.position[0], self.target_point[0]]))
        self.obs_y = np.array(np.average([self.position[1], self.target_point[1]]))
        self.obs_r = np.array([3])
        self.num_obs = 1
        #self.obs_r = np.random.uniform(low=min_radius, high=max_radius, size=self.num_obs)
        #self.num_obs = row_n * col_n
        #obs_pos = np.array(m).T.reshape(-1, 2)
        #self.obs_x, self.obs_y = obs_pos[:,0], obs_pos[:,1]
        #self.obs_r = np.full(self.num_obs, 0.5)

        if options:
            self.renderplots = options['renderplots']

        self.total_reward = 0

        distance = np.hypot(self.obs_x - self.position[0],
                            self.obs_y - self.position[1]) - self.obs_r - self.boat_radius - (self.safety_radius + 0.35)
        distance = distance.reshape(-1)

        self.asmc = UsvAsmc()

        # Delete all obstacles within boat radius
        elems_to_delete = np.flatnonzero(distance < 0)
        self.obs_x = np.delete(self.obs_x, elems_to_delete).reshape(-1, 1)
        self.obs_y = np.delete(self.obs_y, elems_to_delete).reshape(-1, 1)
        self.obs_r = np.delete(self.obs_r, elems_to_delete).reshape(-1, 1)
        self.num_obs -= elems_to_delete.size

        distance = np.hypot(self.obs_x - self.target_point[0],
                            self.obs_y - self.target_point[1]) - self.obs_r - self.boat_radius - (
                               self.safety_radius + 0.35)
        distance = distance.reshape(-1)
        elems_to_delete = np.flatnonzero(distance < 0)
        self.obs_x = np.delete(self.obs_x, elems_to_delete).reshape(-1, 1)
        self.obs_y = np.delete(self.obs_y, elems_to_delete).reshape(-1, 1)
        self.obs_r = np.delete(self.obs_r, elems_to_delete).reshape(-1, 1)

        self.num_obs -= elems_to_delete.size

        self.state = np.zeros(self.state_length)

        state, _, _, _, _ = self.step([-1, 0])
        return state, {}

    @staticmethod
    def _compute_angle_to_point(pos, target, psi):
        return UsvAsmcCaEnv._wrap_angle(
            np.arctan2(target[1] - pos[1], target[0] - pos[0]) - psi
        )

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

        obstacle_positions = np.hstack((posx, posy))[obs_order]

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
            for j in range(num_obs):
                (obs_x, obs_y) = ned_obstacle_positions[i][j]
                obs_idx = obs_order[j]

                if obs_x < 0:
                    # Obstacle is behind sensor
                    # sensors[i][1] = sensor_max_range
                    continue

                delta = (radius[obs_idx] * radius[obs_idx]) - (obs_y * obs_y)
                if delta < 0:
                    continue

                new_distance = obs_x - np.sqrt(delta)
                if new_distance < sensor_max_range:
                    new_distances[i] = min(new_distance[0], sensors[i][1])
                    break
        return new_distances

    ## TODO Add screen space transformation functions
    def render(self, info=None, draw_obstacles=True, show_debug_vars=True):
        if self.renderer is None:
            self.renderer = UsvCaRenderer()

        return self.renderer.render(
            self.position,
            self.sensors,
            self.target_point,
            self.obs_x,
            self.obs_y,
            self.obs_r,
            self.debug_vars,
            show_debug_vars,
            self.plot_vars,
            self.renderplots,
            self.render_mode
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def _oa_reward(self, sensors):
        gamma_theta = 4.0
        gamma_x = 0.0005
        epsilon = 1
        sensors *= self.sensor_max_range

        gammainv = np.power(1 + np.abs(sensors[:, 0] * gamma_theta), -1)
        distelem = np.power(gamma_x * np.power(np.maximum(sensors[:, 1], epsilon), 2), -1)

        numerator = np.sum(distelem)
        denominator = np.sum(gammainv)
        return -(numerator / denominator) / 5e4

        # numerator = np.sum(np.power()
        denominator = np.sum(1 + np.abs(sensors[:, 0] * gamma_theta))
        return -(numerator / denominator)

    def compute_reward(self,
                       tracking_error,
                       angle_to_target,
                       arrived, action,
                       action_history,
                       distance_to_obs,
                       obs_radius):
        info = {}
        te = tracking_error.copy()
        te[2] = 0
        r_tracking_error = -np.hypot(te[0], te[1]) * 35 - np.abs(angle_to_target / np.pi) * 2
        #print(r_tracking_error)
        reward = r_tracking_error
        r_delta = np.sum(np.abs(np.array(action) - np.average(action_history, axis=0)))
        reward -= r_delta * 5
        #reward -= np.abs(action[1]) * 0.2
        reward -= action[1] ** 2 * 4 + (np.abs(action[0]) - 1) ** 2

        reward_zone_r = 3.25
        punishment_zone_r = 1.5
        phi_rz = 0.1
        phi_pz = 0.2
        obs_oa_r = 0
        #print(distance_to_obs)
        if punishment_zone_r < distance_to_obs < reward_zone_r:
            # Rewards zone
            #print('reward')
            obs_oa_r = np.maximum(phi_rz,
                          1.0 / np.tanh(
                              (distance_to_obs - punishment_zone_r) /
                              (reward_zone_r - punishment_zone_r))).item()
            obs_oa_r = np.log(obs_oa_r)
        elif 0 < distance_to_obs < punishment_zone_r:
            # Punishment zone
            #print('punish')
            obs_oa_r = np.maximum(phi_pz,
                            1.0 / np.tanh((distance_to_obs - obs_radius) / (punishment_zone_r - obs_radius))).item()
            obs_oa_r = -np.log(obs_oa_r)
            #print(f"Distance: {distance_to_obs} Obs: {obs_oa_r}")
        elif distance_to_obs < 0:
            # Obstacle
            #print('obstacle')
            obs_oa_r = -18

        #print(obs_oa_r)

        #self.plot_vars['obs_oa_r'] = obs_oa_r / 10
        reward += obs_oa_r * 0.5
        reward_action = action[0] ** 2 / 5 + np.abs(action[1]) ** 2 * 1.5
        #reward -= reward_action

        if arrived:
            reward += 200

        #print(obs_oa_r)

        #self.plot_vars['tracking_error'] = r_tracking_error / 10
        #self.plot_vars['reward'] = reward / 10
        return reward, info

    def test_reward_function(self, angle_to_target, last_angle_to_target, action_history):
        delta_ang = angle_to_target - last_angle_to_target
        reward = 1 - (0.2 * angle_to_target ** 2 + 0.1 * delta_ang ** 2 + 0.001 * action_history[-1][1] ** 2)
        reward = 1 - angle_to_target
        return reward

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
