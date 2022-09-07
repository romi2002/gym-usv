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


class UsvAsmcCaEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, config=None):
        # Integral step (or derivative) for 100 Hz
        self.integral_step = 0.01

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
        self.min_action0 = 0.5
        self.max_action0 = 1.5
        # angle (change to -pi and pi if necessary)
        self.min_action1 = -np.pi
        self.max_action1 = np.pi

        # Reward associated functions anf gains
        self.k_ye = 0.5  # Crosstracking reward

        self.k_uu = 2.0  # Velocity Reward
        self.w_u = 1  # Velocity reward

        self.gamma_theta = 8.0  # 4.0
        self.gamma_x = 0.05  # 0.005
        self.epsilon = 4.0
        self.lambda_reward = 0.85

        self.w_action0 = 0.2
        self.w_action1 = 0.2
        # Action gradual change reward
        self.c_action0 = 1. / np.power((self.max_action0 / 2 - self.min_action0 / 2) / self.integral_step, 2)
        self.c_action1 = 1. / np.power((self.max_action1 / 2 - self.min_action1 / 2) / self.integral_step, 2)
        self.k_action0 = 2.5
        self.k_action1 = 2.25

        # Min and max values of the state
        self.min_u = -2.0
        self.max_u = 2.0
        self.min_v = -1.5
        self.max_v = 1.5
        self.min_r = -4.
        self.max_r = 4.

        self.min_lookahead_error = -np.pi
        self.max_lookahead_error = np.pi
        self.min_courseerror = -10
        self.max_courseerror = 10

        self.min_u_ref = 0.75
        self.max_u_ref = 1.25
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

        self.lookahead_distance = 2.5

        self.screen = None
        self.font = None
        self.clock = None
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

        self.debug_vars['u'] = u
        self.debug_vars['v'] = v
        self.debug_vars['r'] = r

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

        action = [
            self._denormalize_val(action_in[0], self.min_action0, self.max_action0),
            self._denormalize_val(action_in[1], self.min_action1, self.max_action1)
        ]

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
            position,
            self.sensor_num,
            self.sensor_max_range,
            self.radius,
            self.lidar_resolution,
            self.posx,
            self.posy,
            self.num_obs,
            distance)

        # Feasability pooling: compute sectors
        sectors = self._compute_feasability_pooling(self.sector_num, self.sector_size, self.sensor_max_range,
                                                    self.lidar_resolution, self.boat_radius + self.safety_radius,
                                                    self.sensors)
        self.sectors = sectors
        sectors = np.clip((sectors / self.sensor_max_range), 0, 1)

        # Compute lookahead course error
        # compute angle from north to tangent line at lookahead point
        gamma_p = np.arctan2(self.path_deriv(x_d) - self.path_deriv(x_d + 0.1), 0.1).item()
        courseangle_error = self._wrap_angle(gamma_p - psi)

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
        if collision:
            done = True

        if x > self.max_x:
            done = True

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
        # number of obstacles 
        self.num_obs = np.random.randint(low=5, high=25)

        self.path, self.waypoints = generate_path(np.array([x_0, y_0]), np.random.randint(8, 15))
        self.path_deriv = self.path.derivative()
        obstacles = place_obstacles(self.path, self.waypoints, self.num_obs)
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

        self.lambda_reward = 10 ** -np.random.gamma(1, 1/2)
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

    def _transform_points(self, points, x, y, angle):
        if angle is not None:
            s, c = (np.sin(angle), np.cos(angle))
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
            # angle = np.where(np.greater(np.abs(angle), np.pi), np.sign(angle) * (np.abs(angle) - 2 * np.pi), angle)
            initial = ((y - self.min_y) * scale, (x - self.min_x) * scale)
            m = np.math.tan(angle)
            x_f = sensors[i][1] * np.math.cos(angle) + x - self.min_x
            y_f = sensors[i][1] * np.math.sin(angle) + y - self.min_y
            final = (y_f * scale, x_f * scale)
            section = np.floor(i / self.sector_size).astype(int)

            color = (0, 255, 0)

            if sectors[section] < self.sensor_max_range:
                color = (255, 0, 0)
                if (i % 10 == 0):
                    color = (255, 0, 255)
            elif (i % 10 == 0):
                color = (0, 0, 255)

            pygame.draw.line(self.surf, color, initial, final)

    def _draw_boat(self, scale, position):
        import pygame
        boat_width = 15
        boat_height = 20

        l, r, t, b, c, m = -boat_width / 2, boat_width / 2, boat_height, 0, 0, boat_height / 2
        boat_points = [(l, b), (l, m), (c, t), (r, m), (r, b)]
        boat_points = [(x, y - boat_height / 2) for (x, y) in boat_points]
        boat_points = self._transform_points(boat_points, (position[1] - self.min_y) * scale,
                                             (position[0] - self.min_x) * scale, -position[2])

        pygame.draw.polygon(self.surf, (3, 94, 252), boat_points)

    def _draw_obstacles(self, scale):
        import pygame
        for i in range(self.num_obs):
            obs_points = [((self.posy[i][0] - self.min_y) * scale, (self.posx[i][0] - self.min_x) * scale)]
            pygame.draw.circle(self.surf, (0, 0, 255), obs_points[0], self.radius[i][0] * scale)

    def _draw_highlighted_sectors(self, scale):
        import pygame
        x = self.position[0]
        y = self.position[1]
        psi = self.position[2]

        angle = -(2 / 3) * np.pi + psi
        angle = np.where(np.greater(np.abs(angle), np.pi), np.sign(angle) * (np.abs(angle) - 2 * np.pi), angle)
        for i in range(self.sector_num + 1):
            initial = ((y - self.min_y) * scale, (x - self.min_x) * scale)
            x_f = self.sensor_max_range * np.math.cos(angle) + x - self.min_x
            y_f = self.sensor_max_range * np.math.sin(angle) + y - self.min_y
            final = (y_f * scale, x_f * scale)
            pygame.draw.line(self.surf, (0, 0, 0), initial, final, width=2)
            angle = angle + self.sensor_span / self.sector_num
            angle = np.where(np.greater(np.abs(angle), np.pi), np.sign(angle) * (np.abs(angle) - 2 * np.pi), angle)

    def _draw_path(self, scale, path):
        import pygame
        # Draw path
        path_x = np.linspace(self.waypoints[0][0], self.waypoints[-1][0])
        path_y = path(path_x)
        pygame.draw.lines(self.surf, (0, 255, 0), False,
                          [tuple(p) for p in
                           np.vstack([(path_y - self.min_y) * scale, (path_x - self.min_x) * scale]).T], width=2)

    ## TODO Add screen space transformation functions
    def render(self, mode='human', info=None, draw_obstacles=True, show_debug_vars=True):
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
        if self.font is None:
            self.font = pygame.font.SysFont(None, 48)

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        self._draw_sectors(scale, position, sensors, sectors)

        if (draw_obstacles):
            self._draw_obstacles(scale)

        self._draw_boat(scale, position)

        self._draw_path(scale, self.path)

        # Draw target point
        _, _, x_t, y_t, _ = self.target
        pygame.draw.circle(self.surf, (100, 0, 255), ((y_t - self.min_y) * scale, (x_t - self.min_x) * scale), radius=5)

        # Draw safety radius
        safety_radius = (self.boat_radius + self.safety_radius) * scale
        safety = ((y - self.min_y) * scale, (x - self.min_x) * scale)
        pygame.draw.circle(self.surf, (255, 0, 0), safety, safety_radius, width=3)

        self.surf = pygame.transform.flip(self.surf, False, True)

        text_start_pos = (20, 20)
        if show_debug_vars:
            for key, var in self.debug_vars.items():
                text_img = self.font.render(f"{key}: {round(var,4)}", True, (0,0,0))
                self.surf.blit(text_img, text_start_pos)
                text_start_pos = text_start_pos[0], text_start_pos[1] + 30


        if mode == "human":
            self.screen.blit(self.surf, (0, 0))
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
        return np.maximum(np.exp(-self.k_ye * np.power(ye, 2)), np.exp(-self.k_ye * np.abs(ye)))

    def _coursedirection_reward(self, chi_ak, u, v):
        k_cd = 1
        return np.maximum(np.exp(-k_cd * np.power(chi_ak, 2)), np.exp(-k_cd * np.abs(chi_ak)))

    def _oa_reward(self, sensors):
        numerator = np.sum(np.power(self.gamma_x * np.power(np.maximum(sensors[:, 1], self.epsilon), 2), -1))
        denominator = np.sum(1 + np.abs(sensors[:, 0] * self.gamma_theta))
        return -(numerator / denominator)

    def compute_reward(self, courseangle_error, crosstrack_error, action_dot0, action_dot1, collision, u_ref, u, v):
        info = {}
        if (collision == False):
            # Velocity reward
            reward_u = np.clip(np.exp(-self.k_uu * np.abs(u_ref - np.hypot(u, v))), -1, 1) * 0
            # Action velocity gradual change reward
            reward_a0 = np.math.tanh(-self.c_action0 * np.power(action_dot0, 2)) * self.k_action0 * 0
            # Action angle gradual change reward
            reward_a1 = np.math.tanh(-self.c_action1 * np.power(action_dot1, 2)) * self.k_action1 * 0
            self.debug_vars['reward_a0'] = reward_a0
            self.debug_vars['reward_a1'] = reward_a1

            # Path following reward
            reward_coursedirection = self._coursedirection_reward(courseangle_error, u, v)
            reward_crosstrack = self._crosstrack_reward(crosstrack_error)
            reward_pf = (-1 + (reward_coursedirection + 1) * (reward_crosstrack + 1))
            # reward_pf = -1 + reward_coursedirection * reward_crosstrack + reward_u
            # Obstacle avoidance reward
            reward_oa = self._oa_reward(self.sensors) * 0.5

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
            # print(info)

            if (np.abs(reward) > 100000 and not collision):
                print("PANIK")

        else:
            # Collision Reward
            reward = (1 - self.lambda_reward) * -2000

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
