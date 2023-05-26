import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_usv.envs import UsvAsmcCaEnv
from . import simple_env_visualizer

class UsvSimpleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode="rgb_array"):
        self.sensor_count = 32
        self.sensor_span = (2 / 3) * (2 * np.pi)
        self.sensor_max_range = 100
        self.sensor_resolution = self.sensor_span / self.sensor_count

        # state: u, v, r
        # target: angle to target, distance to target
        # environment: sensor measurements
        # self.observation_space = spaces.Dict(
        #     {
        #         "state": spaces.Box(-1, 1, shape=(3,), dtype=np.float64),
        #         "target": spaces.Box(-1, 1, shape=(2,), dtype=np.float64),
        #         # "environment": spaces.Box(-1, 1, shape=(32,), dtype=np.float32),
        #     }
        # )
        self.observation_space = spaces.Box(-1, 1, shape=(15 + self.sensor_count,), dtype=np.float32)

        # dU, dR
        self.action_space = spaces.Box(np.array([0.2, -1]), np.array([1, 1]), shape=(2,), dtype=np.float32)
        # u, v, r
        self.max_action = np.array([3, 0, 3])
        self.reference_velocity = 0
        self.max_acceleration = np.array([1.75, 0, 3])
        self.dt = (1 / 25)

        # Current state
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)

        self.last_action = np.zeros(3)

        # Obstacle states
        self.obstacle_n = 0
        self.obstacle_positions = None
        self.obstacle_radius = None
        self.sensor_data = np.zeros(shape=(self.sensor_count, 2))

        self.ignore_obstacles = False

        self.path_start = np.zeros(2)
        self.path_end = np.zeros(2)
        self.progress = 0
        self.target_position = np.zeros(2)

        self.env_bounds = (0, 10)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.renderer = simple_env_visualizer.SimpleEnvVisualizer(self.env_bounds, render_mode)


    @staticmethod
    def _wrap_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _get_angle_to_target(self):
        delta_pos = (self.target_position - self.position[:2])
        return self._wrap_angle(np.arctan2(delta_pos[1], delta_pos[0]) - self.position[2])


    def _get_target_state(self):
        # Compute angle and distance to target
        distance = np.hypot(*(self.position[:2] - self.target_position))
        angle = self._get_angle_to_target()
        delta_pos = (self.target_position - self.position[:2])
        # print(angle)
        ye = self._get_ye()
        return np.array([angle, distance, ye, self.reference_velocity]) / \
            [np.pi, np.hypot(self.env_bounds[1], self.env_bounds[1]), 10, 10]

    def _get_sensor_state(self):
        return self.sensor_data[:, 1] / self.sensor_max_range

    def _get_kinem_obs(self):
        return np.hstack((
            self.max_action / 10,
            self.max_acceleration / 10
        ))

    def _get_obs(self, action):
        sensor_state = self._get_sensor_state()
        target_state = self._get_target_state()
        action_state = action[[0, 2]] / self.max_action[[0, 2]]
        kinem_state = self._get_kinem_obs()
        return np.hstack((self.velocity / 10, target_state, action_state, kinem_state, sensor_state)).astype(np.float32)
        # return {
        #     "state": self.velocity,
        #     "target": self._get_target_state() / [np.pi, np.hypot(self.env_bounds[1], self.env_bounds[1])]
        # }

    def _get_info(self, reward, action):
        return {
            'position': self.position,
            'velocity': self.velocity,
            'path_start': self.path_start,
            'path_end': self.path_end,
            'reward': reward,
            'action0': action[0],
            'action1': action[2],
            'left_thruster': 0,
            'right_thruster': 0,
            'ye': self._get_ye(),
            'angle_to_target': self._get_target_state()[0],
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        return self.renderer.render_frame(
            self.position,
            self.target_position,
            self.sensor_data,
            self.obstacle_positions,
            self.obstacle_radius,
            self.path_start,
            self.path_end
        )

    def _get_ye(self):
        a_k = np.arctan2(self.path_end[1] - self.path_start[1], self.path_end[0] - self.path_start[0])
        return -(self.position[0] - self.path_start[0]) * np.sin(a_k) + (
                    self.position[1] - self.path_start[1]) * np.cos(
            a_k)

    def _get_closest_point(self):
        x1, y1 = self.path_start
        x2, y2 = self.path_end
        x3, y3, _ = self.position
        dx, dy = x2 - x1, y2 - y1
        det = dx * dx + dy * dy
        a = (dy * (y3 - y1) + dx * (x3 - x1)) / det
        a += 0.005
        a = np.clip(a, self.progress, 1)
        return np.array([x1 + a * dx, y1 + a * dy]), a

    def _get_reward(self, action):
        target_info = self._get_target_state()

        min_sensor = np.min(self.sensor_data[:, 1])
        colision_reward = 0
        if min_sensor < 0.2 and not self.ignore_obstacles:
            colision_reward = -20

        arrived_reward = 0

        delta_action = np.abs(self.last_action - action)
        angle = self._get_angle_to_target()

        ye_reward = np.exp(-np.abs(self._get_ye()) / 0.5) * 1.25
        angle_to_target_reward = np.exp(-np.abs(angle))
        #print(angle_to_target_reward)
        angle_action_reward = -np.abs(self.last_action[2] / self.max_action[2]) * 0.5
        delta_action_reward = np.exp(-np.sum(np.abs(delta_action))) * 0.15
        delta_action_reward = -(np.sum(np.abs(delta_action)) / 2) * 0.15

        angle_action_reward = 0

        velocity_towards_target = (np.cos(angle) * self.last_action[0])
        velocity_track_reward = np.exp(-np.abs(self.last_action[0] - self.reference_velocity)) * 0.05
        reward = arrived_reward + \
                 colision_reward + \
                 ye_reward + \
                 angle_to_target_reward + \
                 velocity_track_reward + \
                 delta_action_reward

        #print(reward)
        reward_info = {
            'ye_reward': ye_reward,
            'angle_to_target_reward': angle_to_target_reward,
            'angle_action_reward': angle_action_reward,
            'delta_action_reward': delta_action_reward,
            'delta_action': np.sum(np.abs(delta_action), axis=0),
            'velocity_track_reward': velocity_track_reward,
            'velocity_towards_target': velocity_towards_target,
            'reference_velocity': self.reference_velocity,
            'reward_velocity': self.last_action[0],
            'reference_velocity_error': self.last_action[0] - self.reference_velocity
        }

        return reward, reward_info

    def _compute_sensor_measurment(self):
        obs_x, obs_y = self.obstacle_positions[:, 0], self.obstacle_positions[:, 1]
        obstacle_distances = np.hypot(obs_x - self.position[0],
                                      obs_y - self.position[1]) - self.obstacle_radius

        obstacle_distances = obstacle_distances.reshape(-1)

        sensors = UsvAsmcCaEnv.compute_sensor_measurments(
            self.position,
            self.sensor_count,
            self.sensor_max_range,
            self.obstacle_radius,
            self.sensor_resolution,
            obs_x.reshape(-1, 1),
            obs_y.reshape(-1, 1),
            self.obstacle_n,
            obstacle_distances
        )

        if self.ignore_obstacles:
            obstacle_distances = np.ones(1)
            sensors[:,1] = np.full(self.sensor_count, self.sensor_max_range)

        return obstacle_distances, sensors

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Set v velocity to 0
        self.velocity[1] = 0

        self.path_start = self.np_random.normal(scale=0.5, size=2) + np.array(
            [self.env_bounds[1], self.env_bounds[1]]) / 2
        self.position = np.hstack(
            (self.np_random.normal(self.path_start, scale=0.75), self.np_random.uniform(-np.pi, np.pi)))
        self.position = np.hstack((self.path_start, self.np_random.uniform(-np.pi, np.pi)))

        # Chose random angle and distance for path
        angle = self.np_random.uniform(-np.pi, np.pi)
        dist = self.np_random.uniform(100, 110)
        self.path_end = self.path_start + np.array([np.cos(angle), np.sin(angle)]) * dist

        self.target_position = self.np_random.uniform(*self.env_bounds, size=2)
        self.velocity = self.np_random.uniform(0.0, 0.15, size=3)
        self.progress = 0

        self.max_action = self.np_random.uniform(1.50, 3, size=3)
        self.max_action[2] = self.np_random.uniform(3, 6)
        self.reference_velocity = self.np_random.uniform(0.75, self.max_action[0])
        # self.max_acceleration = self.np_random.uniform(0.5, 0.75, size=3)
        self.max_acceleration[1] = 0
        self.max_action[1] = 0

        # Generate obstacle positions
        self.obstacle_n = self.np_random.integers(15, 30)
        self.obstacle_positions = self.np_random.uniform(*self.env_bounds, size=(self.obstacle_n, 2))

        # Remove obstacles next to usv position or target position
        distance_to_position = np.hypot(self.position[0] - self.obstacle_positions[:, 0],
                                        self.position[1] - self.obstacle_positions[:, 1])
        distance_to_target = np.hypot(self.target_position[0] - self.obstacle_positions[:, 0],
                                      self.target_position[1] - self.obstacle_positions[:, 1])
        elems_to_delete = np.hstack(
            (np.flatnonzero(distance_to_position < 0.5), np.flatnonzero(distance_to_target < 0.5)))
        self.obstacle_positions = np.delete(self.obstacle_positions, elems_to_delete, axis=0)
        self.obstacle_n = self.obstacle_positions.shape[0]

        if self.obstacle_n == 0:
            # Place one obstacle just so we don't get a crash :(
            print("ADDING AN OBSTACLE BACK IN")
            self.obstacle_positions = self.np_random.uniform(*self.env_bounds, size=(1, 2))
            self.obstacle_n = 1

        if options:
            if 'place_obstacles_on_path' in options and options['place_obstacles_on_path']:
                # Place n obstacles on path
                n_obs = options['place_obstacles_on_path']

                mag = self.np_random.uniform(0, np.hypot(*self.env_bounds), n_obs)

                line_x = self.np_random.normal(np.cos(angle) * mag + self.path_start[0], 1)
                line_y = self.np_random.normal(np.sin(angle) * mag + self.path_start[1], 1)

                path_obstacles = np.hstack((line_x.reshape(-1,1), line_y.reshape(-1,1)))
                self.obstacle_positions = np.concatenate((self.obstacle_positions, path_obstacles))
                self.obstacle_n = self.obstacle_positions.shape[0]

        self.obstacle_radius = self.np_random.uniform(0.15, 0.5, size=self.obstacle_n)

        obs = self._get_obs(action=np.zeros(3))
        info = self._get_info(-1, np.zeros(3))

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def step(self, action, update_position=True):
        action = np.insert(arr=action, obj=[1], values=0)  # Insert v 0
        action = self.max_action * action
        # print(action)
        # print(self.position)

        if update_position:
            action = 0.8 * self.last_action + 0.2 * action

            # Update velocity based off current velocity and velocity action
            delta_velocity = np.clip(action - self.velocity, -self.max_acceleration, self.max_acceleration)
            self.velocity = np.clip(self.velocity + delta_velocity, -self.max_action, self.max_action)
            theta = self.position[2]
            rotated_vel = np.array([self.velocity[0] * np.cos(theta), self.velocity[0] * np.sin(theta), self.velocity[2]])
            self.position = self.position + rotated_vel * self.dt
            # self.position[:2] = np.clip(self.position[:2], *self.env_bounds)
            # self.position[2] = 0

        self.target_position, self.progress = self._get_closest_point()
        obstacle_distance, self.sensor_data = self._compute_sensor_measurment()

        if self.render_mode == "human":
            self._render_frame()

        terminated = np.min(obstacle_distance) < 0.05 and not self.ignore_obstacles
        # print(f"Progress: {self.progress} Dist: {dist_to_target}")
        truncated = np.any((self.position[:2] > self.env_bounds[1]) | (self.position[:2] < 0))

        obs = self._get_obs(self.last_action)
        reward, reward_info = self._get_reward(action)
        # print(reward)
        info = self._get_info(reward, action)
        info.update(reward_info)
        self.last_action = action
        # print(reward)

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
