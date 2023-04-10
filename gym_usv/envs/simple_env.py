import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from gym_usv.envs import UsvAsmcCaEnv

class UsvSimpleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.sensor_count = 32
        self.sensor_span = (2/3) * (2*np.pi)
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
        self.observation_space = spaces.Box(-1, 1, shape=(5 + self.sensor_count,), dtype=np.float32)

        # dU, dR
        self.action_space = spaces.Box(np.array([0.2, -1]), np.array([1, 1]), shape=(2,), dtype=np.float32)
        # u, v, r
        self.max_action = np.array([5, 0, 5])
        self.max_acceleration = np.array([0.75, 0, 5])
        self.dt = (1 / 50)

        # Current state
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)

        self.last_action = np.zeros(2)

        # Obstacle states
        self.obstacle_n = 0
        self.obstacle_positions = None
        self.obstacle_radius = None
        self.sensor_data = np.zeros(shape=(self.sensor_count, 2))

        self.target_position = np.zeros(2)

        self.env_bounds = (0, 10)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.window_size = 512

    @staticmethod
    def _wrap_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _get_target_state(self):
        # Compute angle and distance to target
        distance = np.hypot(*(self.position[:2] - self.target_position))
        angle = self._wrap_angle(np.arctan2(*(self.target_position - self.position[:2])) + self.position[2] - np.pi/2)
        return np.array([angle, distance]) / [np.pi, np.hypot(self.env_bounds[1], self.env_bounds[1])]

    def _get_sensor_state(self):
        return self.sensor_data[:, 1] / self.sensor_max_range

    def _get_obs(self):
        sensor_state = self._get_sensor_state()
        target_state = self._get_target_state()
        return np.hstack((self.velocity / 10, target_state, sensor_state)).astype(np.float32)
        # return {
        #     "state": self.velocity,
        #     "target": self._get_target_state() / [np.pi, np.hypot(self.env_bounds[1], self.env_bounds[1])]
        # }

    def _get_info(self):
        return {

        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        scale = self.window_size / self.env_bounds[1]

        # Draw the target
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self.target_position * scale,
            10
        )

        # Draw sensor lines
        x, y, psi = self.position
        for i, (angle, distance) in enumerate(self.sensor_data):
            # angle = np.where(np.greater(np.abs(angle), np.pi), np.sign(angle) * (np.abs(angle) - 2 * np.pi), angle)
            initial = np.array([x, y])
            x_f = distance * np.math.cos(angle) + x
            y_f = distance * np.math.sin(angle) + y
            final = np.array([x_f, y_f])

            color = (0, 255, 0)

            pygame.draw.line(canvas,
                             color,
                             initial * scale,
                             final * scale)


        # Draw the agent
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            self.position[:2] * scale,
            10
        )

        # Draw "front"
        offset = 0.1
        front_offset = np.array([np.cos(self.position[2]) * offset, np.sin(self.position[2]) * offset])
        pygame.draw.circle(
            canvas,
            (100, 100, 0),
            (front_offset + self.position[:2]) * scale,
            8
        )

        # Draw obstacles
        for i, pos in enumerate(self.obstacle_positions):
            radius = self.obstacle_radius[i]
            pygame.draw.circle(
                canvas,
                (0, 100, 0),
                pos * scale,
                radius * scale
            )


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _get_reward(self):
        target_info = self._get_target_state()

        min_sensor = np.min(self.sensor_data[:, 1])
        colision_reward = 0
        if min_sensor < 0.2:
            colision_reward = -75

        arrived_reward = 0
        if target_info[1] < 0.1:
            arrived_reward = 500

        reward = -target_info[1] / 5 + colision_reward - np.abs(self.last_action[1]) + arrived_reward
        reward = arrived_reward + colision_reward - target_info[1] / 5
        reward = arrived_reward + colision_reward - 0.75 - target_info[1] / 10

        return reward
        return -np.abs(target_info[0]) - target_info[1]  # Use distance to target


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
            obs_x.reshape(-1,1),
            obs_y.reshape(-1,1),
            self.obstacle_n,
            obstacle_distances
        )

        return obstacle_distances, sensors
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Set v velocity to 0
        self.velocity[1] = 0

        # Random start position and initial velocity
        self.position = self.np_random.uniform(*self.env_bounds, size=3)
        self.position[2] = self.np_random.uniform(-np.pi, np.pi)
        self.target_position = self.np_random.uniform(*self.env_bounds, size=2)
        self.velocity = self.np_random.uniform(-0.5, 0.5, size=3)

        self.max_acceleration = self.np_random.uniform(0.25, 1.75, size=3)
        self.max_acceleration[1] = 0

        # Generate obstacle positions
        self.obstacle_n = self.np_random.integers(5, 15)
        self.obstacle_positions = self.np_random.uniform(*self.env_bounds, size=(self.obstacle_n, 2))

        # Remove obstacles next to usv position or target position
        distance_to_position = np.hypot(self.position[0] - self.obstacle_positions[:,0], self.position[1] - self.obstacle_positions[:,1])
        distance_to_target = np.hypot(self.target_position[0] - self.obstacle_positions[:,0], self.target_position[1] - self.obstacle_positions[:,1])
        elems_to_delete = np.hstack((np.flatnonzero(distance_to_position < 0.5), np.flatnonzero(distance_to_target < 0.5)))
        self.obstacle_positions = np.delete(self.obstacle_positions, elems_to_delete, axis=0)
        self.obstacle_n = self.obstacle_positions.shape[0]

        if self.obstacle_n == 0:
            # Place one obstacle just so we don't get a crash :(
            print("ADDING AN OBSTACLE BACK IN")
            self.obstacle_positions = self.np_random.uniform(*self.env_bounds, size=(1, 2))
            self.obstacle_n = 1

        self.obstacle_radius = self.np_random.uniform(0.15, 0.5, size=self.obstacle_n)

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def step(self, action):
        action = np.insert(arr=action, obj=[1], values=0) # Insert v 0
        action = self.max_action * action
        self.last_action = action
        #print(action)

        # Update velocity based off current velocity and velocity action
        delta_velocity = np.clip(action - self.velocity, -self.max_acceleration, self.max_acceleration)
        self.velocity = np.clip(self.velocity + delta_velocity, -self.max_action, self.max_action)
        theta = self.position[2]
        rotated_vel = self.velocity @ np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        rotated_vel = np.array([self.velocity[0] * np.cos(theta), self.velocity[0] * np.sin(theta), self.velocity[2]])
        self.position = self.position + rotated_vel * self.dt
        #self.position[:2] = np.clip(self.position[:2], *self.env_bounds)
        #self.position[2] = 0

        obstacle_distance, self.sensor_data = self._compute_sensor_measurment()

        if self.render_mode == "human":
            self._render_frame()

        terminated = np.hypot(*(self.position[:2] - self.target_position)) < 0.5 \
                     or np.min(obstacle_distance) < 0.1

        obs = self._get_obs()
        info = self._get_info()
        reward = self._get_reward()
        #print(reward)

        return obs, reward, terminated, False, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
