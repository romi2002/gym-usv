import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class UsvSimpleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
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
        self.observation_space = spaces.Box(-1, 1, shape=(5,), dtype=np.float64)

        # dU, dR
        self.action_space = spaces.Box(np.array([0.5, -1]), np.array([1, 1]), shape=(2,), dtype=np.float64)
        # u, v, r
        self.max_action = np.array([5, 0, 5])
        self.max_acceleration = np.array([0.75, 0, 10])
        self.dt = (1 / 50)

        # Current state
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)

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
        angle = np.arctan2(*(self.target_position - self.position[:2])) - self.position[2]
        return np.array([self._wrap_angle(angle), distance])

    def _get_obs(self):
        target_state = self._get_target_state() / [np.pi, np.hypot(self.env_bounds[1], self.env_bounds[1])]
        return np.hstack((self.velocity, target_state))
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

        # Draw the agent
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            self.position[:2] * scale,
            10
        )

        # Draw "front"
        offset = 0.1
        front_offset = np.array([np.cos(self.position[2]) * offset, -np.sin(self.position[2]) * offset])
        pygame.draw.circle(
            canvas,
            (100, 100, 0),
            (front_offset + self.position[:2]) * scale,
            8
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
        return -target_info[1]
        return -np.abs(target_info[0]) - target_info[1]  # Use distance to target

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random start position and initial velocity
        self.position = self.np_random.uniform(*self.env_bounds, size=3)
        self.position[2] = self.np_random.uniform(-np.pi, np.pi)
        self.target_position = self.np_random.uniform(*self.env_bounds, size=2)
        self.velocity = self.np_random.uniform(-0.5, 0.5, size=3)
        # Set v velocity to 0
        self.velocity[1] = 0

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def step(self, action):
        action *= np.array([1, 1])
        action = np.insert(arr=action, obj=[1], values=0) # Insert v 0
        action = self.max_action * action

        # Update velocity based off current velocity and velocity action
        delta_velocity = np.clip(action - self.velocity, -self.max_acceleration, self.max_acceleration)
        self.velocity = np.clip(self.velocity + delta_velocity, -self.max_action, self.max_action)
        theta = self.position[2]
        rotated_vel = self.velocity @ np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        self.position = self.position + rotated_vel * self.dt
        self.position[:2] = np.clip(self.position[:2], *self.env_bounds)

        if self.render_mode == "human":
            self._render_frame()

        terminated = np.hypot(*(self.position[:2] - self.target_position)) < 0.5
        obs = self._get_obs()
        info = self._get_info()
        reward = self._get_reward()
        #print(reward)

        return obs, reward, terminated, False, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
