import pygame
import numpy as np


class SimpleEnvVisualizer():
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, env_bounds, render_mode="rgb_array", window_size=512):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.window_size = window_size
        self.env_bounds = env_bounds

    def render_frame(self, position,
                     target_position,
                     sensor_data,
                     obstacle_positions,
                     obstacle_radius,
                     path_start,
                     path_end):
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
            target_position * scale,
            10
        )

        # Draw sensor lines
        x, y, psi = position
        for i, (angle, distance) in enumerate(sensor_data):
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
            position[:2] * scale,
            10
        )

        # Draw "front"
        offset = 0.1
        front_offset = np.array([np.cos(position[2]) * offset, np.sin(position[2]) * offset])
        pygame.draw.circle(
            canvas,
            (100, 100, 0),
            (front_offset + position[:2]) * scale,
            8
        )

        # Draw obstacles
        for i, pos in enumerate(obstacle_positions):
            radius = obstacle_radius[i]
            pygame.draw.circle(
                canvas,
                (0, 100, 0),
                pos * scale,
                radius * scale
            )

        # Draw path line
        pygame.draw.line(
            canvas,
            (100, 0, 0),
            path_start * scale,
            path_end * scale,
            width=5
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

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
