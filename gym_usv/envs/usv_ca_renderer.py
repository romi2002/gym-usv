import numpy as np
import pygame


class UsvCaRenderer():
    def __init__(self):
        screen_width = 400
        screen_height = 800
        self.screen_dim = (screen_width, screen_height)

        world_width = 20
        self.scale = screen_width / world_width

        # Map limits in meters
        self.max_y = 10
        self.min_y = -10
        self.max_x = 30
        self.min_x = -10

        self.screen = None
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.font = pygame.font.SysFont('arial', 24)

    def _draw_sensors(self, surf, position, sensors):
        x, y, psi = position

        for i, s in enumerate(sensors):
            angle = s[0] + psi
            # angle = np.where(np.greater(np.abs(angle), np.pi), np.sign(angle) * (np.abs(angle) - 2 * np.pi), angle)
            initial = ((y - self.min_y) * self.scale, (x - self.min_x) * self.scale)
            x_f = s[1] * np.math.cos(angle) + x - self.min_x
            y_f = s[1] * np.math.sin(angle) + y - self.min_y
            final = (y_f * self.scale, x_f * self.scale)

            color = (0, 255, 0)

            pygame.draw.line(surf, color, initial, final)

    def _draw_sectors(self, surf, position, sensors, sectors, sector_size):
        x, y, psi = position

        for i, s in enumerate(sensors):
            angle = s[0] + psi
            # angle = np.where(np.greater(np.abs(angle), np.pi), np.sign(angle) * (np.abs(angle) - 2 * np.pi), angle)
            initial = ((y - self.min_y) * self.scale, (x - self.min_x) * self.scale)
            m = np.math.tan(angle)
            x_f = s[1] * np.math.cos(angle) + x - self.min_x
            y_f = s[1] * np.math.sin(angle) + y - self.min_y
            final = (y_f * self.scale, x_f * self.scale)
            section = np.floor(i / sector_size).astype(int)

            color = (0, 255, 0)

            if sectors[section] < 1:
                color = (255, 0, 0)
                if (i % 10 == 0):
                    color = (255, 0, 255)
            elif (i % 10 == 0):
                color = (0, 0, 255)

            pygame.draw.line(surf, color, initial, final)

    @staticmethod
    def _transform_points(points, x, y, angle):
        if angle is not None:
            s, c = (np.sin(angle), np.cos(angle))
            points = [(px * c - py * s, px * s + py * c) for (px, py) in points]
        points = [(px + x, py + y) for (px, py) in points]
        return points

    def _draw_boat(self, surf, position):
        boat_width = 15
        boat_height = 20

        l, r, t, b, c, m = -boat_width / 2, boat_width / 2, boat_height, 0, 0, boat_height / 2
        boat_points = [(l, b), (l, m), (c, t), (r, m), (r, b)]
        boat_points = [(x, y - boat_height / 2) for (x, y) in boat_points]
        boat_points = self._transform_points(boat_points, (position[1] - self.min_y) * self.scale,
                                             (position[0] - self.min_x) * self.scale, -position[2])

        pygame.draw.polygon(surf, (3, 94, 252), boat_points)

    def _draw_obstacles(self, surf, obstacle_x, obstacle_y, obstacle_radius):
        for i in range(len(obstacle_x)):
            obs_points = [((obstacle_y[i][0] - self.min_y) * self.scale, (obstacle_x[i][0] - self.min_x) * self.scale)]
            pygame.draw.circle(surf, (0, 0, 255), obs_points[0], obstacle_radius[i][0] * self.scale)

    def _draw_highlighted_sectors(self, surf, sensor_max_range, sensor_span, sector_num, position):
        x, y, psi = position

        angle = -(2 / 3) * np.pi + psi
        angle = np.where(np.greater(np.abs(angle), np.pi), np.sign(angle) * (np.abs(angle) - 2 * np.pi), angle)
        for i in range(self.sector_num + 1):
            initial = ((y - self.min_y) * self.scale, (x - self.min_x) * self.scale)
            x_f = sensor_max_range * np.math.cos(angle) + x - self.min_x
            y_f = sensor_max_range * np.math.sin(angle) + y - self.min_y
            final = (y_f * self.scale, x_f * self.scale)
            pygame.draw.line(surf, (0, 0, 0), initial, final, width=2)
            angle = angle + sensor_span / sector_num
            angle = np.where(np.greater(np.abs(angle), np.pi), np.sign(angle) * (np.abs(angle) - 2 * np.pi), angle)

    def _draw_path(self, surf, waypoints, path):
        # Draw path
        path_x = np.linspace(waypoints[0][0], waypoints[-1][0])
        path_y = path(path_x)
        pygame.draw.lines(surf, (0, 255, 0), False,
                          [tuple(p) for p in
                           np.vstack([(path_y - self.min_y) * self.scale, (path_x - self.min_x) * self.scale]).T],
                          width=2)

    @staticmethod
    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def render(self,
               position,
               sensors,
               target_point,
               obstacle_x,
               obstacle_y,
               obstacle_radius,
               debug_vars,
               show_debug_vars,
               mode='human',
               render_fps=60
               ):

        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(self.screen_dim)

        surf = pygame.Surface(self.screen_dim)
        surf.fill((255, 255, 255))

        self._draw_obstacles(surf, obstacle_x, obstacle_y, obstacle_radius)
        self._draw_sensors(surf, position, sensors)
        self._draw_boat(surf, position)

        # Draw target point
        x_t, y_t = target_point
        pygame.draw.circle(surf, (0, 255, 0), ((y_t - self.min_y) * self.scale, (x_t - self.min_x) * self.scale),
                           radius=10)

        # TODO Draw safety radius
        # safety_radius = (self.boat_radius + self.safety_radius) * scale
        # safety = ((y - self.min_y) * scale, (x - self.min_x) * scale)
        # pygame.draw.circle(self.surf, (255, 0, 0), safety, safety_radius, width=3)

        surf = pygame.transform.flip(surf, False, True)

        text_start_pos = (20, 20)
        if show_debug_vars:
            for key, var in debug_vars.items():
                if isinstance(var, str):
                    text_img = self.font.render(f"{key}: {var}", True, (0, 0, 0))
                else:
                    text_img = self.font.render(f"{key}: {round(var, 4)}", True, (0, 0, 0))
                surf.blit(text_img, text_start_pos)
                text_start_pos = text_start_pos[0], text_start_pos[1] + 30

        if mode == "human":
            self.screen.blit(surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(render_fps)
            pygame.display.flip()
        if mode == "rgb_array":
            return self._create_image_array(surf, surf, self.screen_dim)
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return False

    def close(self):
        pygame.display.quit()
        self.isopen = False
        pygame.quit()
