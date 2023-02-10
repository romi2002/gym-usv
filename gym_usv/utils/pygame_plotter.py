import pygame
from collections import deque
import numpy as np
from typing import Tuple

pygame.font.init()
font = pygame.font.SysFont('arial', 24)

def render_plot(data: deque,
                name: str,
                surface: pygame.surface,
                start_coords: Tuple[int, int],
                size: Tuple[int, int],
                y_lim: Tuple[float, float] = (-1.0, 1.0)):
    pygame.draw.rect(surface, (10,10,10), (start_coords, size))
    #text_img = font.render(name, True, (0, 255, 255))
    #surface.blit(text_img, start_coords)
    x_inc = size[0] / len(data)
    y_scale = size[1] / (y_lim[0] - y_lim[1])
    y_offset = start_coords[1] + size[1] / 2
    draw_pos = (start_coords[0], start_coords[1] + size[1])
    for i, _ in enumerate(data):
        if i == 0:
            continue

        d_0 = np.clip(data[i - 1], y_lim[0], y_lim[1]) * y_scale + y_offset
        d_1 = np.clip(data[i], y_lim[0], y_lim[1]) * y_scale + y_offset

        pygame.draw.line(surface,
                         (250, 0, 0),
                         (draw_pos[0], d_0),
                         (draw_pos[0] + x_inc, d_1), width=2)
        draw_pos = (draw_pos[0] + x_inc, draw_pos[1])