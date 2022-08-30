import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt


def generate_path(start_point, num_waypoints, angle_mean=0, angle_std=0.50, length_mean=3, length_std=0.1):
    path_data = np.vstack([np.clip(np.random.normal(angle_mean, angle_std, num_waypoints), -np.pi/2 + 0.1, np.pi/2 - 0.1),
                           np.random.normal(length_mean, length_std, num_waypoints)]).T

    waypoints = np.tile(path_data[:, 1], (2, 1)).T * np.vstack([np.cos(path_data[:, 0]), np.sin(path_data[:, 0])]).T
    waypoints[0] = start_point
    waypoints = np.cumsum(waypoints, axis=0)
    path = PchipInterpolator(waypoints[:, 0], waypoints[:, 1])
    return path, waypoints


def place_obstacles(path, waypoints, num_obs, obs_pos_std=8, obs_rad_mean=0.5, obs_rad_std=0.1, obs_min_size=0.01):
    path_dx = path.derivative()

    # x,y, radius
    min_x = np.min(waypoints[:, 0])
    max_x = np.max(waypoints[:, 0])

    # base x pos, displacement, deriv offset, deriv
    obstacle_data = np.vstack([
        np.random.uniform(min_x, max_x, num_obs),
        np.random.normal(0, obs_pos_std, num_obs),
        np.random.uniform(np.pi, np.pi * 2, num_obs),
        np.zeros((1, num_obs))]).T

    obstacle_data[:, 3] = path_dx(obstacle_data[:, 0])
    obs_angle = np.arctan2(obstacle_data[:, 3], obstacle_data[:, 0]) + obstacle_data[:, 2]
    obstacle_pos = np.array([obstacle_data[:, 0], path(obstacle_data[:, 0])]) + obstacle_data[:, 1] * np.array(
        [np.cos(obs_angle), np.sin(obs_angle)])
    obstacles = np.vstack([obstacle_pos, np.random.normal(obs_rad_mean, obs_rad_std, num_obs)]).T
    obstacles = obstacles[obstacles[:, 2] > obs_min_size]
    # obstacles = np.hstack([obstacle_pos, np.random.uniform(5,10,num_obs)]).T
    return obstacles


def plot_path(path, waypoints, obstacles):
    X = np.linspace(waypoints[0][0], waypoints[-1][0])
    Y = path(X)
    plt.plot(X, Y)
    plt.scatter(waypoints[:, 0], waypoints[:, 1])
    plt.scatter(obstacles[:, 0], obstacles[:, 1], s=obstacles[:, 2] * 10)
    plt.show()


def simplified_lookahead(path, waypoints, current_x, lookahead):
    # simple "lookahead", take current boat position, add lookahead distance to x
    # simpler than an actual lookahead implementation, speedy boi
    x = np.maximum(current_x + lookahead, waypoints[0][0]) # prevent going before x0
    return x, path(x)
