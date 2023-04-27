import gymnasium
import gym_usv
import numpy as np
import argparse
import time
import faulthandler
import pygame
def experiment_1_options():
    options = {}
    options['obs_x'] = np.array([-6,0,6,3,-3])
    options['obs_y'] = np.array([0,0,0,4,4])
    options['obs_r'] = np.array([1.5,1.5,1.5,1.5,1.5])
    options['start_position'] = np.array([0,-8,0])
    options['target_point'] = np.array([0, 8, 0])
    options['renderplots'] = False
    return options
def experiment_2_options():
    options = {'obs_x': np.array([]), 'obs_y': np.array([]), 'obs_r': np.array([])}

    def draw_vert_wall(options, start_x, end_x, y, radius=1):
        x = np.arange(start_x, end_x, radius * 2)
        y = np.full(len(x), y)
        r = np.full(len(x), radius)

        options['obs_x'] = np.concatenate((options['obs_x'], x))
        options['obs_y'] = np.concatenate((options['obs_y'], y))
        options['obs_r'] = np.concatenate((options['obs_r'], r))

    draw_vert_wall(options, -10, 30, -4, 0.5)
    draw_vert_wall(options, -10, 30, 1, 0.5)

    indexes_to_remove = [8, 9, 60, 61]
    options['obs_x'] = np.delete(options['obs_x'], indexes_to_remove)
    options['obs_y'] = np.delete(options['obs_y'], indexes_to_remove)
    options['obs_r'] = np.delete(options['obs_r'], indexes_to_remove)

    options['obs_x'] = np.append(options['obs_x'], [-10, 7])
    options['obs_y'] = np.append(options['obs_y'], [-3, -8])
    options['obs_r'] = np.append(options['obs_r'], [5, 5])

    options['start_position'] = np.array([0, -8, np.pi / 2])
    options['target_point'] = np.array([0, 8, 0])
    options['renderplots'] = False
    return options

if __name__ == '__main__':
    faulthandler.enable()
    parser = argparse.ArgumentParser(description='Test usv-asmc-ca env')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--no-render', dest='render', action='store_false')
    parser.set_defaults(render=True)
    parser.add_argument('--steps', type=int, nargs='?', default=5000)
    args = parser.parse_args()

    env = gymnasium.make('usv-simple', render_mode='human', max_episode_steps=5000)
    env.reset(options=experiment_1_options())
    start = time.perf_counter()
    action = np.array([0, -1.0])
    r = 0
    for i in range(args.steps):
        # if r == 50:
        #     print('reset')
        #     env.reset()
        #     r = 0
        # r += 1
        _, _, done, truncated, info = env.step(action)
        if(args.render):
            env.render()

        if done or truncated:
            break

        if args.render:
            keys = pygame.key.get_pressed()
            vel = 1
            if keys[pygame.K_LEFT]:
                action[1] = -1
            elif keys[pygame.K_RIGHT]:
                action[1] = 1
            else:
                action[1] = 0

            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 0

        action = np.clip(action, -1, 1)
        time.sleep(0.025)

    print(f"Completed Execution in {time.perf_counter() - start} seconds")