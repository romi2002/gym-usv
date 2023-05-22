import os
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from gymnasium.wrappers.record_video import capped_cubic_video_schedule
import gym_usv
from torch import nn
from wandb_callback import WandbCallback
from config import config_sac, config_ppo
import functools

class VideoCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(VideoCallback, self).__init__(verbose)

        self.video_filenames = set()

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        pass

    def _on_rollout_end(self) -> None:
        for filename in os.listdir(f"videos/{run.id}"):
            if filename not in self.video_filenames and filename.endswith(".mp4"):
                wandb.log({f"videos": wandb.Video(f"videos/{run.id}/{filename}")})
                self.video_filenames.add(filename)

    def _on_training_end(self) -> None:
        pass

def make_env():
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.FrameStack(env, 5)
    env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
    env = gym.wrappers.RecordVideo(env, f"videos/{run.id}")
    return env

def video_trigger(step):
    step /= 200
    if step < 1000:
        return int(round(step ** (1.0 / 3))) ** 3 == step
    else:
        return step % 1000 == 0

if __name__ == "__main__":
    env_id = "usv-simple"
    total_timesteps = 10e6

    config = config_sac

    run = wandb.init(
        project="usv-asmc-simple-no-obstacles",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )

    env = make_vec_env(env_id, n_envs=4, env_kwargs={"render_mode": "rgb_array"})
    #env = make_vec_env(env_id, n_envs=64, env_kwargs={"render_mode": "rgb_array"}, vec_env_cls=SubprocVecEnv)
    env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=video_trigger)
    env = VecFrameStack(env, n_stack=5)

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}", **config)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[WandbCallback(
            gradient_save_freq=5000,
            model_save_freq=100000,
            model_save_path=f"models/{run.id}",
            verbose=1,
        ), VideoCallback()],
    )
    run.finish()