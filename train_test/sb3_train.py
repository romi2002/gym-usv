import os
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
import gym_usv
from torch import nn

env_name = "usv-simple"
total_timesteps = 10e6
config_ppo = {
    "use_sde": True,
    #"learning_rate": 0.001,
    "sde_sample_freq": 4,
    "n_steps": 512,
    "batch_size": 64,
    # "gamma": 0.999,
    "policy_kwargs": dict(log_std_init=-2,
           ortho_init=True,
           activation_fn=nn.ReLU,
           net_arch=dict(pi=[512, 512], vf=[512, 512])
           )
}

config_sac = {
    "use_sde": True,
    #"learning_rate": 0.001,
    "sde_sample_freq": 4,
    "buffer_size": 1000000,
    "batch_size": 256,
    "ent_coef": 'auto',
    "gamma": 0.99,
    "tau": 0.01,
    "train_freq": 1,
    "gradient_steps": 1,
    "learning_starts": 10000,
    # "gamma": 0.999,
    "policy_kwargs": dict(
        log_std_init=-2,
        net_arch=[400, 300]
    )
}

config = config_sac

run = wandb.init(
    project="usv-asmc-simple",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    save_code=True,  # optional
)

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
    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
    env = gym.wrappers.RecordVideo(env, f"videos/{run.id}")
    return env

env = DummyVecEnv([make_env])
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}", **config)
model.learn(
    total_timesteps=total_timesteps,
    callback=[WandbCallback(
        gradient_save_freq=100,
        model_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ), VideoCallback()],
)
run.finish()