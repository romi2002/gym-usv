from torch import nn

config_ppo = {
    "use_sde": True,
    #"learning_rate": 0.001,
    "sde_sample_freq": 4,
    "n_steps": 2048,
    "batch_size": 64,
    # "gamma": 0.999,
    "policy_kwargs": dict(log_std_init=-2,
           ortho_init=False,
           activation_fn=nn.ReLU,
           net_arch=dict(pi=[256, 256], vf=[256, 256])
           )
}

config_sac = {
    "use_sde": True,
    "sde_sample_freq": 4,
    #"learning_rate": 0.001,
    "buffer_size": 400000,
    "batch_size": 256,
    "ent_coef": 'auto',
    "train_freq": 8,
    "gradient_steps": 8,
    "learning_starts": 50000,
    "learning_rate": 0.0001,
    "gamma": 0.99,
    # "gamma": 0.999,
    "policy_kwargs": dict(
        log_std_init=-3,
        net_arch=[400, 300]
    ),
    "lambda_t": 10,
    "lambda_s": 5,
    "eps_s": 0.1
}