import numpy as np
import pandas as pd
from stable_baselines3 import PPO, TD3
from sb3_contrib import TQC, RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from rl4greencrab.envs.green_crab_monthly_env import greenCrabMonthEnv
from rl4greencrab.envs.green_crab_monthly_env_norm import greenCrabMonthEnvNormalized
import gymnasium as gym
import logging

print('start training',flush=True)

config = {
    "w_mort_scale" : 600,
    "growth_k": 0.70,
    "random_start": True,
    "curriculum": False
}

gcme = greenCrabMonthEnv(config)
gmonthNorm = greenCrabMonthEnvNormalized(config)
vec_env = make_vec_env(greenCrabMonthEnvNormalized, n_envs=12, env_kwargs={'config':config})

model_config = {
    'policy':"MultiInputLstmPolicy",
    'env':vec_env,
    'verbose':0,
    'normalize_advantage': True,
    'batch_size': 256,
    'n_steps': 1024,
    'gamma': 0.9999,
    'learning_rate': 0.0003,
    'ent_coef': 0.00429,
    'clip_range': 0.1,
    'n_epochs': 10,
    'gae_lambda': 0.9,
    'max_grad_norm': 0.5,
    'vf_coef': 0.19,
    'use_sde': False,
    'sde_sample_freq': 8,
    'tensorboard_log':"/home/rstudio/logs",
    'policy_kwargs': dict(log_std_init=0.0, ortho_init=False,
                       lstm_hidden_size=128,
                       n_lstm_layers = 2,
                       enable_critic_lstm=True,
                       net_arch=[64, 32, 16])
}

def model_train(model_name):
    model = PPO('MultiInputPolicy', vec_env, verbose=0, tensorboard_log="/home/rstudio/logs") #defualt PPO
    if model_name == 'PPO':
        model = PPO('MultiInputPolicy', vec_env, verbose=0, tensorboard_log="/home/rstudio/logs")
    elif model_name == 'TQC':
        model = TQC('MultiInputPolicy', vec_env, verbose=0, tensorboard_log="/home/rstudio/logs")
    elif model_name == 'TD3':
        model = TD3('MultiInputPolicy', vec_env, verbose=0, tensorboard_log="/home/rstudio/logs")
    elif model_name == 'RecurrentPPO':
        model = RecurrentPPO(**model_config)

    print(f'start train {model_name}', flush=True)
    
    lstm_hidden_size = model_config['policy_kwargs']['lstm_hidden_size']
    n_lstm_layers = model_config['policy_kwargs']['n_lstm_layers']
    net_arch = model_config['policy_kwargs']['net_arch']
    use_sde = model_config['use_sde']
    model_path = f'{model_name}_gcmenorm_{lstm_hidden_size}_{n_lstm_layers}_{net_arch}_{use_sde}'
    print(model_path, flush=True)    
    
    model.learn(
            total_timesteps= 10000000,
            progress_bar=False,
        )
    model.save(model_path)
    
# model_train('PPO')
# model_train('TQC')
# model_train('TD3')
model_train('RecurrentPPO')

print("finish training")