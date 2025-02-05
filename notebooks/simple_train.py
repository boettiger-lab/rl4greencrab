import numpy as np
import pandas as pd
from stable_baselines3 import PPO, TD3
from sb3_contrib import TQC, RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from rl4greencrab.envs.green_crab_monthly_env import greenCrabMonthEnv
from rl4greencrab.envs.green_crab_monthly_env_norm import greenCrabMonthEnvNormalized
import gymnasium as gym
import logging

print('start training')

config = {}

gcme = greenCrabMonthEnv()
gmonthNorm = greenCrabMonthEnvNormalized()
vec_env = make_vec_env(greenCrabMonthEnvNormalized, n_envs=12)

def model_train(model_name):
    model = PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log="/home/rstudio/logs") #defualt PPO
    if model_name == 'PPO':
        model = PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log="/home/rstudio/logs")
    elif model_name == 'TQC':
        model = TQC("MlpPolicy", vec_env, verbose=0, tensorboard_log="/home/rstudio/logs")
    elif model_name == 'TD3':
        model = TD3("MlpPolicy", vec_env, verbose=0, tensorboard_log="/home/rstudio/logs")
    elif model_name == 'RecurrentPPO':
        model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=0, tensorboard_log="/home/rstudio/logs")

    print(f'start train {model_name}')
    model.learn(
            total_timesteps= 2000000, 
            progress_bar=False,
        )
    model_path = model_name + '_gcmenorm'
    model.save(model_path)
    
model_train('PPO')
model_train('TQC')
model_train('TD3')
model_train('RecurrentPPO')

print("finish training")