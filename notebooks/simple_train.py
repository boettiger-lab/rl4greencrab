import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_density, geom_line, geom_point, geom_violin, facet_grid, labs, theme, facet_wrap

from stable_baselines3 import PPO, TD3
from sb3_contrib import TQC
from stable_baselines3.common.env_util import make_vec_env

from rl4greencrab import greenCrabSimplifiedEnv,  simulator
import gym

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
import gymnasium as gym
import logging
import numpy as np

from gymnasium import spaces
from scipy.stats import norm

import sample_params

config = {
        'action_reward_scale': np.array([0.08, 0.08, 0.4]),
        'max_action': 3000,
        # 'env_stoch': 0.,
        'trapm_pmax': 10 * 0.1 * 2.75e-5, #2.26e-6,
        'trapf_pmax': 10 * 0.03 * 2.75e-5, #8.3e-7,
        'traps_pmax': 10 * 2.75e-5, #2.75e-5,

        'loss_a': 100,
        'loss_b': 10,
        'loss_c': 0.5,
        
        'action_reward_exponent': 10,
    }

gcse = greenCrabSimplifiedEnv(config)
vec_env = make_vec_env(greenCrabSimplifiedEnv, n_envs=12)
eval_envs = vec_env

model = TQC("MlpPolicy", 
            vec_env, 
            verbose=0, 
            gamma= 0.9999, 
            learning_rate = 0.020439420278073966, 
            batch_size = 16, 
            buffer_size = 10000, 
            learning_starts= 0, 
            train_freq = 16, 
            tau = 0.05, 
            top_quantiles_to_drop_per_net = 1)
model.learn(
	total_timesteps= 1000000, 
	progress_bar=False,
)
print("finish training")
model.save("tqc_gcse_short")
