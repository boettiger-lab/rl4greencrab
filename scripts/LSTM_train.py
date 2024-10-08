import numpy as np
import ray
from ray import tune
from ray.rllib.models.catalog import ModelCatalog

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from rl4greencrab import greenCrabSimplifiedEnv, simulator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algo", help="Algo to train", type=str, 
                    choices=[
                        'PPO', 'RecurrentPPO', 'ppo', 'recurrentppo', 'RPPO', 'rppo',
                        'ARS', 'A2C', 'ars', 'a2c',
                        'DDPG', 'ddpg',
                        'HER', 'her',
                        'SAC', 'sac'
                        'TD3', 'td3',
                        'TQC', 'tqc',
                    ]
)
parser.add_argument("-t", "--time-steps", help="N. timesteps to train for", type=int)
parser.add_argument(
    "-ne", 
    "--n-envs", 
    help="Number of envs to use simultaneously for faster training. " 
        "Check algos for compatibility with this arg.", 
    type=int,
)

args = parser.parse_args()

print("start training greencrab")

config = {
        'action_reward_scale': np.array([0.08, 0.08, 0.4]),
        'max_action': 3000,
        # 'env_stoch': 0.,
        'trapm_pmax': 10 * 0.1 * 2.75e-5, #2.26e-6,
        'trapf_pmax': 10 * 0.03 * 2.75e-5, #8.3e-7,
        'traps_pmax': 10 * 2.75e-5, #2.75e-5,

        'loss_a': 0.2,
        'loss_b': 5,
        'loss_c': 5,
        
        'action_reward_exponent': 10,
    }

gcse = greenCrabSimplifiedEnv(config)
evalEnv = gcse

def train_lstm(timesteps):
    print("Start training for LSTM_PPO")
    
    model = RecurrentPPO("MlpLstmPolicy", gcse, verbose=1, tensorboard_log="/home/rstudio/logs")
    model.learn(timesteps, progress_bar=False)
    model.save("../notebooks/recurrent_ppo_greencrab_long.zip")
    
    print("Finish Training")

def train_attention_net(timesteps):
    ray.init()
    tune.register_env("gcse", lambda c: greenCrabSimplifiedEnv(config))
    
    tune.run("PPO", config={
        "env": "gcse",
        "model": {
            "use_attention": True
        },
        "framework": "torch",
    },
    storage_path = "/home/rstudio/logs",
    stop={
        "timesteps_total": timesteps  # Set a limit of timesteps for training
    })
    