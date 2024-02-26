#!/opt/venv/bin/python
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

manual_kwargs = {}
if args.algo:
    manual_kwargs['algo'] = args.algo
if args.time_steps:
    manual_kwargs['total_timesteps'] = args.time_steps
if args.n_envs:
    manual_kwargs['n_envs'] = args.n_envs

import os
boilerplate_cfg = os.path.join("..", "hyperpars", "gcse-boilerplate.yml")


import rl4greencrab
from rl4greencrab.utils.sb3 import sb3_train 

sb3_train(boilerplate_cfg, **manual_kwargs)
