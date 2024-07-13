#!/usr/bin/env python
# coding: utf-8
# %%

# # Part 2: other environments and RL training
# ---
# 
# In this notebook we will go over some of the variations of `greenCrabEnv` available in this package, and over the syntax for training RL algorithms on instances of these environments.

# ## 0. Setup
# ---
# As with Part 1 of this series, uncomment the following cell in order to install our package if you haven't done so already. After that restart the jupyter kernel.

# %%


# %pip install -e ..


# %%

# %%


import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_density, geom_line, geom_point, geom_violin, facet_grid, labs, theme, facet_wrap

from stable_baselines3 import PPO, TD3
from sb3_contrib import TQC
from stable_baselines3.common.env_util import make_vec_env

from rl4greencrab import greenCrabSimplifiedEnv, simulator
import gym

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances

from typing import Any, Dict

import optuna
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn
import gym
from rl_zoo3 import linear_schedule
from collections.abc import Callable
import logging
from pathlib import Path
import os
from sample_params import sample_ppo_params, sample_td3_params, sample_tqc_params

# ## 1. Other envs
# ---
# 
# We will go over two other envs provided by our package: `greenCrabSimplifiedEnv` and `timeSeriesEnv`.
# Let's focus on the first one of these envs.
# 
# ### greenCrabSimplifiedEnv
# 
# `greenCrabSimplifiedEnv` is closely related to `greenCrabEnv` and only varies in small aspects.
# Let's examine these aspecs one by one.
# The first aspect is its action space:

# %%

config = {
        'action_reward_scale': np.array([0.08, 0.08, 0.4]),
        'max_action': 3000,
        # 'env_stoch': 0.,
        'trapm_pmax': 10 * 0.1 * 2.75e-5, #2.26e-6,
        'trapf_pmax': 10 * 0.03 * 2.75e-5, #8.3e-7,
        'traps_pmax': 10 * 2.75e-5, #2.75e-5,
        'action_reward_exponent': 10,
    }

gcse = greenCrabSimplifiedEnv(config=config)
vec_env = make_vec_env(greenCrabSimplifiedEnv, n_envs=12)
eval_envs = vec_env


# ### Config

# %%


N_TRIALS = 100  # Maximum number of trials
N_JOBS = 1 # Number of jobs to run in parallel
N_STARTUP_TRIALS = 5  # Stop random sampling after N_STARTUP_TRIALS
N_EVALUATIONS = 2  # Number of evaluations during the training
N_TIMESTEPS = 500000 # Training budget
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_ENVS = 12
N_EVAL_EPISODES = 10
TIMEOUT = int(100 * 100)  # 3 hrs in seconds

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": gcse,
    "tensorboard_log": "/home/rstudio/logs"
}

study_result_path = 'rl4greencrab/notebooks/study_results'
save_tunning_model_name = "tunning_best_gcse_ppo_config_1"
file_name_to_write = 'study_results_ppo_congfig_1_cartpole.csv'


# ### Define Search Space in another sample_params.py


# ### define objective function

# %%

# Custom exception for NaN values
class NaNTrialException(optuna.exceptions.OptunaError):
    pass

from stable_baselines3.common.callbacks import EvalCallback

class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    
    :param eval_env: Evaluation environement
    :param trial: Optuna trial object
    :param n_eval_episodes: Number of evaluation episodes
    :param eval_freq:   Evaluate the agent every ``eval_freq`` call of the callback.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic policy.
    :param verbose:
    """

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate policy (done in the parent class)
            super()._on_step()
            self.eval_idx += 1
            # Send report to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


# %%


# objective all models
def objective_flex(model_parameter, model_train):
    """
    Objective function using by Optuna to evaluate
    one configuration (i.e., one set of hyperparameters).

    Given a trial object, it will sample hyperparameters,
    evaluate it and report the result (mean episodic reward after training)

    :param trial: Optuna trial object
    :return: Mean episodic reward after training
    """
    def objective_final(trial: optuna.Trial)->float:
        kwargs = DEFAULT_HYPERPARAMS.copy() 
        ### YOUR CODE HERE
        # TODO: 
        # 1. Sample hyperparameters and update the default keyword arguments: `kwargs.update(other_params)`
        # 2. Create the evaluation envs
        # 3. Create the `TrialEvalCallback`
    
        # 1. Sample hyperparameters and update the keyword arguments
        kwargs.update(model_parameter(trial,N_EVAL_ENVS,1,None))
        # Create the RL model
        model = model_train(**kwargs)
    
        # 2. Create envs used for evaluation using `make_vec_env`, `ENV_ID` and `N_EVAL_ENVS`
        eval_env = greenCrabSimplifiedEnv()
        # 3. Create the `TrialEvalCallback` callback defined above that will periodically evaluate
        # and report the performance using `N_EVAL_EPISODES` every `EVAL_FREQ`
        # TrialEvalCallback signature:
        # TrialEvalCallback(eval_env, trial, n_eval_episodes, eval_freq, deterministic, verbose)
        eval_callback = TrialEvalCallback(eval_env, trial, N_EVAL_EPISODES, EVAL_FREQ, deterministic=True)
    
        nan_encountered = False
        try:
            # Train the model
            model.learn(N_TIMESTEPS, callback=eval_callback)
        except (AssertionError,ValueError) as e:
            # Sometimes, random hyperparams can generate NaN
            print(e)
            nan_encountered = True
            
        finally:
            # Check if the current trial has the best accuracy
            if trial.number >= 1:
                if eval_callback.last_mean_reward > study.best_trial.value:
                    print("save the current best model")
                    model.save(os.path.join(study_result_path,save_tunning_model_name)) # save the trained best model
            # Free memory
            model.env.close()
            vec_env.close()
    
        # Tell the optimizer that the trial failed
        if nan_encountered:
            print("fail with value None")
            raise optuna.exceptions.TrialPruned() # try to skip the trail if the trail encounter NaN
    
        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()
    
        return eval_callback.last_mean_reward
    return objective_final


# %%
objective_PPO = objective_flex(sample_ppo_params, PPO)
objective_TD3 = objective_flex(sample_td3_params, TD3)
objective_TQC = objective_flex(sample_tqc_params, TQC)


# %%


import torch as th

# Set pytorch num threads to 1 for faster training
th.set_num_threads(1)
# Select the sampler, can be random, TPESampler, CMAES, ...
sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
# Do not prune before 1/3 of the max budget is used
pruner = MedianPruner(
    n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
)
# set logging to a log file ,Configure the logging
# logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(message)s')

logger = logging.getLogger()

logger.setLevel(logging.INFO)  # Setup the root logger.
logger.addHandler(logging.FileHandler("foo.log", mode="w"))

# Create the study and start the hyperparameter optimization
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

try:
    print("start the training")
    study.optimize(objective_PPO, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
except KeyboardInterrupt:
    pass

print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")

print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

print("  User attrs:")
for key, value in trial.user_attrs.items():
    print(f"    {key}: {value}")

result_file_path = os.path.join(study_result_path,file_name_to_write)

if not os.path.isfile(result_file_path):
    study.trials_dataframe().to_csv(result_file_path)
else:
    hyperparameter_df = pd.read_csv(result_file_path)# read current csv file
    # insert new hyperparameter into existed df
    pd.concat([hyperparameter_df, study.trials_dataframe()],ignore_index=True).to_csv(result_file_path)

# Write report
# study.trials_dataframe().to_csv("study_results_ppo_cartpole.csv")


fig1 = plot_optimization_history(study)
fig2 = plot_param_importances(study)

fig1.show()
fig2.show()


# ### PPO
