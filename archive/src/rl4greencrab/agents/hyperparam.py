from typing import Any

import numpy as np
import optuna
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn


def convert_onpolicy_params(sampled_params: dict[str, Any]) -> dict[str, Any]:
    hyperparams = sampled_params.copy()

    # TODO: account when using multiple envs
    # if batch_size > n_steps:
    #     batch_size = n_steps

    hyperparams["gamma"] = 1 - sampled_params["one_minus_gamma"]
    del hyperparams["one_minus_gamma"]

    hyperparams["gae_lambda"] = 1 - sampled_params["one_minus_gae_lambda"]
    del hyperparams["one_minus_gae_lambda"]

    net_arch = sampled_params["net_arch"]
    del hyperparams["net_arch"]

    for name in ["batch_size", "n_steps"]:
        if f"{name}_pow" in sampled_params:
            hyperparams[name] = 2 ** sampled_params[f"{name}_pow"]
            del hyperparams[f"{name}_pow"]

    net_arch = {
        "tiny": dict(pi=[64], vf=[64]),
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch]

    activation_fn_name = sampled_params["activation_fn"]
    del hyperparams["activation_fn"]

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation_fn_name]

    return {
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        },
        **hyperparams,
    }

def sample_ppo_params(trial: optuna.Trial, additional_args: dict) -> dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    # From 2**5=32 to 2**10=1024
    # batch_size_pow = trial.suggest_int("batch_size_pow", 2, 10)
    # From 2**5=32 to 2**12=4096
    n_steps_pow = trial.suggest_int("n_steps_pow", 5, 12)
    n_steps = 2 ** n_steps_pow
    n_envs = additional_args.get("n_envs", 12)  # Default to 1 if not provided
    total_rollout = n_steps * n_envs
    
    valid_batch_sizes = get_divisors(total_rollout)
    batch_size = trial.suggest_categorical("batch_size", valid_batch_sizes)
    
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    one_minus_gae_lambda = trial.suggest_float("one_minus_gae_lambda", 0.0001, 0.1, log=True)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])

    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2)
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    # lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    # if lr_schedule == "linear":
    #     learning_rate = linear_schedule(learning_rate)

    # Display true values
    trial.set_user_attr("gamma", 1 - one_minus_gamma)
    trial.set_user_attr("n_steps", 2**n_steps_pow)
    # trial.set_user_attr("batch_size", 2**batch_size_pow)
    sampled_params = {
        "n_steps_pow": n_steps_pow,
        # "batch_size_pow": batch_size_pow,
        "batch_size": batch_size,
        "one_minus_gamma": one_minus_gamma,
        "one_minus_gae_lambda": one_minus_gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "max_grad_norm": max_grad_norm,
        "net_arch": net_arch,
        "activation_fn": activation_fn,
    }

    return convert_onpolicy_params(sampled_params)

def get_divisors(n: int, min_val: int = 32, max_val: int = 1024) -> list[int]:
    """Return valid divisors within [min_val, max_val] for given n."""
    return [d for d in range(min_val, min(max_val + 1, n + 1)) if n % d == 0]