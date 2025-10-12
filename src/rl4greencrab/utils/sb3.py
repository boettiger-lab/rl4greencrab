import yaml
import os
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, HER, DDPG
from sb3_contrib import TQC, ARS, RecurrentPPO
from rl4greencrab import LipschitzPPO

def algorithm(algo):
    algos = {
        'PPO': PPO, 
        'ppo': PPO,
        'RecurrentPPO': RecurrentPPO,
        'RPPO': RecurrentPPO,
        'recurrentppo': RecurrentPPO,
        'rppo': RecurrentPPO,
        #
        'ARS': ARS,
        'ars': ARS,
        'A2C': A2C, 
        'a2c':A2C ,
        #
        'DDPG': DDPG, 
        'ddpg': DDPG,
        #
        'HER': HER, 
        'her': HER,
        #
        'SAC': SAC, 
        'sac': SAC,
        #
        'TD3': TD3, 
        'td3': TD3,
        #
        'TQC': TQC, 
        'tqc': TQC,
        #
        'LipschitzPPO': LipschitzPPO
    }
    return algos[algo]

# Map string names to actual functions/classes
activation_map = {
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
}

def sb3_train(config_file, **kwargs):
    with open(config_file, "r") as stream:
        options = yaml.safe_load(stream)
        options = {**options, **kwargs}
        # updates / expands on yaml options with optional user-provided input

    if "n_envs" in options:
        env = make_vec_env(
            options["env_id"], options["n_envs"], env_kwargs={"config": options["config"]}
        )
        print(f'env config: {options["config"]}')
    else:
        env = gym.make(options["env_id"])
    ALGO = algorithm(options["algo"])
    PLOICY = options.get("policy", "MlpPolicy")
    model_id = options["algo"] + "-" + options["env_id"]  + "-" + options["id"]
    model_config = options.get('model_config', {})
    policy_kwargs = model_config.get("policy_kwargs", {})
    
    # subprocess activation_fn parameter
    if "activation_fn" in policy_kwargs:
        act = model_config["policy_kwargs"]["activation_fn"]
        model_config["policy_kwargs"]["activation_fn"] = activation_map[act]

    save_id = os.path.join(options["save_path"], model_id)

    model = ALGO(
            PLOICY,
            env,
            verbose=0,
            tensorboard_log=options["tensorboard"],
            **model_config
        )
    
    os.makedirs(model.tensorboard_log, exist_ok=True)
    progress_bar = options.get("progress_bar", False)
    model.learn(total_timesteps=options["total_timesteps"], tb_log_name=model_id, progress_bar=progress_bar)

    os.makedirs(options["save_path"], exist_ok=True)
    model.save(save_id)
    print(f"Saved {options['algo']} model at {save_id}", flush=True)
    
    return model