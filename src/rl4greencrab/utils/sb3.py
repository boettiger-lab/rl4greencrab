import yaml
import os

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
from sb3_contrib import TQC, ARS

def algorithm(algo):
    algos = {
        "PPO": PPO,
        "ARS": ARS,
        "TQC": TQC,
        "A2C": A2C,
        "SAC": SAC,
        "DQN": DQN,
        "TD3": TD3,
        "ppo": PPO,
        "ars": ARS,
        "tqc": TQC,
        "a2c": A2C,
        "sac": SAC,
        "dqn": DQN,
        "td3": TD3,
    }
    return algos[algo]

def sb3_train(config_file):
    with open(config_file, "r") as stream:
        options = yaml.safe_load(stream)

    vec_env = make_vec_env(
        options["env_id"], options["n_envs"], env_kwargs={"config": options["config"]}
    )
    ALGO = algorithm(options["algo"])
    model_id = options["algo"] + "-" + options["env_id"]  + "-" + options["id"]
    save_id = os.path.join(options["save_path"], model_id)

    model = ALGO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        tensorboard_log=options["tensorboard"],
        use_sde=options["use_sde"],
    )

    progress_bar = options.get("progress_bar", False)
    model.learn(total_timesteps=options["total_timesteps"], tb_log_name=model_id, progress_bar=progress_bar)

    os.makedirs(options["save_path"], exist_ok=True)
    model.save(save_id)
    print(f"Saved {options['algo']} model at {save_id}")
    
    return model