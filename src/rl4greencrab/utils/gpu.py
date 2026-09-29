import os

import pandas as pd
import yaml

from rl4greencrab.agents.gpu_ppo import GPUPPO
from rl4greencrab.envs.gpu_env import TwoActGPU


def gpu_train(config_file, **kwargs):
    """
    Train GPUPPO from the same yaml files used by `sb3_train`.

    Keys used: config, total_timesteps, id, save_path, tensorboard,
    model_config (PPO hyperparameters and policy_kwargs), plus optional
    n_envs (default 1024; the yaml's CPU value of 12 is ignored unless
    `gpu_n_envs` is set), seed and device. `algo`, `policy` and `env_id` are
    ignored: the algorithm is always PPO on TwoActGPU (normalized actions).
    """
    with open(config_file, "r") as stream:
        options = yaml.safe_load(stream)
    options = {**options, **kwargs}
    config = dict(options["config"])
    if "param_csv" in config:
        config["param_df"] = pd.read_csv(config["param_csv"])

    env = TwoActGPU(config, num_envs=options.get("gpu_n_envs", 1024),
                    device=options.get("device"), seed=options.get("seed"))
    model_config = dict(options.get("model_config", {}))
    model_config.pop("use_sde", None)  # SB3-only option
    model = GPUPPO(env, seed=options.get("seed"), tensorboard_log=options.get("tensorboard"), **model_config)

    model_id = "GPUPPO-(" + config["observation_type"] + ")-" + str(options["id"])
    model.learn(total_timesteps=options["total_timesteps"], tb_log_name=model_id,
                log_interval=options.get("log_interval", 10))
    path = model.save(os.path.join(options["save_path"], model_id))
    print(f"Saved GPUPPO model at {path}", flush=True)
    return model
