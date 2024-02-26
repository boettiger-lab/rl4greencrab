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
    try:
        full_path = save_id + ".zip"
        deploy_model(full_path, "sb3/"+path, repo=options["repo"])
        deploy_model(config_file, "sb3/"+config_file, repo=options["repo"])
    except:
        print("Could not deploy model to hugging face :(.")

def deploy_model(path, path_in_repo, repo, clean=False):
    api = HfApi()
    if path_in_repo is None:
        path_in_repo = basename(path)
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path_in_repo,
        repo_id=repo,
        repo_type="model")
    if clean:
        pathlib.Path(path).unlink()

def sb3_train_v2(options = dict):
    vec_env = make_vec_env(
        options["env_id"], options["n_envs"], env_kwargs={"config": options["config"]}
    )
    ALGO = algorithm(options["algo"])
    model_id = options["algo"] + "-" + options["env_id"]  + "-" + options["id"]

    model = ALGO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        tensorboard_log=options["tensorboard"],
        use_sde=options["use_sde"],
    )
    progress_bar = options.get("progress_bar", False)
    model.learn(total_timesteps=options["total_timesteps"], tb_log_name=model_id, progress_bar=progress_bar)

    model.save(model_id)
    path = model_id + ".zip"
    # deploy_model(path, "sb3/"+path, repo=options["repo"])
    # deploy_model(config_file, "sb3/"+config_file, repo=options["repo"])

def sb3_train_metaenv(config_file):
    with open(config_file, "r") as stream:
        options = yaml.safe_load(stream)
    
    vec_env = make_vec_env(
        options["meta_env_id"], 
        options["n_envs"], 
        env_kwargs={
            "config": 
            {
                "base_env": gym.make(options["base_env_id"], config=options["base_env_cfg"]), 
                **options["meta_env_kwargs"],
            }
        }
    )
    ALGO = algorithm(options["algo"])
    model_id = "{}-Meta_{}-Base_{}-{}".format(
        options["algo"], 
        options["meta_env_id"], 
        options["base_env_id"], 
        options["id"],
    )
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

    model.save(save_id)
    # path = model_id + ".zip"
    # deploy_model(path, "sb3/"+path, repo=options["repo"])
    # deploy_model(config_file, "sb3/"+config_file, repo=options["repo"])