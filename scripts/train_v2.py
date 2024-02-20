#!/opt/venv/bin/python
import rl4greencrab
from rl4greencrab import sb3_train_v2
from rl4greencrab import invasive_IPM_v2 as ipm_v2
from rl4greencrab import ts_env_v1

IPM_CFG = {
    'r': 0.5,
    'imm': 2000,
    'problem_scale': 2000,
    'action_reward_scale': 0.5, # cost per unit action in ipm
    'env_stoch': 0.1
}

OPTIONS = {
    "env_id" : "TimeSeries-v1",
    "n_envs" : 12,
    "config" : {"base_env_cls": ipm_v2, "base_env_cfg": IPM_CFG, "N_mem": 5},
    "algo" : "PPO",
    "id" : "NMem_5",
    "tensorboard" : "/home/rstudio/logs",
    "use_sde" : True,
    "total_timesteps" : 6000000,
}

sb3_train_v2(OPTIONS)
