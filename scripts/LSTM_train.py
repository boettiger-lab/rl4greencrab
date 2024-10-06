import numpy as np
import ray

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor

from rl4fisheries import AsmEnv, Msy, ConstEsc, CautionaryRule
from rl4fisheries.envs.asm_fns import get_r_devs, observe_total
from rl4fisheries import AsmCRLike

# CONFIG = {"s":  0.86, "noiseless": False, "testing_harvs": False}
CONFIG = {
    'observation_fn_id': 'observe_1o',
    'n_observs': 1,
    'harvest_fn_name': "trophy"
    # 'upow': 0.6,
    # 'use_custom_harv_vul': True,
    # 'use_custom_surv_vul': True,
}

pol_env = AsmEnv(config=CONFIG)
evalEnv = pol_env

print("Start training for LSTM_PPO")

model = RecurrentPPO("MlpLstmPolicy", pol_env, verbose=1, tensorboard_log="/home/rstudio/logs")
model.learn(10000000, progress_bar=False)
model.save("../notebooks/recurrent_ppo_fisheries.zip")

print("Finish Training")