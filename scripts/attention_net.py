import ray
from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from rl4greencrab import greenCrabSimplifiedEnv, simulator
import numpy as np

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
    "timesteps_total": 1000000  # Set a limit of 100,000 timesteps for training
})