# Confirm environment is correctly defined:
from stable_baselines3.common.env_checker import check_env
from rl4greencrab import (
    twoActEnv,
    TwoActNormalized
)

config = {
    "w_mort_scale" : 600,
    "growth_k": 0.70,
    'random_start':True,
    'var_penalty_const': 0,
    'observation_type': 'count-biomass-time'
}

def test_GC():
    check_env(twoActEnv(config), warn=True)
    check_env(TwoActNormalized(config), warn=True)
    