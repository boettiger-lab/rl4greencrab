# Confirm environment is correctly defined:
from stable_baselines3.common.env_checker import check_env
from rl4greencrab import (
    twoActEnv,
    TwoActNormalized
)
import pandas as pd
import os

param_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'posterior', 'params.csv'))

config = {
    'random_start':True,
    'observation_type': 'count-biomass-time',
    'param_df': param_df
}

def test_GC():
    check_env(twoActEnv(config), warn=True)
    check_env(TwoActNormalized(config), warn=True)
    