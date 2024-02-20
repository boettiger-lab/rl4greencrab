# Confirm environment is correctly defined:
from stable_baselines3.common.env_checker import check_env
from rl4greencrab import invasive_IPM_v2

def test_GC():
    check_env(invasive_IPM_v2(), warn=True)

