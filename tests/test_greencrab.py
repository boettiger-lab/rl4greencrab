# Confirm environment is correctly defined:
from stable_baselines3.common.env_checker import check_env
from rl4greencrab import greenCrabEnv, greenCrabSimplifiedEnv, timeSeriesEnv

def test_GC():
    check_env(greenCrabEnv(), warn=True)
    check_env(greenCrabSimplifiedEnv(), warn=True)
    check_env(timeSeriesEnv(), warn=True)
    