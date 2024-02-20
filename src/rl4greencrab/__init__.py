from rl4greencrab.invasive_ipm import invasive_IPM, invasive_IPM_v2
from rl4greencrab.ts_model import ts_env_v1
from rl4greencrab.util import sb3_train, sb3_train_v2

from gymnasium.envs.registration import register
register(id="GreenCrab-v1", entry_point="rl4greencrab.invasive_ipm:invasive_IPM")
register(id="GreenCrab-v2", entry_point="rl4greencrab.invasive_ipm:invasive_IPM_v2")
register(id="TimeSeries-v1", entry_point="rl4greencrab.ts_model:ts_env_v1")