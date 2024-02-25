from rl4greencrab.envs.green_crab_ipm import greenCrabEnv, greenCrabSimplifiedEnv
from rl4greencrab.envs.time_series import timeSeriesEnv
from rl4greengrab.agents.const_action import constAction
from rl4greengrab.agents.const_escapement import constEsc
# from envs.util import sb3_train, sb3_train_v2, sb3_train_metaenv

from gymnasium.envs.registration import register
register(
    id="GreenCrab", 
    entry_point="rl4greencrab.green_crab_ipm:greenCrabEnv",
)
register(
    id="GreenCrabSimpl", 
    entry_point="rl4greencrab.green_crab_ipm:greenCrabSimplifiedEnv"
)
register(
    id="TimeSeries", 
    entry_point="rl4greencrab.time_series:TimeSeriesEnv",
)
