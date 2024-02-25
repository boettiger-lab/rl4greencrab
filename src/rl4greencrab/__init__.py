from rl4greencrab.envs.green_crab_ipm import greenCrabEnv, greenCrabSimplifiedEnv
from rl4greencrab.envs.time_series import timeSeriesEnv
from rl4greencrab.agents.const_action import constAction
from rl4greencrab.agents.const_escapement import constEsc
from rl4greencrab.utils.simulate import simulator

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
