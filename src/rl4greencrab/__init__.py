from rl4greencrab.envs.green_crab_ipm import greenCrabEnv, greenCrabSimplifiedEnv
from rl4greencrab.envs.time_series import timeSeriesEnv
from rl4greencrab.agents.const_action import constAction, constActionNatUnits
from rl4greencrab.agents.const_escapement import constEsc
from rl4greencrab.utils.simulate import simulator

from gymnasium.envs.registration import register
register(
    id="gcenv", 
    entry_point="rl4greencrab.envs.green_crab_ipm:greenCrabEnv",
)
register(
    id="gcsenv", 
    entry_point="rl4greencrab.envs.green_crab_ipm:greenCrabSimplifiedEnv"
)
register(
    id="tsenv", 
    entry_point="rl4greencrab.envs.time_series:TimeSeriesEnv",
)
