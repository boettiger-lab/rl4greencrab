from rl4greencrab.envs.green_crab_ipm import greenCrabEnv, greenCrabSimplifiedEnv
from rl4greencrab.envs.time_series import timeSeriesEnv
from rl4greencrab.envs.green_crab_monthly_env import greenCrabMonthEnv
from rl4greencrab.agents.const_action import constAction, constActionNatUnits, multiConstAction
from rl4greencrab.agents.const_escapement import constEsc
from rl4greencrab.utils.simulate import simulator, get_simulator, evaluate_agent

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
register(
    id="monthenv", 
    entry_point="rl4greencrab.envs.green_crab_monthly_env:greenCrabMonthEnv",
)
