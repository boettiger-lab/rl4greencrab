from rl4greencrab.envs.green_crab_ipm import greenCrabEnv, greenCrabSimplifiedEnv
from rl4greencrab.envs.time_series import timeSeriesEnv
from rl4greencrab.envs.green_crab_monthly_env import greenCrabMonthEnv
from rl4greencrab.envs.green_crab_monthly_env_norm import greenCrabMonthEnvNormalized
from rl4greencrab.envs.green_crab_movingAvg import greenCrabMonthNormalizedMoving
from rl4greencrab.agents.const_action import constAction, constActionNatUnits, multiConstAction
from rl4greencrab.agents.const_escapement import constEsc
from rl4greencrab.agents.hyperparam import *
from rl4greencrab.agents.ensemble_ppo import *
from rl4greencrab.utils.simulate import simulator, get_simulator, evaluate_agent
from rl4greencrab.utils.plot_util import environment_simulation, plot_selected_sizes

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
    entry_point="rl4greencrab.envs.time_series:timeSeriesEnv",
)
register(
    id="monthenv", 
    entry_point="rl4greencrab.envs.green_crab_monthly_env:greenCrabMonthEnv",
)
register(
    id="monthenvnorm", 
    entry_point="rl4greencrab.envs.green_crab_monthly_env_norm:greenCrabMonthEnvNormalized",
)
register(
    id="simpleEnv", 
    entry_point="rl4greencrab.envs.simple_env:SimpleEnv",

)
register(
    id="monenvmoving", 
    entry_point="rl4greencrab.envs.green_crab_movingAvg:greenCrabMonthNormalizedMoving",
)

