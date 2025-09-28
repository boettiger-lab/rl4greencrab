from rl4greencrab.envs.green_crab_ipm import greenCrabEnv, greenCrabSimplifiedEnv
from rl4greencrab.envs.time_series import timeSeriesEnv
from rl4greencrab.envs.green_crab_monthly_env import greenCrabMonthEnv
from rl4greencrab.envs.green_crab_monthly_env_norm import greenCrabMonthEnvNormalized
from rl4greencrab.envs.green_crab_monthly_env_simple import greenCrabMonthEnvSimple
from rl4greencrab.envs.green_crab_monthly_env_simple_norm import greenCrabMonthEnvSimpleNormalized
from rl4greencrab.envs.green_crab_monthly_env_size import greenCrabMonthEnvSize
from rl4greencrab.envs.green_crab_monthly_env_size_norm import greenCrabMonthEnvSizeNormalized
from rl4greencrab.envs.green_crab_movingAvg import greenCrabMonthNormalizedMoving
from rl4greencrab.envs.green_crab_env_2act import greenCrabMonthEnvTwoAct
from rl4greencrab.envs.green_crab_env_2act_norm import greenCrabMonthEnvTwoActNormalized
from rl4greencrab.agents.const_action import constAction, constActionNatUnits, multiConstAction
from rl4greencrab.agents.const_escapement import constEsc
from rl4greencrab.agents.LipschitzPPO import *
from rl4greencrab.agents.hyperparam import *
from rl4greencrab.agents.ensemble_ppo import *
from rl4greencrab.utils.simulate import simulator, get_simulator, evaluate_agent
from rl4greencrab.utils.plot_utils import *
from rl4greencrab.utils.plot_agents import *

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
    id="twoactmonth", 
    entry_point="rl4greencrab.envs.green_crab_env_2act:greenCrabMonthEnvTwoAct",
)
register(
    id="twoactmonthnorm", 
    entry_point="rl4greencrab.envs.green_crab_env_2act_norm:greenCrabMonthEnvTwoActNormalized",
)
register(
    id="monthenvnorm", 
    entry_point="rl4greencrab.envs.green_crab_monthly_env_norm:greenCrabMonthEnvNormalized",
)
register(
    id="gcmonthenvsimple", 
    entry_point="rl4greencrab.envs.green_crab_monthly_env_simple:greenCrabMonthEnvSimple",
)
register(
    id="gcmonthenvsimplenorm", 
    entry_point="rl4greencrab.envs.green_crab_monthly_env_simple_norm:greenCrabMonthEnvSimpleNormalized",
)
register(
    id="gcmonthenvsize", 
    entry_point="rl4greencrab.envs.green_crab_monthly_env_size:greenCrabMonthEnvSize",
)
register(
    id="gcmonthenvsizenorm", 
    entry_point="rl4greencrab.envs.green_crab_monthly_env_size_norm:greenCrabMonthEnvSizeNormalized",
)
register(
    id="simpleEnv", 
    entry_point="rl4greencrab.envs.simple_env:SimpleEnv",

)
register(
    id="monenvmoving", 
    entry_point="rl4greencrab.envs.green_crab_movingAvg:greenCrabMonthNormalizedMoving",
)

