from rl4greencrab.envs.twoAction_cutomize import twoActEnv
from rl4greencrab.envs.twoAction_norm import TwoActNormalized
from rl4greencrab.agents.const_action import constAction, constActionNatUnits, multiConstAction
from rl4greencrab.agents.const_escapement import constEsc
from rl4greencrab.agents.LipschitzPPO import *
from rl4greencrab.agents.hyperparam import *
from rl4greencrab.utils.simulate import simulator, get_simulator, evaluate_agent
from rl4greencrab.utils.plot_utils import *
from rl4greencrab.utils.plot_agents import *

from gymnasium.envs.registration import register
register(
    id="twoactenv", 
    entry_point="rl4greencrab.envs.twoAction_cutomize:twoActEnv",
)
register(
    id="twoactenvnorm", 
    entry_point="rl4greencrab.envs.twoAction_norm:TwoActNormalized",
)


