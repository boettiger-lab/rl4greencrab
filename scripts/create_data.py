from rl4greencrab.envs.simple_env import SimpleEnv
from stable_baselines3 import PPO, TD3
from sb3_contrib import TQC, RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from rl4greencrab.envs.green_crab_monthly_env_norm import greenCrabMonthEnvNormalized
from rl4greencrab import evaluate_agent, multiConstAction, simulator
import pandas as pd
import numpy as np
from rl4greencrab import plot_agent
import ray
from skopt import gp_minimize, gbrt_minimize 
from skopt.plots import plot_convergence, plot_objective
from rl4greencrab import greenCrabMonthEnvTwoAct, greenCrabMonthEnvTwoActNormalized, greenCrabMonthEnvTwoActSize, greenCrabMonthEnvTwoActSizeNormalized

agent_dir = '../saved_agents/twoActEnv'
env_id = 'twoactmonthsizenorm'
save_dir_name = '../notebooks/greencrab_two_act_size_env'

td3Agent = TD3.load(f"{agent_dir}/TD3-{env_id}-1", device="cpu")
ppoAgent = PPO.load(f"{agent_dir}/PPO-{env_id}-1", device="cpu")
tqcAgent = TQC.load(f"{agent_dir}/TQC-{env_id}-1", device="cpu")
recurrentPPOAgent = RecurrentPPO.load(f"{agent_dir}/RecurrentPPO-{env_id}-1", device="cpu")

# td3Agent_var = TD3.load(f"{agent_dir}/TD3-{env_id}-2", device="cpu")
# ppoAgent_var = PPO.load(f"{agent_dir}/PPO-{env_id}-2", device="cpu")
# tqcAgent_var = TQC.load(f"{agent_dir}/TQC-{env_id}-2", device="cpu")
# recurrentPPOAgent_var = RecurrentPPO.load(f"{agent_dir}/RecurrentPPO-{env_id}-2", device="cpu")

agent_list = [td3Agent, ppoAgent, tqcAgent, recurrentPPOAgent]

config = {
    "w_mort_scale" : 600,
    "growth_k": 0.70,
    'random_start':True,
    'var_penalty_const': 0
    # "curriculum": True
}

evalEnv =  greenCrabMonthEnvTwoActSizeNormalized(config)
N_EPS = 30

ppo_rew = evaluate_agent(agent=ppoAgent, env=evalEnv, ray_remote=False).evaluate(n_eval_episodes=N_EPS)
td3_rew = evaluate_agent(agent=td3Agent, env=evalEnv, ray_remote=False).evaluate(n_eval_episodes=N_EPS)
tqc_rew = evaluate_agent(agent=tqcAgent, env=evalEnv, ray_remote=False).evaluate(n_eval_episodes=N_EPS)
reppo_rew = evaluate_agent(agent=recurrentPPOAgent, env=evalEnv, ray_remote=False).evaluate(n_eval_episodes=N_EPS)

print(f"""
PPO mean rew = {ppo_rew}
TQC mean rew = {tqc_rew}
TD3 mean rew = {td3_rew}
RecurrentPPO mean rew = {reppo_rew}
""")

# ppo_rew = evaluate_agent(agent=ppoAgent_var, env=evalEnv, ray_remote=False).evaluate(n_eval_episodes=N_EPS)
# td3_rew = evaluate_agent(agent=td3Agent_var, env=evalEnv, ray_remote=False).evaluate(n_eval_episodes=N_EPS)
# tqc_rew = evaluate_agent(agent=tqcAgent_var, env=evalEnv, ray_remote=False).evaluate(n_eval_episodes=N_EPS)
# reppo_rew = evaluate_agent(agent=recurrentPPOAgent_var, env=evalEnv, ray_remote=False).evaluate(n_eval_episodes=N_EPS)

# print(f"""
# PPO mean rew = {ppo_rew}
# TQC mean rew = {tqc_rew}
# TD3 mean rew = {td3_rew}
# RecurrentPPO mean rew = {reppo_rew}
# """)

# start generating dataframe using env simulation
config = {
    "w_mort_scale" : 600,
    "growth_k": 0.70,
    'random_start':True,
    'var_penalty_const': 0,
    'control_randomness': True
    # "curriculum": True
}

for agent in agent_list:
    agent_name = agent.__class__.__name__
    evalEnv =  greenCrabMonthEnvTwoActSizeNormalized(config)
    ppo_plot_agent = plot_agent(env_sim_df=None, 
                                agent_name=f'{save_dir_name}/{agent_name}_agent_size', 
                                env=evalEnv, 
                                agent=ppoAgent, 
                                save_dir='.')
    df = ppo_plot_agent.gen_env_sim_df(rep=1, obs_names=['crabs','months'])
    ppo_plot_agent.save_df(ppo_plot_agent.env_simulation_df, f'{agent_name}_sim_500')

