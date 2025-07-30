import numpy as np
import pandas as pd
from rl4greencrab import plot_agent, generate_policy_rews_df_dfs
from rl4greencrab.utils.sb3 import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-csv", "--csv_file", help="Path to csv file", type=str)
parser.add_argument("-agent", help="Path to agent", type=str)
parser.add_argument("-algo", help="algorithm used", type=str)
parser.add_argument("-agent_name", help="name of the agent", type=str)
parser.add_argument("-env_id", help="env to use", type=str)
parser.add_argument("-save_dir", help="where to save plots", type=str)
args = parser.parse_args()

env = gym.make(args.env_id)

if args.csv_file != None:
    df = pd.read_csv(args.csv_file)
else:
    raise ValueError("didn't provide csv to load")
    
if args.agent != None:
    ALGO = algorithm(args.algo)
    agent = ALGO.load(args.agent, device="cpu")

save_dir = args.save_dir
agent_name = args.agent_name

plotting_agent = plot_agent(env_sim_df=df, agent_name=agent_name, save_dir = save_dir)

plotting_agent.agent_action_overtime_plots()
plotting_agent.agent_ob_overtime_plots(obs_name='obs0')
plotting_agent.agent_ob_overtime_plots(obs_name='obs1')
plotting_agent.obs_vs_acts_plots(ob_name='obs0')
plotting_agent.obs_vs_acts_plots(ob_name='obs1')
plotting_agent.gaussian_smoothing()

# example useage: 
# python plot_script.py -csv ../notebooks/rl4greencrab/data/ppo_env.csv -agent_name ppo_agent -env_id monthenvnorm -save_dir .


