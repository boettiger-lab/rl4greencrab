import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
parser.add_argument("--plots", help="a list of plots to create",  nargs='+', type=str)
args = parser.parse_args()

env = gym.make(args.env_id)

plot_items = ['agent_action_overtime_plots', 
              'agent_ob_overtime_plots',
              'obs_vs_acts_plots',
              'gaussian_smoothing'
             ]

if args.csv_file != None:
    df = pd.read_csv(args.csv_file)
else:
    raise ValueError("didn't provide csv to load")

if args.plots != None:
    plot_items = args.plots
    
if args.agent != None:
    ALGO = algorithm(args.algo)
    agent = ALGO.load(args.agent, device="cpu")

save_dir = args.save_dir
agent_name = args.agent_name

plotting_agent = plot_agent(env_sim_df=df, agent_name=agent_name, save_dir = save_dir)

for p in plot_items:
    if p == 'agent_action_overtime_plots':
        plotting_agent.agent_action_overtime_plots()
    elif p == 'agent_ob_overtime_plots':
        plotting_agent.agent_ob_overtime_plots(obs_name='obs0')
        plotting_agent.agent_ob_overtime_plots(obs_name='obs1')
    elif p == 'obs_vs_acts_plots':
        plotting_agent.obs_vs_acts_plots(ob_name='obs0')
        plotting_agent.obs_vs_acts_plots(ob_name='obs1')
    elif p == 'gaussian_smoothing':
        plotting_agent.gaussian_smoothing()

#clean up
plt.close("all")

# example useage: 
# python plot_script.py -csv ../notebooks/rl4greencrab/data/ppo_env.csv -agent_name ppo_agent -env_id monthenvnorm -save_dir .

