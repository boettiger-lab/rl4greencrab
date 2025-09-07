import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rl4greencrab import plot_agent, generate_policy_rews_df_dfs
from rl4greencrab.utils.sb3 import *

agent_name_lists = ['zero_constant', 'optimal_constant', 'ppo', 'rppo', 'tqc', 'td3']

SAVE_DIR = './sample_agents_output'

CSV_DIR = '../notebooks/rl4greencrab/data/sim_rep500/'

for agent_name in agent_name_lists:
    save_dir = SAVE_DIR
    agent_name = agent_name
    csv_file = os.path.join(CSV_DIR, f'{agent_name}_sim_500.csv')
    
    # load dataframe
    df = pd.read_csv(csv_file)

    # create plotting agent
    plotting_agent = plot_agent(env_sim_df=df, agent_name=agent_name, save_dir = save_dir)
    
    plotting_agent.agent_action_overtime_plots(show=False)
    plotting_agent.agent_ob_overtime_plots(obs_name='obs0', show=False)
    plotting_agent.agent_ob_overtime_plots(obs_name='obs1', show=False)
    plotting_agent.obs_vs_acts_plots(ob_name='obs0', show=False)
    plotting_agent.obs_vs_acts_plots(ob_name='obs1', show=False)
    plotting_agent.gaussian_smoothing()

    plt.close("all")
