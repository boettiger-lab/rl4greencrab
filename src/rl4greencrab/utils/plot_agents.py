import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from rl4greencrab.utils.plot_utils import *
from rl4greencrab.utils.simulate import *

class plot_agent:
    def __init__(self, env_sim_df, agent_name, env=None, agent=None, save_dir='.'):
        self.env_simulation_df = env_sim_df
        self.agent_name = agent_name # simulation for a specific agent
        self.env = env
        self.agent = agent
        self.save_dir = os.path.join(save_dir, agent_name)
        print(self.save_dir)

    def agent_action_overtime_plots(self):
        df = self.env_simulation_df
        fig = plt.figure(figsize=(8,4))
        df[df.rep == 0].plot(
            x='t', y=['act0','act1','act2'],
            title="Actions over Time"
        )
        self.save_fig(fig, f"actions_over_time.png")
        plt.show()

    def agent_ob_overtime_plots(self, obs_name):
        # Observation over Time
        fig = plt.figure(figsize=(8,4))
        df[df.rep == 0].plot(
            x='t', y=obs_name,
            title=f"{obs_name} over Time"
        )
        self.save_fig(fig, f"{obs_name}_over_time.png")
        plt.show()

    def obs_vs_acts_plots(self, ob_name):
        fig = plt.figure(figsize=(6,6))
        subset = df[df.rep == 0]
        plt.scatter(subset[ob_name], subset['act0'], label=f'act0 vs {ob_name}', alpha=0.7)
        plt.scatter(subset[ob_name], subset['act1'], label=f'act1 vs {ob_name}', alpha=0.7)
        plt.scatter(subset[ob_name], subset['act2'], label=f'act2 vs {ob_name}', alpha=0.7)
        plt.title(f'Scatter: actions vs {ob_name}')
        plt.xlabel(ob_name)
        plt.ylabel('action')
        plt.legend()
        self.save_fig(fig, f"{ob_name}_vs_actions.png")
        plt.show()

    # gaussian smooth the actions to generate gpp agents and df
    def gaussian_smoothing(self):
        df = self.env_simulation_df.loc[:, ['obs0', 'obs1', 'act0', 'act1', 'act2']]
        gpp= GaussianProcessPolicy(df, length_scale=1, noise_level=1)
        gpp_df, state_df= generate_gpp_episodes(gpp, env, reps=5)
        gpp_df.to_csv(os.path.join(self.save_dir,f"ppo{ITERATIONS}_GPP.csv.xz"), index = False)
        self.gpp_df = gpp_df
        return gpp_df, state_df

    def gen_env_sim_df(self):
        self.env_simulation_df = pd.DataFrame(environment_simulation(self.env, self.agent))
        return self.env_simulation_df
    
    def save_df(self, df, name):
        df.to_csv(os.path.join(self.save_dir,f"{name}.csv"), index = False)

    # helper to save a figure
    def save_fig(self, fig, fname):
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            outpath = os.path.join(self.save_dir, fname)
            fig.savefig(outpath, dpi=300, bbox_inches='tight')
            print(f"Saved {outpath}")