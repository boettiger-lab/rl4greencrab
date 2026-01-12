import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from rl4greencrab.envs.green_crab_monthly_env_norm import greenCrabMonthEnvNormalized
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from rl4greencrab.utils.plot_utils import *
from rl4greencrab.utils.simulate import *

config = {
    "w_mort_scale" : 600,
    "growth_k": 0.70,
    'random_start':True,
    'var_penalty_const': 0
    # "curriculum": True
}
greencrab_month_norm_env = greenCrabMonthEnvNormalized(config)

class plot_agent:
    def __init__(self, env_sim_df, agent_name, env=greencrab_month_norm_env, agent=None, save_dir='.'):
        self.env_simulation_df = env_sim_df
        self.agent_name = agent_name # simulation for a specific agent
        self.env = env
        self.agent = agent
        self.save_dir = os.path.join(save_dir, agent_name)
        if self.env_simulation_df is None:
            self.gen_env_sim_df()

    def agent_action_overtime_plots(self, rep=0, show=True):
        df = self.env_simulation_df
        df = df[df.rep == rep]
        
        action_cols = [c for c in df.columns if c.startswith("act")]
    
        fig, ax = plt.subplots(figsize=(8, 4))
    
        for col in action_cols:
            ax.plot(df["t"], df[col], label=col)

        ax.set_xlabel('t')
        ax.set_ylabel('Action value')
        ax.set_title('Actions over Time')
        ax.legend()
        
        self.save_fig(fig, f"actions_over_time.png")
        if show:
            plt.show()

    def agent_ob_overtime_plots(self, obs_name, rep=0, show=True):
        # Observation over Time
        fig = plt.figure(figsize=(8,4))
        df = self.env_simulation_df
        df = df[df.rep == rep]
        fig, ax = plt.subplots(figsize=(8, 4))

        # Handle case where each observation is an array
        first_val = df[obs_name].iloc[0]
        if isinstance(first_val, np.ndarray):
            obs_array = np.stack(df[obs_name].values)  # shape: (timesteps, features)
            timesteps = df['t'].values
            for i in range(obs_array.shape[1]):
                ax.plot(timesteps, obs_array[:, i], label=f"{obs_name}[{i}]")
            ax.legend()
        else:
            ax.plot(df['t'], df[obs_name], label=obs_name)
    
        ax.set_xlabel('t')
        ax.set_ylabel(obs_name)
        ax.set_title(f'{obs_name} over Time')
    
        self.save_fig(fig, f"{obs_name}_over_time.png")
        if show:
            plt.show()

    def obs_vs_acts_plots(self, ob_name, rep=0, show=True):
        fig = plt.figure(figsize=(6,6))
        df = self.env_simulation_df
        subset = df[df.rep == rep]
        action_cols = [c for c in subset.columns if c.startswith("act")]

        for col in action_cols:
            plt.scatter(subset[ob_name], subset[col], label=f'{col} vs {ob_name}', alpha=0.7)
        
        plt.title(f'Scatter: actions vs {ob_name}')
        plt.xlabel(ob_name)
        plt.ylabel('action')
        plt.legend()
        self.save_fig(fig, f"{ob_name}_vs_actions.png")
        if show:
            plt.show()

    def state_heatmap(self, rep=0, use_log=True):
        fig = state_heatmap(self.env_simulation_df, rep=rep, use_log=use_log)
        self.save_fig(fig, f"{ob_name}_state_headmap.png")

    # gaussian smooth the actions to generate gpp agents and df
    """ arguments:
    max_reps: max number of repitition used to fit gaussian process
    """
    def gaussian_smoothing(
        self,
        max_reps=10,
        reps=5,
        obs_prefix="obs",
        act_prefix="act",
        obs_cols=None,
        act_cols=None,
        length_scale=1.0,
        noise_level=1.0,
    ):
        df = self.env_simulation_df
        df = df[df["rep"] <= max_reps].copy()
    
        # pick columns
        if obs_cols is None:
            obs_cols = sorted([c for c in df.columns if c.startswith(obs_prefix)])
        if act_cols is None:
            act_cols = sorted([c for c in df.columns if c.startswith(act_prefix)])
    
        # keep only what GPP needs
        keep_cols = obs_cols + act_cols
        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in env_simulation_df: {missing}")
    
        df = df.loc[:, keep_cols]
    
        gpp = GaussianProcessPolicy(df, length_scale=length_scale, noise_level=noise_level)
        gpp_df, state_df = generate_gpp_episodes(gpp, self.env, reps=reps)
    
        out_path = os.path.join(self.save_dir, f"{self.agent_name}_GPP.csv.xz")
        gpp_df.to_csv(out_path, index=False)
    
        self.gpp_df = gpp_df
        return gpp_df, state_df

    def gen_env_sim_df(self, rep=10):
        if self.agent == None:
            raise ValueError("didn't provide an agent for simulation")
        data = environment_simulation(self.env, self.agent, reps=rep)
        self.env_simulation_df = pd.DataFrame(data)
        return self.env_simulation_df

    def load_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        self.env_simulation_df = df
        return df
    
    def save_df(self, df, name):
        os.makedirs(self.save_dir, exist_ok=True)
        df.to_csv(os.path.join(self.save_dir,f"{name}.csv"), index = False)

    # helper to save a figure
    def save_fig(self, fig, fname):
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            outpath = os.path.join(self.save_dir, fname)
            fig.savefig(outpath, dpi=300, bbox_inches='tight')
            print(f"Saved {outpath}")