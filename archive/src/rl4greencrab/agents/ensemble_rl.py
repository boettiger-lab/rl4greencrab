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

class EnsembleRL:
    def __init__(self,env, 
                rppoAgent,
                 tqcAgent,
                 tensorboard_log="",
                 n_agents=2, 
                 verbose=0,
                 ):
        self.env = env
        self.verbose = 0
        self.tensorboard_log = tensorboard_log
        self.rppoAgent = rppoAgent
        self.tqcAgent = tqcAgent

    def predict(self, obs, deterministic=True):
        # Collect action probabilities from each agent
        actions = []
        rppo_action, _ = self.rppoAgent.predict(obs, deterministic=deterministic)
        tqc_action, _ = self.tqcAgent.predict(obs, deterministic=deterministic)
        final_action = np.array([tqc_action[0], tqc_action[1], rppo_action[2]])
        return final_action, None

if __name__ == "__main__":
    config = {
        "w_mort_scale" : 600,
        "growth_k": 0.70,
        'random_start':True,
        'var_penalty_const': 0
        # "curriculum": True
    }
    evalEnv =  greenCrabMonthEnvNormalized(config)
    td3Agent = TD3.load(f"varianceRatio0/TD3_gcmenorm", device="cpu")
    ppoAgent = PPO.load(f"varianceRatio0/PPO_gcmenorm", device="cpu")
    tqcAgent = TQC.load(f"varRatio0.3_new/TQC_gcmenorm", device="cpu")
    recurrentPPOAgent = RecurrentPPO.load("varianceRatio0.3/RecurrentPPO_gcmenorm_256_1_varR0.3", device="cpu")

    ensembelAgent = EnsembleRL(env = evalEnv, rppoAgent = recurrentPPOAgent, tqcAgent = tqcAgent)
    evalEnv =  greenCrabMonthEnvNormalized(config)
    ensembelAgent_plot_agent = plot_agent(env_sim_df=None, 
                                agent_name='ensembelAgent', 
                                env=evalEnv, 
                                agent=ensembelAgent, 
                                save_dir='.')
    ensembelAgent_plot_agent.gen_env_sim_df(rep=100)
    ensembelAgent_plot_agent.save_df(ensembelAgent_plot_agent.env_simulation_df, 'ensembelAgent_plot_agent_sim_500')
    ensembelAgent_plot_agent.agent_action_overtime_plots()