import numpy as np
import cloudpickle as pickle  
from pathlib import Path
from stable_baselines3 import PPO
from rl4greencrab import sample_ppo_params
import optuna

class EnsemblePPO:
    def __init__(self, policy, env, 
                 tensorboard_log="",
                 n_agents=3, 
                 agents=[], 
                 verbose=0,
                 ):
        self.env = env
        self.policy = policy
        self.verbose = 0
        self.tensorboard_log = tensorboard_log
        if agents==[]:
            agents = self.make_ensemble_agents(env, n_agents)
        else:
            self.agents = agents

    def predict(self, obs, deterministic=True):
        # Collect action probabilities from each agent
        actions = []
        for agent in self.agents:
            act, _ = agent.predict(obs, deterministic=deterministic)
            actions.append(act)
        return np.mean(actions, axis=0), None

    def learn(self, total_timesteps, progress_bar=False):
        for i, agent in enumerate(self.agents):
            model = agent.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
            
    # save sub_trained_model
    def save(self, save_path):
        save_folder_path = Path(save_path)
        if not save_folder_path.exists():
            save_folder_path.mkdir(parents=True, exist_ok=True)
            
        # save meta data
        meta_data = {"env": self.env, 
                     "n_agents":len(self.agents),
                    "policy":self.policy,
                    }
        with open(f"{save_path}/models_info.pkl", "wb") as f:
            pickle.dump(meta_data, f)
            
        # save agents
        for i, agent in enumerate(self.agents):
            agent.save(f'{save_path}/model_{i}')

    def load(load_path):
        agents = []
        model_dir = Path(load_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {model_dir}")

        # load back meta_data
        meta_path = model_dir / "models_info.pkl"
        with open(meta_path, "rb") as f:
            meta_data = pickle.load(f)
        
        for zip_path in model_dir.glob("*.zip"):  # iterate over every *.zip in the folder
            name = zip_path.stem                  # filename without extension
            print(f"Loading {zip_path} …")
            agents.append(PPO.load(zip_path))

        return EnsemblePPO(
            env = meta_data['env'],
            policy = meta_data['policy'],
            n_agents = len(agents),
            agents = agents,
        )
        

    def make_ensemble_agents(self, env, n_agents: int):
        agents = []
        for _ in range(n_agents):
            study = optuna.create_study(direction="maximize")
            trial = study.ask()  # Dummy trial without actual optimization
            params = sample_ppo_params(trial, additional_args={})
            if self.tensorboard_log != "":
                agent = PPO(self.policy, env,verbose=self.verbose, tensorboard_log=self.tensorboard_log, **params)
            else:
                agent = PPO(self.policy, env,verbose=self.verbose, **params)
            agents.append(agent)

        self.agents = agents
        return agents
        