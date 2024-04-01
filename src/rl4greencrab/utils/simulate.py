import numpy as np

class simulator:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def simulate(self, reps=1):
        self.results = []
        env = self.env
        agent = self.agent
        for rep in range(reps): # try score as average of 100 replicates, still a noisy measure
            episode_reward = 0.0
            observation, _ = env.reset()
            for t in range(env.Tmax):
                action, _ = agent.predict(observation, deterministic=True)
                observation, reward, terminated, done, info = env.step(action)
                episode_reward += reward
                if terminated or done:
                    break
            self.results.append(episode_reward)      
        return self.results

    def simulate_full(self, reps=10):
        observation_list = []
        action_list = []
        ep_rew_list = []
        reps_list = []
        t_list = []
        #
        env = self.env
        agent = self.agent
        for rep in range(reps): # try score as average of 100 replicates, still a noisy measure
            episode_reward = 0.0
            observation, _ = env.reset()
            for t in range(env.Tmax):
                action, _ = agent.predict(observation, deterministic=True)
                observation, reward, terminated, done, info = env.step(action)
                episode_reward += reward
                #
                observation_list.append(observation)
                action_list.append(action)
                ep_rew_list.append(episode_reward)
                reps_list.append(rep)
                t_list.append(t)
                #
                if terminated or done:
                    break
        return {
            't': t_list,
            'obs': observation_list,
            'act': action_list,
            'rew': ep_rew_list,
            'rep': reps_list,
        }

    def simulate_full_named_obs_acts(self, reps=10, obs_names = None, acts_names = None):
        num_obs = np.prod(self.env.observation_space.shape)
        num_acts = np.prod(self.env.action_space.shape)
        obs_names = obs_names or [f'obs{i}' for i in range(num_obs)]
        acts_names = acts_names or [f'act{i}' for i in range(num_acts)]
        #
        data = {
            't': [],
            **{obsn: [] for obsn in obs_names},
            **{actn: [] for actn in acts_names},
            'rew': [],
            'rep': [],
        }
        env = self.env
        agent = self.agent
        for rep in range(reps): # try score as average of 100 replicates, still a noisy measure
            episode_reward = 0.0
            observation, _ = env.reset()
            for t in range(env.Tmax):
                action, _ = agent.predict(observation, deterministic=True)
                #
                data['rew'].append(episode_reward)
                data['rep'].append(rep)
                data['t'].append(t)
                for idx, obs_name in enumerate(obs_names):
                    data[obs_name].append(observation[idx])
                for idx, act_name in enumerate(acts_names):
                    data[act_name].append(action[idx])
                #
                observation, reward, terminated, done, info = env.step(action)
                episode_reward += reward
                #
                if terminated or done:
                    break
        return data












        
        

    