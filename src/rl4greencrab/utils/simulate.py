class simulator:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def simulate(self, reps=10):
        self.results = []
        env = self.env
        agent = self.agent
        for rep in range(reps): # try score as average of 100 replicates, still a noisy measure
            episode_reward = 0.0
            observation, _ = env.reset()
            for t in range(env.Tmax):
                action, _ = agent.predict(observation, deterministic=True)
                observation, reward, terminated, done, info = env.step(self.unnormalize_action(action))
                observation = self.normalize_observation(observation)
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
                action = self.unnormalize_action(action)
                observation, reward, terminated, done, info = env.step(action)
                observation = self.normalize_observation(observation)
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

    def unnormalize_action(self, action):
        min_act = self.env.action_space.low
        act_width = self.env.action_space.high - self.env.action_space.low
        return min_act + act_width * (action + 1) / 2

    def normalize_observation(self, observation):
        min_obs = self.env.observation_space.low
        obs_width = self.env.observation_space.high - self.env.observation_space.low   
        return -1 + 2 * (observation - min_obs) / obs_width 
        

    