from gymnasium import spaces
import numpy as np
from scipy.stats import norm

from rl4greencrab import greenCrabEnv

class greenCrabNineMonth(greenCrabEnv):
    def __init__(self, config={}):
        super().__init__(config=config)
        self.observation_space = spaces.Box(
            np.float32(9 * [-1]),
            np.float32(9 * [1]),
        )
        self.action_space = spaces.Box(
            np.float32(9 * [-1]),
            np.float32(9 * [1]),
        )
        self.action_reward_scale = config.get("action_reward_scale", 0.05)
        self.max_action = 10_000

    def reset(self, *, seed=42, options=None):
        unnorm_obs, info = super().reset()
        norm_obs = 2 * unnorm_obs / self.max_obs - 1
        return norm_obs, info

    def step(self,action):
        #size selective harvest rate, given action
        harvest_rate = np.array(
            [
                1 - np.exp( -(
                    self.size_sel_log(
                        self.traps_pmax, 
                        self.traps_midpoint, 
                        self.traps_k
                    ) * self.max_action * (1+action[i]) / 2
                ))
                for i in range(self.ntime)
            ]
        ) # n_size columns, ntime rows

        #add pop at t=1
        size_freq = np.zeros(shape=(self.nsize,self.ntime),dtype='object')
        size_freq[:,0] = self.state
        
        #create array to store # removed
        removed = np.zeros(shape=(self.nsize,self.ntime),dtype='object')
        
        #calculate removed and record observation at t=0
        #apply monthly harvest rate
        removed[:,0] = [np.random.binomial(size_freq[k,0], harvest_rate[0, k]) for k in range(self.nsize)]
            
        #record the catch in the observation space
        self.observations[0] = np.sum(removed[:,0])

        
        #loop through intra-annual change (9 total months), t=2+        
        for j in range(self.ntime-1):
            #project to next month
            n_j = self.gm_ker@(size_freq[:,j] - removed[:,j])

            size_freq[:,j+1] = [np.random.binomial(n=n_j[k], p=self.pmort) for k in range(self.nsize)]
            removed[:,j+1] = [np.random.binomial(size_freq[k,j+1], harvest_rate[j+1,k]) for k in range(self.nsize)]

        # get observations and normalize them
        self.observations = np.array([np.sum(removed[:,j]) for j in range(self.ntime)], dtype = np.float32)
        self.observations = 2 * self.observations / self.max_obs - 1 


        #calculate new adult population after overwinter mortality
        new_adults = [ np.random.binomial(size_freq[k,8],self.w_mort_exp[k]) for k in range(self.nsize) ]

        #simulate new recruits
        local_recruits = np.random.normal(self.dd_growth(size_freq[:,self.ntime-1]),self.env_stoch)
        nonlocal_recruits = np.random.poisson(self.imm)*(1-np.sum(size_freq[:,self.ntime-1])/self.K)
        recruit_total = local_recruits + nonlocal_recruits

        #get sizes of recruits
        recruit_sizes = (norm.cdf(self.bndry[1:(self.nsize+1)],self.init_mean_recruit,self.init_sd_recruit)-\
         norm.cdf(self.bndry[0:self.nsize],self.init_mean_recruit,self.init_sd_recruit))*recruit_total

        #store new population size (and cap off at zero pop)
        self.state = np.maximum(recruit_sizes + new_adults, 0)

        #calculate reward
        self.reward = self.reward_func(action)
        self.years_passed += 1

        done = bool(self.years_passed > self.Tmax)

        return self.observations, self.reward, done, done, {}

    def reward_func(self,action):
        def month_trap_cost(self, action_at_month: np.float32):
            cost_of_1 = 1 - (99/self.max_action)**0.8 - ((self.max_action - 100)/self.max_action) ** 4
            n_traps =  self.max_action * (1+action_at_month) / 2
            if n_traps < 1:
                return 0
            elif 1 <= n_traps < 100:
                return cost_of_1 + ((n_traps-1) / self.max_action) ** 0.8
            elif 100 <= n_traps:
                return cost_of_1 + (99/self.max_action) ** 0.8 + ((n_traps - 100) / self.max_action) ** 4

        costs = np.float32([month_trap_cost(self, action_at_month) for action_at_month in action])
        total_cost = np.sum(costs) * self.action_reward_scale

        ecological_damage = (
            self.loss_a / (
                1+np.exp(
                    - self.loss_b * (
                        np.sum(self.state) / self.area - self.loss_c
                    )
                )
            )
        )
        
        reward = - ecological_damage - total_cost
        print(f"eco: {ecological_damage:.5f}, cost: {total_cost:.5f}")
        return reward
        
        