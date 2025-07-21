import gymnasium as gym
import logging
import numpy as np
import random

from gymnasium import spaces
from gymnasium.spaces import Tuple, Box, Discrete, Dict
from scipy.stats import norm


class greenCrabMonthEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config=None,
    ):
        config=config or {}
        
        # parameters
        self.seed = self.config.get("seed", None)
        self.RandNumbGen = np.random.default_rng(seed=self.seed)
        
        self.growth_k = np.float32(config.get("growth_k", 0.43))
        self.growth_xinf = np.float32(config.get("growth_xinf", 109))
        self.growth_sd = np.float32(config.get("growth_sd", 2.5))
        self.nmortality = np.float32(config.get("nmortality", 0.03))

        self.trapm_pmax = np.float32(config.get("trapm_pmax", 0.00044))
        self.trapm_sigma = np.float32(config.get("trapm_sigma", 6.47))
        self.trapm_xmax = np.float32(config.get("trapm_xmax", 45.02))
        #
        self.trapf_pmax = np.float32(config.get("trapf_pmax", 0.00029))
        self.trapf_k = np.float32(config.get("trapf_k", 0.36))
        self.trapf_midpoint = np.float32(config.get("trapf_midpoint", 38.72))
        #
        self.traps_pmax = np.float32(config.get("traps_pmax", 0.00448))
        self.traps_k = np.float32(config.get("traps_k", 0.33))
        self.traps_midpoint = np.float32(config.get("traps_midpoint", 46.47))
        
        self.init_mean_recruit = config.get("init_mean_recruit", 9.31)
        self.init_sd_recruit = config.get("init_sd_recruit", 1.5)
        self.init_mean_adult = config.get("init_mean_adult", 47.9)
        self.init_sd_adult = config.get("init_sd_adult", 8.1)
        self.init_n_recruit = config.get("init_n_recruit", 0)
        self.init_n_adult = config.get("init_n_adult", 0)
        
        self.w_mort_scale = config.get("w_mort_scale", 200)
        self.K = config.get("K", 25000) #carrying capacity

        self.imm = config.get("imm", 5000) # mean colonization/immigration rate --> randomize 
        
        self.r = config.get("r", 1) #intrinsic rate of growth

        self.max_action = config.get("max_action", 3000)
        self.max_obs = config.get("max_obs", 2000)
        
        self.area = config.get("area", 30000)
        self.loss_a = config.get("loss_a", 0.265)
        self.loss_b = config.get("loss_b", 2.80)
        self.loss_c = config.get("loss_c", 2.99)
        
        self.minsize = config.get("minsize", 5)
        self.maxsize = config.get("maxsize", 110)
        self.nsize = config.get("nsize", 21)
        self.ntime = config.get("ntime", 9)
        
        self.delta_t = config.get("delta_t", 1/12)
        self.env_stoch = config.get("env_stoch", 0.1)
        
        self.action_reward_scale = np.array(config.get("action_reward_scale", [0.08, 0.08, 0.4]))
        self.action_reward_exponent = config.get("action_reward_exponent", 1)
        
        self.config = config

        # Preserve these for reset
        self.observations = {"crabs": np.array([0, 0], dtype=np.float32), "months": 1}
        self.reward = 0
        self.month_passed = 0
        self.curr_month = 3 #start with third month
        self.Tmax = config.get("Tmax", 100)
                
        self.bndry = self.boundary() # boundary points for age class weights
        self.state = self.init_state()
        self.midpts = self.midpoints()
        self.gm_ker = self.g_m_kernel()
        self.w_mort = self.w_mortality()
        self.w_mort_exp = np.exp(-self.w_mort)
        self.pmort = np.exp(-self.nmortality)

        self.monthly_size = np.zeros(shape=(self.nsize,1),dtype='object')
        self.random_start = config.get('random_start', False)
        self.curriculum_enabled = config.get('curriculum', False)
        
        self.action_stacks = [] # storing whole year action -> store normalized action
        self.variance_penalty_ratio = config.get('var_penalty_const', 1)

        # Action space
        # action -- # traps per month
        self.action_space = spaces.Box(
            np.array([0, 0, 0], dtype=np.float32),
            np.array(3*[self.max_action], dtype=np.float32),
            dtype=np.float32,
        )

        self.max_mean_biomass = self.get_biomass_size()[-1]
        
        # Observation space with month observation feature
        self.observation_space = spaces.Dict({
           "crabs": spaces.Box(
                low=np.array([0, 0]),  # Lower bounds: original obs (0)
                high=np.array([self.max_obs, self.max_mean_biomass]),  # Upper bounds: obs max,
                shape=(2,),
                dtype=np.float32
            ), 
            "months": spaces.Discrete(12, start=1)
        })
        
    def step(self,action):
        #size selective harvest rate, given action
        harvest_rate = (
            1 - np.exp( -(
                self.size_sel_norm()*action[0] 
                #
                + self.size_sel_log(
                    self.trapf_pmax, self.trapf_midpoint, self.trapf_k
                ) * action[1] 
                #
                + self.size_sel_log(
                    self.traps_pmax, self.traps_midpoint, self.traps_k
                ) * action[2]
            ))
        )
        removed = np.zeros(shape=(self.nsize,1),dtype='object')
        size_freq = np.zeros(shape=(self.nsize,1),dtype='object')
        if self.curr_month == 3:
            size_freq[:,0] = self.state
            removed[:,0] = [
                rng.binomial(n=size_freq[k,0], p=harvest_rate[k]) 
                for k in range(self.nsize)
            ]
            self.action_stacks = []
        else:
            size_freq[:] = [
                rng.binomial(n=self.monthly_size[k].tolist(), p=self.pmort) 
                for k in range(self.nsize)
            ]
            removed[:] = [
                rng.binomial(n=size_freq[k].tolist(), p=harvest_rate[k]) 
                for k in range(self.nsize)
            ]
        self.monthly_size = self.gm_ker@(size_freq[:] - removed[:]) 

        # update actions stacks
        normalized_action = action / self.max_action * 2 - 1
        self.action_stacks.append(normalized_action)
        
        #update observation space
        biomass = np.sum(self.get_biomass_size() * removed[:,0])
        crab_counts = np.sum(removed[:,0])
        mean_biomass = biomass/crab_counts if crab_counts != 0 else 0
        self.observations = {
            "crabs": np.array([crab_counts, mean_biomass], dtype=np.float32), 
            "months": self.curr_month
        }
        
        self.state = self.monthly_size.reshape(21,)

        #calculate reward
        self.reward = self.reward_func(action)
        self.month_passed += 1
        self.curr_month += 1

        if self.curr_month > 11: 
            # winter mortality
            new_adults = [
                self.RandNumbGen.binomial(n=size_freq[k,0], p=self.w_mort_exp[k]) 
                for k in range(self.nsize) 
            ]

            #simulate new recruits for next year
            local_recruits = self.RandNumbGen.normal(
                loc=self.dd_growth(size_freq[:]), scale=self.env_stoch
            )
            
            nonlocal_recruits = self.non_localrecruit(size_freq)
            recruit_total = local_recruits + nonlocal_recruits
    
            #get sizes of recruits
            recruit_sizes = ( # integrated pdf between neighboring weights * total_recr
                norm.cdf(
                    x=self.bndry[1:(self.nsize+1)],
                    loc=self.init_mean_recruit,
                    scale=self.init_sd_recruit,
                ) 
                - norm.cdf(
                    x=self.bndry[0:self.nsize],
                    loc=self.init_mean_recruit,
                    scale=self.init_sd_recruit,
                )
            ) * recruit_total

            #store new population size (and cap off at zero pop)
            self.state = np.maximum(recruit_sizes + new_adults, 0)

            # jump to next year March
            self.curr_month = 3 

        done = bool(self.month_passed > self.Tmax)

        return self.observations, self.reward, done, done, {}

    def reset(self, *, seed=None, options=None):
        if not hasattr(self, "total_episodes_seen"):
            self.total_episodes_seen = 0
        else:
            self.total_episodes_seen += 1 
        
        self.state = self.init_state()
        self.month_passed = 0

        # for tracking only
        self.reward = 0

        # curriculumn learning
        if self.curriculum_enabled:
            # Increase difficulty slowly with training progress
            progress = self.get_curriculum_progress()  # value between 0 and 1
            low = int(self.max_obs * (0.4 - 0.2 * progress))   # gets wider over time
            high = int(self.max_obs * (0.6 + 0.2 * progress))
            low = max(0, low)
            high = min(self.max_obs, high)
        else:
            low = 0
            high = self.max_obs
        
        if self.random_start:
            # self.init_n_adult = self.np_random.integers(low, high + 1)
            pass

        self.observations = {
            "crabs": np.array([0, 0], dtype=np.float32), 
            "months": 1
        } 

        return self.observations, {}

    #################
    #helper functions
    
    # calculate progress value for curriculum training
    def get_curriculum_progress(self):
        """Returns a value from 0.0 to 1.0 based on total episodes seen."""
        return min(1.0, self.total_episodes_seen / 1000_000)

    #set up boundary points of IPM mesh
    def boundary(self):
        boundary = (
            self.minsize
            + np.arange(0, self.nsize+1, 1) * (self.maxsize - self.minsize)
            / self.nsize
        )
        return boundary

    #set up mid points of IPM mesh
    def midpoints(self):
        midpoints = 0.5*(self.bndry[0:self.nsize]+self.bndry[1:(self.nsize+1)])
        return midpoints

    #function for initial state
    def init_state(self):
        init_pop = (
            norm.cdf(
                self.bndry[1:(self.nsize+1)],
                self.init_mean_adult,self.init_sd_adult)
            - norm.cdf(
                self.bndry[0:self.nsize],
                self.init_mean_adult,self.init_sd_adult)
            ) * self.init_n_adult
            + (
                norm.cdf(
                    self.bndry[1:(self.nsize+1)],
                    self.init_mean_recruit,
                    self.init_sd_recruit
                )
                - norm.cdf(
                    self.bndry[0:self.nsize],
                    self.init_mean_recruit,
                    self.init_sd_recruit
                )
            ) * self.init_n_recruit
        return init_pop

    #function for logistic size selectivity curve
    def size_sel_log(self, trap_pmax, trap_midpts, trap_k):
        size_sel = trap_pmax / (
            1 + np.exp(-trap_k * (self.midpts - trap_midpts))
        )
        return size_sel

    #function for gaussian size selectivity curve
    def size_sel_norm(self):
        size_sel = self.trapm_pmax * np.exp(
            - (self.midpts - self.trapm_xmax) ** 2 
            / (2 * self.trapm_sigma ** 2)
        )
        return size_sel

    #function for growth/mortality kernel
    def g_m_kernel(self):
        array = np.empty(shape=(self.nsize,self.nsize),dtype='object')
        for i in range(self.nsize):
            mean = (
                (self.growth_xinf - self.midpts[i])
                * (1 - np.exp(- self.growth_k * self.delta_t)) 
                + self.midpts[i]
            )
            kernel[:,i] = (
                norm.cdf(
                    self.bndry[1:(self.nsize+1)],
                    mean,
                    self.growth_sd,
                )
                - norm.cdf(
                    self.bndry[0:self.nsize],
                    mean,
                    self.growth_sd,
                )
            )
        return kernel

    # function for overwinter mortality
    def w_mortality(self):
        wmort = self.w_mort_scale / self.midpts ** 2
        return wmort

    # function for density dependent growth
    def dd_growth(self,popsize):
        dd_recruits = np.sum(popsize) * self.r * (1-np.sum(popsize) / self.K)
        return dd_recruits

    def non_localrecruit(self, size_freq):
        big_year = self.RandNumbGen.choice(a=2, p=[0.8,0.2])
        deterministic_recruit = 1 - np.sum(size_freq[:]) / self.K

        if big_year:
            return max(self.RandNumbGen.normal(8e4, 1e4) * deterministic_recruit, 0) 
            
        else:
            return max(self.RandNumbGen.normal(8e3, 1e3) * deterministic_recruit, 0) 

    # function for getting biomass from crab size
    def get_biomass_size(self):
        biomass = [-0.071 * y + 0.003 * y**2 + 0.00002 * y**3 for y in self.midpts]
        return [np.max([0, b]) for b in biomass]
    
    # function for reward
    # two part reward function:
    # 1. impact on environment (function of crab density)
    # 2. penalty for how much effort we expended (function of action)
    def reward_func(self,action):
        def trap_cost(action, max_action, exponent):
            return np.array(
                [
                    (action[0]/max_action) ** exponent,
                    (action[1]/max_action) ** exponent,
                    (action[2]/max_action) ** exponent,
                ]
            )
        biomass = np.sum(self.get_biomass_size() * self.state) # get biomass
        reward = (
            -self.loss_a 
            /
            (
                1+np.exp(-self.loss_b*(biomass/self.area-self.loss_c))
            )
            - np.sum(
                self.action_reward_scale 
                * trap_cost(action, self.max_action, self.action_reward_exponent) 
            )
        )
        # discourage high std in a year
        if self.curr_month == 11:
            action_std = np.std(self.action_stacks, axis=0)
            reward -= self.variance_penalty_ratio * np.sum(action_std)
        return reward