import gymnasium as gym
import logging
import numpy as np
import random
from numpy.random import default_rng
from gymnasium import spaces
from gymnasium.spaces import Tuple, Box, Discrete, Dict
from scipy.stats import norm, lognorm, gamma

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

class twoActEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config=None,
    ):
        
        config=config or {}
        
        # parameters
        self.control_randomness = config.get("control_randomness", False)
        if self.control_randomness:
            self.np_random, _ = gym.utils.seeding.np_random(config.get("seed", 42))
            # migration‑only RNG
            self.mig_rng, _ = gym.utils.seeding.np_random(config.get("seed_migration", 1337))
        else:
            self.np_random = default_rng()
            self.mig_rng = default_rng()
        
        self.growth_k = np.float32(config.get("growth_k", 0.70))
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
        
        self.w_mort_scale = config.get("w_mort_scale", 600)
        self.K = config.get("K", 25000) # carrying capacity

        self.imm = config.get("imm", 5000) # mean colonization/immigration rate --> randomize 
        
        self.r = config.get("r", 1) # intrinsic rate of growth

        self.max_action = config.get("max_action", 3000)
        self.max_obs = config.get("max_obs", 2000)
        
        self.area = config.get("area", 30000)
        self.loss_a = config.get("loss_a", 0.265)
        self.loss_b = config.get("loss_b", 2.80)
        self.loss_c = config.get("loss_c", 2.99)
        
        self.minsize = config.get("minsize", 0)
        self.maxsize = config.get("maxsize", 110)
        self.nsize = config.get("nsize", 22)
        self.ntime = config.get("ntime", 9)
        
        self.delta_t = config.get("delta_t", 1/12)
        self.env_stoch = config.get("env_stoch", 0.1)
        
        self.action_reward_scale = np.array(config.get("action_reward_scale", [0.08, 0.08]))
        self.action_reward_exponent = config.get("action_reward_exponent", 1)
        
        self.config = config

        # Preserve these for reset
        # self.observations = np.zeros(shape=9, dtype=np.float32)
        # self.observations = (np.array([0, 0], dtype=np.float32), 1)
        self.reward = 0
        self.month_passed = 0
        self.curr_month = 3 #start with third month
        self.Tmax = config.get("Tmax", 100)
                
        # Initial variables
        self.bndry = self.boundary()
        self.state = np.zeros(self.nsize) # begin with popsize = 0
        self.midpts = self.midpoints()
        self.gm_ker = self.g_m_kernel()
        self.w_mort = self.w_mortality()
        self.w_mort_exp = np.exp(-self.w_mort)
        self.pmort = np.exp(-self.nmortality)

        self.random_start = config.get('random_start', False)
        self.curriculum_enabled = config.get('curriculum', False)

        self.action_stacks = [] # storing whole year action -> store normalized action
        self.variance_penalty_ratio = config.get('var_penalty_const', 1)
        self.non_local_crabs = []
        self.recruit_sizes = np.zeros(self.nsize)

        # Action space
        # action -- # traps per month
        self.action_space = spaces.Box(
            np.array([0, 0], dtype=np.float32),
            np.array(2*[self.max_action], dtype=np.float32),
            dtype=np.float32,
        )

        self.max_mean_biomass = self.get_biomass_size()[-1]

        self.observation_type = config.get('observation_type', 'count-biomass-time')
        self.observation_space = self.get_observations_space()
        self.observations = self.initial_observation()

        self.crab_caught = []
        
        
    def step(self,action):

        # size selective harvest rate, given action
        harvest_rate = (
            1 - np.exp( -(self.size_sel_norm() * action[0] + self.size_sel_log() * action[1]))
        )

        # create temporary size-structured pop
        pop = np.zeros(shape=(self.nsize, 1), dtype='object')
        pop[:,0] = self.state

        # calculate number removed
        removed = np.zeros(shape=(self.nsize, 1), dtype='object')
        removed[:,0] = [self.np_random.binomial(pop[k,0], harvest_rate[k]) for k in range(self.nsize)]

        if self.curr_month == 3:
            self.action_stacks = []

        # project one time step (growth and mortality)
        next_pop = self.gm_ker @ (pop - removed)
        
        # add recruits
        if self.curr_month == 5:
            next_pop = next_pop + self.recruit_sizes.reshape(self.nsize, 1)
        
        self.state = np.maximum(next_pop.reshape(self.nsize), 0)

        # update action stacks
        normalized_action = action / self.max_action * 2 - 1
        self.action_stacks.append(normalized_action)

        # update observation space
        biomass = np.sum(self.get_biomass_size() * removed[:,0])
        crab_counts = np.sum(removed[:,0])
        mean_biomass = biomass/crab_counts if crab_counts != 0 else 0
        self.observations = self.update_observation(crab_counts, mean_biomass, removed)
        self.crab_caught = removed[:,0]

        # calculate reward
        self.reward = self.reward_func(action)

        # iterate the month
        self.month_passed += 1
        self.curr_month += 1

        # calculate new adult population after overwinter mortality
        if self.curr_month > 11:
            state_col = self.state.reshape(self.nsize, 1)
            new_adults = [self.np_random.binomial(int(self.state[k]), self.w_mort_exp[k]) for k in range(self.nsize)]

            # simulate new recruits for next year
            local_recruits = self.np_random.normal(self.dd_growth(state_col), self.env_stoch)
            nonlocal_recruits = self.non_localrecurit(state_col)
            recruit_total = local_recruits + nonlocal_recruits

            logging.debug('local recruits = {}'.format(local_recruits))
            logging.debug('nonlocal recruits = {}'.format(nonlocal_recruits))

            # get recruit vector for next year
            var = self.init_sd_recruit ** 2
            shape = self.init_mean_recruit ** 2 / var
            rate = self.init_mean_recruit / var
            self.recruit_sizes = (
                (gamma.cdf(self.bndry[1:(self.nsize+1)], a=shape, scale=1/rate) -
                 gamma.cdf(self.bndry[0:self.nsize], a=shape, scale=1/rate)) * recruit_total
            )
            self.state = np.maximum(new_adults, 0)
            self.curr_month = 3

        done = bool(self.month_passed > self.Tmax)

        return self.observations, self.reward, done, done, {}
        
    def reset(self, *, seed=42, options=None):
        if not hasattr(self, "total_episodes_seen"):
            self.total_episodes_seen = 0
        else:
            self.total_episodes_seen += 1 
        
        self.state = np.zeros(self.nsize) # begin with popsize = 0
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
            self.init_n_adult = self.np_random.integers(low, high + 1)

        self.curr_month = 3
        self.observations = self.initial_observation()

        return self.observations, {}

    #################
    # helper functions
    def get_observations_space(self):
        if self.observation_type == 'count-biomass-time':
            return spaces.Dict({
               "crabs": spaces.Box(
                    low=np.array([0, 0]),  # Lower bounds: original obs (0)
                    high=np.array([self.max_obs, self.max_mean_biomass]),  # Upper bounds: obs max,
                    shape=(2,),
                    dtype=np.float32
                ), 
                "months": spaces.Discrete(12, start=1)
            })
        elif self.observation_type == 'count-time':
            return spaces.Dict({
               "crabs": spaces.Box(
                    low=np.array([0]),
                    high=np.array([self.max_obs]),
                    shape=(1,),
                    dtype=np.float32
                ),
                "months": spaces.Discrete(12, start=1)
            })
        elif self.observation_type == 'size-time':
            return  spaces.Dict({
                   "crabs": spaces.Box(
                    np.zeros(shape=self.nsize, dtype=np.float32),
                    self.max_obs * np.ones(shape=self.nsize, dtype=np.float32),
                    dtype=np.float32,
                ),
                "months": spaces.Discrete(12, start=1)
            })
        elif self.observation_type == 'size':
            return  spaces.Dict({
                   "crabs": spaces.Box(
                    np.zeros(shape=self.nsize, dtype=np.float32),
                    self.max_obs * np.ones(shape=self.nsize, dtype=np.float32),
                    dtype=np.float32,
                )
            })
        elif self.observation_type == 'count':
            return spaces.Dict({
               "crabs": spaces.Box(
                    low=np.array([0]),  # Lower bounds: original obs (0)
                    high=np.array([self.max_obs]),  # Upper bounds: obs max,
                    shape=(1,),
                    dtype=np.float32
                )
            })
        elif self.observation_type == 'count-biomass':
            return spaces.Dict({
               "crabs": spaces.Box(
                    low=np.array([0, 0]),  # Lower bounds: original obs (0)
                    high=np.array([self.max_obs, self.max_mean_biomass]),  # Upper bounds: obs max,
                    shape=(2,),
                    dtype=np.float32
                )
            })
        elif self.observation_type == 'biomass-time':
            return spaces.Dict({
               "crabs": spaces.Box(
                    low=np.array([0]),  # Lower bounds: original obs (0)
                    high=np.array([self.max_mean_biomass]),  # Upper bounds: obs max,
                    shape=(1,),
                    dtype=np.float32
                ),
                "months": spaces.Discrete(12, start=1)
            })
            
    def initial_observation(self):
        if self.observation_type == 'count-biomass-time':
            return {"crabs": np.array([0, 0], dtype=np.float32), "months": self.curr_month}
        if self.observation_type == 'count-time':
            return {"crabs": np.array([0], dtype=np.float32), "months": self.curr_month}
        if self.observation_type == 'size-time':
            return {"crabs":  np.zeros(self.nsize, dtype=np.float32), "months":  self.curr_month}
        if self.observation_type == 'biomass-time':
            return {"crabs": np.array([0], dtype=np.float32), "months": self.curr_month}
        if self.observation_type == 'size':
            return {"crabs":  np.zeros(self.nsize, dtype=np.float32)}
        if self.observation_type == 'count':
            return {"crabs": np.array([0], dtype=np.float32)}
        if self.observation_type == 'count-biomass':
            return {"crabs": np.array([0, 0], dtype=np.float32)}

        
    
    def update_observation(self, crab_counts, mean_biomass, removed):
        if self.observation_type == 'count-biomass-time':
            return {"crabs": np.array([crab_counts, mean_biomass], dtype=np.float32), "months": self.curr_month}
        if self.observation_type == 'count-time':
            return {"crabs": np.array([crab_counts], dtype=np.float32), "months": self.curr_month}
        if self.observation_type == 'size-time':
            return {"crabs": np.array(removed[:,0], dtype=np.float32), "months": self.curr_month}
        if self.observation_type == 'size':
            return {"crabs": np.array(removed[:,0], dtype=np.float32)}
        if self.observation_type == 'count':
            return {"crabs": np.array([crab_counts], dtype=np.float32)}
        if self.observation_type == 'count-biomass':
            return {"crabs": np.array([crab_counts, mean_biomass], dtype=np.float32)}
        if self.observation_type == 'biomass-time':
            return {"crabs": np.array([mean_biomass], dtype=np.float32), "months": self.curr_month}
        
    # calculate progress value for curriculum training
    def get_curriculum_progress(self):
        """Returns a value from 0.0 to 1.0 based on total episodes seen."""
        return min(1.0, self.total_episodes_seen / 1000_000)

    # set up boundary points of IPM mesh
    def boundary(self):
        boundary = self.minsize+np.arange(0,(self.nsize+1),1)*(self.maxsize-self.minsize)/self.nsize
        return boundary

    # set up mid points of IPM mesh
    def midpoints(self):
        midpoints = 0.5*(self.bndry[0:self.nsize]+self.bndry[1:(self.nsize+1)])
        return midpoints

    # function for logistic size selectivity curve
    def size_sel_log(self):
        size_sel = self.trapf_pmax/(1+np.exp(-self.trapf_k*(self.midpts-self.trapf_midpoint)))
        return size_sel

    # function for gaussian size selectivity curve
    def size_sel_norm(self):
        size_sel = self.trapm_pmax*np.exp(-(self.midpts-self.trapm_xmax)**2/(2*self.trapm_sigma**2))
        return size_sel

    #function for growth/mortality kernel
    def g_m_kernel(self):
        array = np.empty(shape=(self.nsize,self.nsize),dtype='object')
        for i in range(self.nsize):
            mean = (self.growth_xinf-self.midpts[i])*(1-np.exp(-self.growth_k*self.delta_t)) + self.midpts[i]
            array[:,i] = (norm.cdf(self.bndry[1:(self.nsize+1)],mean,self.growth_sd)-\
                          norm.cdf(self.bndry[0:self.nsize],mean,self.growth_sd))
        return array

    #function for overwinter mortality
    def w_mortality(self):
        wmort = self.w_mort_scale/self.midpts**2
        return wmort

    # function for density dependent growth
    def dd_growth(self,popsize):
        dd_recruits = np.sum(popsize)*self.r*(1-np.sum(popsize)/self.K)
        return dd_recruits

    # Calculate newborn green crab for the coming year
    def non_localrecurit(self, size_freq):
        # self.imm * self.np_random.lognormal()*(1-np.sum(size_freq[:])/self.K) # 0.2 for high 80000, 0.8 for low 8000 
        outcomes = [0, 1]
        probabilities = [0.8, 0.2]
        
        if self.mig_rng.choice(outcomes, size=1, replace=True, p=probabilities)[0] == 0:
            non_local_crabs = max(self.mig_rng.normal(8000, 1000) * (1-np.sum(size_freq[:])/self.K), 0)
        else:
            non_local_crabs = max(self.mig_rng.normal(80000, 10000) * (1-np.sum(size_freq[:])/self.K), 0) 

        self.non_local_crabs.append(non_local_crabs)
        return non_local_crabs

    # function for getting biomass from crab size
    def get_biomass_size(self):
        biomass = [-0.071 * y + 0.003 * y**2 + 0.00002 * y**3 for y in self.midpts]
        return [np.max([0, b]) for b in biomass]

    # function to get actual crab caught size distribution at time t
    def get_crab_caught(self):
        return self.crab_caught
    
    #function for reward
    # two part reward function:
    # 1. impact on environment (function of crab density)
    # 2. penalty for how much effort we expended (function of action)
    def reward_func(self,action):
        def trap_cost(action, max_action, exponent):
            return np.array(
                [
                    (action[0]/max_action) ** exponent,
                    (action[1]/max_action) ** exponent,
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