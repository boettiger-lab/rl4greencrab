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
        
        ######################
        # randomness control #
        ######################

        self.control_randomness = config.get("control_randomness", False)
        if self.control_randomness:
            self.np_random, _ = gym.utils.seeding.np_random(config.get("seed", 42))
            # migration‑only RNG
            self.mig_rng, _ = gym.utils.seeding.np_random(config.get("seed_migration", 1337))
        else:
            self.np_random = default_rng()
            self.mig_rng = default_rng()
        
        self.random_start = config.get('random_start', False)

        ###########################################################
        # pop dynamic model parameters (not drawn from posterior) #
        ###########################################################

        # overwinter motrality
        self.w_mort_scale = config.get("w_mort_scale", 600)

        # growth and density dependence
        self.K = config.get("K", 25000) # carrying capacity
        self.imm = config.get("imm", 5000) # mean colonization/immigration rate --> randomize 
        self.r = config.get("r", 1) # intrinsic rate of growth
        self.env_stoch = config.get("env_stoch", 0.1)
        
        # constants
        self.max_action = config.get("max_action", 3000)
        self.max_obs = config.get("max_obs", 2000)
        self.minsize = config.get("minsize", 0)
        self.maxsize = config.get("maxsize", 110)
        self.nsize = config.get("nsize", 22)
        self.ntime = config.get("ntime", 9)
        self.bndry = self.boundary()
        self.midpts = self.midpoints()
        

        ###################
        # reward function #
        ###################

        # action penalization
        self.action_reward_scale = np.array(config.get("action_reward_scale", [0.08, 0.08]))
        self.action_reward_exponent = config.get("action_reward_exponent", 1)

        # ecological change
        self.area = config.get("area", 30000)
        self.loss_a = config.get("loss_a", 0.265)
        self.loss_b = config.get("loss_b", 2.80)
        self.loss_c = config.get("loss_c", 2.99)

                
        #####################
        # Initial variables #
        #####################

        self.config = config

        self.reward = 0
        self.month_passed = 0
        self.curr_month = 4 #start with fourth month
        self.Tmax = config.get("Tmax", 100)

        # initial state (proper init happens in reset after param draw)
        self.state = np.zeros(self.nsize)

        # winter mortality
        self.w_mort = self.w_mortality()
        self.w_mort_exp = np.exp(-self.w_mort)

        # empty vectors for recruits
        self.non_local_crabs = []
        self.recruit_sizes = np.zeros(self.nsize)


        ################
        # Action space #
        ################

        # action -- # traps per month
        self.action_stacks = [] # storing whole year action -> store normalized action
        self.action_space = spaces.Box(
            np.array([0, 0], dtype=np.float32),
            np.array(2*[self.max_action], dtype=np.float32),
            dtype=np.float32,
        )


        #####################
        # Observation space #
        #####################

        self.max_mean_biomass = self.get_biomass_size()[-1]
        self.observation_type = config.get('observation_type', 'count-biomass-time')
        self.observation_space = self.get_observations_space()
        self.observations = self.initial_observation()

        self.crab_caught = []
        
        
    def step(self, action):

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

        if self.curr_month == 4:
            self.action_stacks = []

        # project one time step (growth and mortality)
        D1 = self.D[self.curr_month - 4]
        D2 = self.D[self.curr_month - 3]
        survival = np.exp(-(D2 - D1) * (self.mort_beta + self.mort_alpha / self.midpts ** 2))
        proj_matrix = self.g_kernel(D1, D2) * survival
        next_pop = proj_matrix @ (pop - removed)
        
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
        if self.curr_month > 10:

            # convert state into a column
            state_col = self.state.reshape(self.nsize, 1)

            # project forward based on growth and stochastic overwinter mortality
            D1 = self.D[self.curr_month - 4]
            D2 = self.D[0] + 1.0
            next_pop = self.g_kernel(D1, D2) @ self.state
            new_adults = [self.np_random.binomial(int(next_pop[k]), self.w_mort_exp[k]) for k in range(self.nsize)]

            # simulate new recruits for next year
            local_recruits = self.np_random.normal(self.dd_growth(state_col), self.env_stoch)
            nonlocal_recruits = self.non_localrecruit(state_col)
            recruit_total = local_recruits + nonlocal_recruits

            # get recruit vector for next year
            var = self.init_sd_recruit ** 2
            shape = self.init_mean_recruit ** 2 / var
            rate = self.init_mean_recruit / var
            self.recruit_sizes = (
                (gamma.cdf(self.bndry[1:(self.nsize+1)], a=shape, scale=1/rate) -
                 gamma.cdf(self.bndry[0:self.nsize], a=shape, scale=1/rate)) * recruit_total
            )

            # add new adults to state
            self.state = np.maximum(new_adults, 0)

            # reset month
            self.curr_month = 4

        done = bool(self.month_passed > self.Tmax)

        return self.observations, self.reward, done, done, {}
        
    def reset(self, *, seed=42, options=None):

        #######################################################
        # pop dynamic model parameters (drawn from posterior) #
        #######################################################

        # get posterior index
        param_df = self.config.get("param_df", None)
        index = self.np_random.integers(0, len(param_df))

        # growth parameters
        self.growth_k = np.float32(param_df.loc[index, 'growth_k'])
        self.growth_xinf = np.float32(param_df.loc[index, 'growth_xinf'])
        self.growth_sd = np.float32(param_df.loc[index, 'growth_sd'])
        self.growth_A  = np.float32(param_df.loc[index, 'growth_A']) 
        self.growth_ds = np.float32(param_df.loc[index, 'growth_ds'])
        self.D = (np.array([91, 121, 152, 182, 213, 244, 274, 305]) - 91) / 365

        # natural mortality
        self.mort_alpha = np.float32(param_df.loc[index, 'mort_alpha'])
        self.mort_beta = np.float32(param_df.loc[index, 'mort_beta'])

        # trap size-selective hazard rate parameters
        # minnow traps
        self.trapm_pmax = np.float32(param_df.loc[index, 'trapm_pmax'])
        self.trapm_sigma = np.float32(param_df.loc[index, 'trapm_sigma'])
        self.trapm_xmax = np.float32(param_df.loc[index, 'trapm_xmax'])
        # fukui traps
        self.trapf_pmax = np.float32(param_df.loc[index, 'trapf_pmax'])
        self.trapf_k = np.float32(param_df.loc[index, 'trapf_k'])
        self.trapf_midpoint = np.float32(param_df.loc[index, 'trapf_midpoint'])
        # shrimp traps
        #self.traps_pmax = np.float32(param_df.loc[index, 'traps_pmax'])
        #self.traps_k = np.float32(param_df.loc[index, 'traps_k'])
        #self.traps_midpoint = np.float32(param_df.loc[index, 'traps_midpoint'])
        
        # initial adult and recruit sizes
        self.init_mean_recruit = np.float32(param_df.loc[index, 'init_mean_recruit'])
        self.init_sd_recruit = np.float32(param_df.loc[index, 'init_sd_recruit'])
        self.init_mean_adult = np.float32(param_df.loc[index, 'init_mean_adult'])
        self.init_sd_adult = np.float32(param_df.loc[index, 'init_sd_adult'])

        if not hasattr(self, "total_episodes_seen"):
            self.total_episodes_seen = 0
        else:
            self.total_episodes_seen += 1 
        
        self.month_passed = 0

        # for tracking only
        self.reward = 0

        if self.random_start:
            self.init_n_adult = self.np_random.integers(0, self.max_obs + 1)
        else:
            self.init_n_adult = config.get("init_n_adult", 0)
        self.state = self.init_state()

        self.curr_month = 4
        self.observations = self.initial_observation()

        return self.observations, {}


    ####################
    # helper functions #
    ####################

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
        
    
    # set up boundary points of IPM mesh
    def boundary(self):
        boundary = self.minsize+np.arange(0,(self.nsize+1),1)*(self.maxsize-self.minsize)/self.nsize
        return boundary

    # set up mid points of IPM mesh
    def midpoints(self):
        midpoints = 0.5*(self.bndry[0:self.nsize]+self.bndry[1:(self.nsize+1)])
        return midpoints

    # function for initial state
    def init_state(self):
        init_pop = (lognorm.cdf(self.bndry[1:(self.nsize+1)], s=self.init_sd_adult, scale=np.exp(self.init_mean_adult)) -
            lognorm.cdf(self.bndry[0:self.nsize], s=self.init_sd_adult, scale=np.exp(self.init_mean_adult))) * self.init_n_adult

        return init_pop

    # function for logistic size selectivity curve
    def size_sel_log(self):
        size_sel = self.trapf_pmax/(1+np.exp(-self.trapf_k*(self.midpts-self.trapf_midpoint)))
        return size_sel

    # function for gaussian size selectivity curve
    def size_sel_norm(self):
        size_sel = self.trapm_pmax*np.exp(-(self.midpts-self.trapm_xmax)**2/(2*self.trapm_sigma**2))
        return size_sel

    # function for seasonal growth kernel
    def g_kernel(self, D1, D2):
        S_t  = (self.growth_A * self.growth_k / (2 * np.pi)) * np.sin(2 * np.pi * (D2 - self.growth_ds))
        S_t0 = (self.growth_A * self.growth_k / (2 * np.pi)) * np.sin(2 * np.pi * (D1 - self.growth_ds))
        array = np.empty(shape=(self.nsize, self.nsize), dtype='object')
        for i in range(self.nsize):
            increment = (self.growth_xinf - self.midpts[i]) * (1 - np.exp(-self.growth_k * (D2 - D1) - S_t + S_t0))
            mean = self.midpts[i] + increment
            array[:,i] = (norm.cdf(self.bndry[1:(self.nsize+1)], mean, self.growth_sd) -
                          norm.cdf(self.bndry[0:self.nsize], mean, self.growth_sd))
        # normalize columns
        for i in range(self.nsize):
            array[:,i] = array[:,i] / np.sum(array[:,i])
        return array

    # function for overwinter mortality
    def w_mortality(self):
        wmort = self.w_mort_scale/self.midpts**2
        return wmort

    # function for density dependent growth
    def dd_growth(self,popsize):
        dd_recruits = np.sum(popsize)*self.r*(1-np.sum(popsize)/self.K)
        return dd_recruits

    # Calculate newborn green crab for the coming year
    def non_localrecruit(self, pop):
        
        outcomes = [0, 1]
        probabilities = [0.8, 0.2]
        
        if self.mig_rng.choice(outcomes, size=1, replace=True, p=probabilities)[0] == 0:
            non_local_crabs = max(self.mig_rng.normal(8000, 1000) * (1-np.sum(pop[:])/self.K), 0)
        else:
            non_local_crabs = max(self.mig_rng.normal(80000, 10000) * (1-np.sum(pop[:])/self.K), 0) 

        self.non_local_crabs.append(non_local_crabs)
        return non_local_crabs

    # function for getting biomass from crab size
    def get_biomass_size(self):
        biomass = [-0.071 * y + 0.003 * y**2 + 0.00002 * y**3 for y in self.midpts]
        return [np.max([0, b]) for b in biomass]

    # function to get actual crab caught size distribution at time t
    def get_crab_caught(self):
        return self.crab_caught
    
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
        return reward