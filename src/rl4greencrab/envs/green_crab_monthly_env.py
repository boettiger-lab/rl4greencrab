import gymnasium as gym
import logging
import numpy as np

from gymnasium import spaces
from scipy.stats import norm

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

"""
Taken from IPM_202040117.ipynb, modified minor aspects to be able to interface
with ts_model.py
"""

class greenCrabMonthEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config=None,
    ):
        # if config == {}:
        #     config = {
        #         "Tmax": 100,
        #         "growth_k": 0.43, "growth_xinf": 109, "growth_sd": 2.5, "nmortality": 0.03,
        #         "trapm_sigma": 0.15, "trapm_xmax": 44, "trapm_pmax": 0.0005, "trapf_pmax": 0.0008,
        #         "trapf_k": 0.5, "trapf_midpoint": 45, "init_mean_recruit": 15, "init_sd_recruit": 1.5,
        #         "init_mean_adult": 65, "init_sd_adult": 8, "init_n_recruit": 1000, "init_n_adult": 1000,
        #         "w_mort_scale": 5, "K": 25000, "imm": 10, "r": 50, "area": 4000,"loss_a": 0.265,
        #         "loss_b": 2.80, "loss_c": 2.99, "minsize": 5, "maxsize": 110, "nsize": 21, "ntime":9,"delta_t": 1/12,
        #         "env_stoch": 0.1, "action_reward_scale":0.001
        #     }
        
        config=config or {}
        
        # parameters
        self.growth_k = np.float32(config.get("growth_k", 0.43))
        self.growth_xinf = np.float32(config.get("growth_xinf", 109))
        self.growth_sd = np.float32(config.get("growth_sd", 2.5))
        self.nmortality = np.float32(config.get("nmortality", 0.03))

        self.trapm_pmax = np.float32(config.get("trapm_pmax", 10 * 0.1 * 2.75e-5))
        self.trapm_sigma = np.float32(config.get("trapm_sigma", 6))
        self.trapm_xmax = np.float32(config.get("trapm_xmax", 47))
        #
        self.trapf_pmax = np.float32(config.get("trapf_pmax", 10 * 0.03 * 2.75e-5))
        self.trapf_k = np.float32(config.get("trapf_k", 0.4))
        self.trapf_midpoint = np.float32(config.get("trapf_midpoint", 41))
        #
        self.traps_pmax = np.float32(config.get("traps_pmax", 10 * 2.75e-5))
        self.traps_k = np.float32(config.get("traps_k", 0.4))
        self.traps_midpoint = np.float32(config.get("traps_midpoint", 45))
        
        self.init_mean_recruit = config.get("init_mean_recruit", 15)
        self.init_sd_recruit = config.get("init_sd_recruit", 1.5)
        self.init_mean_adult = config.get("init_mean_adult", 65)
        self.init_sd_adult = config.get("init_sd_adult", 8)
        self.init_n_recruit = config.get("init_n_recruit", 0)
        self.init_n_adult = config.get("init_n_adult", 0)
        
        self.w_mort_scale = config.get("w_mort_scale", 5)
        self.K = config.get("K", 25000) #carrying capacity

        self.imm = config.get("imm", 1000) #colonization/immigration rate
        self.theta = 5 #dispersion parameter for immigration
        
        self.r = config.get("r", 1) #intrinsic rate of growth

        self.max_action = config.get("max_action", 3000)
        self.max_obs = config.get("max_obs", 2000)
        
        self.area = config.get("area", 4000)
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
        self.action_reward_exponent = config.get("action_reward_exponent", 10)
        
        self.config = config

        # Preserve these for reset
        self.observations = np.zeros(shape=9, dtype=np.float32)
        self.reward = 0
        self.month_passed = 0
        self.curr_month = 3 #start with third month
        self.Tmax = config.get("Tmax", 100)
                
        # Initial variables
        self.bndry = self.boundary()
        self.state = self.init_state()
        self.midpts = self.midpoints()
        self.gm_ker = self.g_m_kernel()
        self.w_mort = self.w_mortality()
        self.w_mort_exp = np.exp(-self.w_mort)
        self.pmort = np.exp(-self.nmortality)

        self.monthly_size = np.zeros(shape=(self.nsize,1),dtype='object')

        # Action space
        # action -- # traps per month
        self.action_space = spaces.Box(
            np.array([0, 0, 0], dtype=np.float32),
            np.array(3*[self.max_action], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            np.zeros(shape=1, dtype=np.float32),
            self.max_obs ,
            dtype=np.float32,
        )
        
    def step(self,action):
        #size selective harvest rate, given action
        harvest_rate = (
            1 - np.exp( -(
                self.size_sel_norm()*action[0] 
                + self.size_sel_log(self.trapf_pmax, self.trapf_midpoint, self.trapf_k)*action[1] 
                + self.size_sel_log(self.traps_pmax, self.traps_midpoint, self.traps_k)*action[2]
            ))
        )
        removed = np.zeros(shape=(self.nsize,1),dtype='object')
        size_freq = np.zeros(shape=(self.nsize,1),dtype='object')
        if self.curr_month == 3:
            #add pop at t=1
            size_freq[:,0] = self.state
            #create array to store # removed
            #calculate removed and record observation at month = 3
            removed[:,0] = [np.random.binomial(size_freq[k,0], harvest_rate[k]) for k in range(self.nsize)]
        else:
            size_freq[:] = [np.random.binomial(n=self.monthly_size[k].tolist(), p=self.pmort) for k in range(self.nsize)]
            removed[:] = [np.random.binomial(size_freq[k].tolist(), harvest_rate[k]) for k in range(self.nsize)]
        self.monthly_size = self.gm_ker@(size_freq[:] - removed[:]) # calculate for next month
            
        #record the catch in the observation space
        self.observations[0] = np.sum(removed[:,0])
        #TODO: update self.state for every month or use different parameter for reward calculation
        self.state = self.monthly_size.reshape(21,) # calculate crab popluation after remove crab caught

        #calculate reward
        self.reward = self.reward_func(action)
        self.month_passed += 1
        self.curr_month += 1

        #calculate new adult population after overwinter mortality, how do we deal with for single month? 
        if self.curr_month > 11: 
            new_adults = [np.random.binomial(size_freq[k,0],self.w_mort_exp[k]) for k in range(self.nsize) ]

            #simulate new recruits for next year? 
            local_recruits = np.random.normal(self.dd_growth(size_freq[:]),self.env_stoch)
            mu = self.imm
            r = self.theta
            p = r / (r + mu)
            nonlocal_recruits = np.random.negative_binomial(r,p)*(1-np.sum(size_freq[:])/self.K)
            recruit_total = local_recruits + nonlocal_recruits
    
            logging.debug('local recruits = {}'.format(local_recruits))
            logging.debug('nonlocal recruits = {}'.format(nonlocal_recruits))
    
            #get sizes of recruits
            recruit_sizes = (norm.cdf(self.bndry[1:(self.nsize+1)],self.init_mean_recruit,self.init_sd_recruit)-\
             norm.cdf(self.bndry[0:self.nsize],self.init_mean_recruit,self.init_sd_recruit))*recruit_total

            #store new population size (and cap off at zero pop)
            self.state = np.maximum(recruit_sizes + new_adults, 0)

        if (self.curr_month > 11) : self.curr_month = 3 # jump to next year March

        done = bool(self.month_passed > self.Tmax)

        # if np.sum(self.state) <= 0.001:
        #     done = True

        return self.observations, self.reward, done, done, {}
        
    def reset(self, *, seed=42, options=None):
        self.state = self.init_state()
        self.years_passed = 0

        # for tracking only
        self.reward = 0

        self.observations = np.zeros(shape=1, dtype=np.float32)

        return self.observations, {}

    #################
    #helper functions

    #set up boundary points of IPM mesh
    def boundary(self):
        boundary = self.minsize+np.arange(0,(self.nsize+1),1)*(self.maxsize-self.minsize)/self.nsize
        return boundary

    #set up mid points of IPM mesh
    def midpoints(self):
        midpoints = 0.5*(self.bndry[0:self.nsize]+self.bndry[1:(self.nsize+1)])
        return midpoints

    #function for initial state
    def init_state(self):
        init_pop = (norm.cdf(self.bndry[1:(self.nsize+1)],self.init_mean_adult,self.init_sd_adult)-\
         norm.cdf(self.bndry[0:self.nsize],self.init_mean_adult,self.init_sd_adult))*self.init_n_adult+\
        (norm.cdf(self.bndry[1:(self.nsize+1)],self.init_mean_recruit,self.init_sd_recruit)-\
         norm.cdf(self.bndry[0:self.nsize],self.init_mean_recruit,self.init_sd_recruit))*self.init_n_recruit
        return init_pop

    #function for logistic size selectivity curve
    def size_sel_log(self, trap_pmax, trap_midpts, trap_k):
        size_sel = trap_pmax/(1+np.exp(-trap_k*(self.midpts-trap_midpts)))
        return size_sel

    #function for gaussian size selectivity curve
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
        wmort = self.w_mort_scale/self.midpts
        return wmort

    #function for density dependent growth
    def dd_growth(self,popsize):
        dd_recruits = np.sum(popsize)*self.r*(1-np.sum(popsize)/self.K)
        return dd_recruits

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
                    (action[2]/max_action) ** exponent,
                ]
            )
        reward = (
            -self.loss_a 
            /
            (
                1+np.exp(-self.loss_b*(np.sum(self.state)/self.area-self.loss_c))
            )
            - np.sum(
                self.action_reward_scale 
                * trap_cost(action, self.max_action, self.action_reward_exponent) 
            )
        )
        return reward