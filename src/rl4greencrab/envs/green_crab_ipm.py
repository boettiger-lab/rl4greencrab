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

class greenCrabEnv(gym.Env):
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
        
        self.w_mort_scale = config.get("w_mort_scale", 500)
        self.K = config.get("K", 25000) #carrying capacity
        self.imm = config.get("imm", 1000) #colonization/immigration rate
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
        self.action_reward_exponent = config.get("action_reward_exponent", 10)
        
        self.config = config

        # Preserve these for reset
        self.observations = np.zeros(shape=9, dtype=np.float32)
        self.reward = 0
        self.years_passed = 0
        self.Tmax = config.get("Tmax", 100)
                
        # Initial variables
        self.bndry = self.boundary()
        self.state = self.init_state()
        self.midpts = self.midpoints()
        self.gm_ker = self.g_m_kernel()
        self.w_mort = self.w_mortality()
        self.w_mort_exp = np.exp(-self.w_mort)
        self.pmort = np.exp(-self.nmortality)

        # Action space
        # action -- # traps per month
        self.action_space = spaces.Box(
            np.array([0, 0, 0], dtype=np.float32),
            np.array(3*[self.max_action], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            np.zeros(shape=self.ntime, dtype=np.float32),
            self.max_obs * np.ones(shape=self.ntime, dtype=np.float32),
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

        #add pop at t=1
        size_freq = np.zeros(shape=(self.nsize,self.ntime),dtype='object')
        size_freq[:,0] = self.state
        
        #create array to store # removed
        removed = np.zeros(shape=(self.nsize,self.ntime),dtype='object')
        
        #calculate removed and record observation at t=1
        #apply monthly harvest rate
        removed[:,0] = [np.random.binomial(size_freq[k,0], harvest_rate[k]) for k in range(self.nsize)]
        # for k in range(self.nsize):
        #     removed[k,0] = np.random.binomial(size_freq[k,0], harvest_rate[k])

            
        #record the catch in the observation space
        self.observations[0] = np.sum(removed[:,0])


        #
        # From profiling: it seems like the model would run faster if the somatic growth/removal part was deterministic
        #
        # i.e. instead of ```np.random.binomial(n=n_j[k], p=...)``` have ```n_j[l] * self.pmort```
        #
        # could we model all the randomness as a final random vector added to size_freq[:, -1] ?
        #
        
        #loop through intra-annual change (9 total months), t=2+        
        for j in range(self.ntime-1):
            #project to next month
            n_j = self.gm_ker@(size_freq[:,j] - removed[:,j])

            size_freq[:,j+1] = [np.random.binomial(n=n_j[k], p=self.pmort) for k in range(self.nsize)]
            removed[:,j+1] = [np.random.binomial(size_freq[k,j+1], harvest_rate[k]) for k in range(self.nsize)]
            
        self.observations = np.array([np.sum(removed[:,j]) for j in range(self.ntime)], dtype = np.float32)
            
            # for k in range(21):
            #     #project to next size frequency
            #     size_freq[k,j+1] = np.random.binomial(
            #         n=n_j[k], 
            #         p=self.pmort
            #     )
            #     #apply monthly harvest rate
            #     removed[k,j+1] = np.random.binomial(size_freq[k,j+1], harvest_rate[k])
            # #record the catch/effort in the observation space
            # self.observations[j+1] = np.sum(removed[:,j+1])

        #calculate new adult population after overwinter mortality
        new_adults = [ np.random.binomial(size_freq[k,8],self.w_mort_exp[k]) for k in range(self.nsize) ]
        # new_adults = []
        # for k in range(21):
        #     new_adults = np.append(new_adults,np.random.binomial(size_freq[k,8],np.exp(-self.w_mort)[k]))

        #simulate new recruits
        local_recruits = np.random.normal(self.dd_growth(size_freq[:,self.ntime-1]),self.env_stoch)
        nonlocal_recruits = np.random.poisson(self.imm)*(1-np.sum(size_freq[:,self.ntime-1])/self.K)
        recruit_total = local_recruits + nonlocal_recruits

        logging.debug('local recruits = {}'.format(local_recruits))
        logging.debug('nonlocal recruits = {}'.format(nonlocal_recruits))

        #get sizes of recruits
        recruit_sizes = (norm.cdf(self.bndry[1:(self.nsize+1)],self.init_mean_recruit,self.init_sd_recruit)-\
         norm.cdf(self.bndry[0:self.nsize],self.init_mean_recruit,self.init_sd_recruit))*recruit_total

        #store new population size (and cap off at zero pop)
        self.state = np.maximum(recruit_sizes + new_adults, 0)

        #calculate reward
        self.reward = self.reward_func(action)
        self.years_passed += 1

        done = bool(self.years_passed > self.Tmax)

        # if np.sum(self.state) <= 0.001:
        #     done = True

        return self.observations, self.reward, done, done, {}
        
    def reset(self, *, seed=42, options=None):
        self.state = self.init_state()
        self.years_passed = 0

        # for tracking only
        self.reward = 0

        self.observations = np.zeros(shape=self.ntime, dtype=np.float32)

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
        wmort = self.w_mort_scale/self.midpts**2
        return wmort

    #function for density dependent growth
    def dd_growth(self,popsize):
        dd_recruits = np.sum(popsize)*self.r*(1-np.sum(popsize)/self.K)
        return dd_recruits

    # function for getting biomass from crab size
    def get_biomass_size(self):
        biomass = [-0.071 * y + 0.003 * y**2 + 0.00002 * y**3 for y in self.midpts]
        return [np.max([0, b]) for b in biomass]

    #function for reward
    # two part reward function:
    # 1. impact on environment (function of crab biomass)
    # 2. penalty for how much effort we expended (function of action)
    def reward_func(self, action):
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
        return reward


class greenCrabSimplifiedEnv(greenCrabEnv):
    """ like invasive_IPM but with simplified observations and normalized to -1, 1 space. """
    def __init__(self, config={}):
        super().__init__(config=config)
        self.observation_space = spaces.Box(
            np.array([-1,-1], dtype=np.float32),
            np.array([1,1], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            np.float32([-1, -1, -1]),
            np.float32([1, 1, 1]),
            dtype=np.float32,
        )
        self.max_action = config.get('max_action', 2000) # ad hoc based on previous values
        self.cpue_normalization = config.get('cpue_normalization', 100)
        
    def step(self, action):
        action_natural_units = np.maximum( self.max_action * (1 + action)/2 , 0.)
        obs, rew, term, trunc, info = super().step(
            np.float32(action_natural_units)
        )
        normalized_cpue = 2 * self.cpue_2(obs, action_natural_units) - 1
        # observation = np.float32(np.append(normalized_cpue, action))
        observation = normalized_cpue
        rew = 10 * rew # use larger rewards, possibly makes trainer easier?
        return observation, rew, term, trunc, info

    def reset(self, *, seed=42, options=None):
        _, info = super().reset(seed=seed, options=options)

        # completely new  obs
        return - np.ones(shape=self.observation_space.shape, dtype=np.float32), info

    def cpue_2(self, obs, action_natural_units):
        # If you don't set traps, the catch-per-effort is 0/0.  Should be NaN, but we call it 0
        if np.sum(action_natural_units) <= 0:
            return np.float32([0,0])
#            return np.float32([np.NaN,np.NaN]) 
        # can't tell which traps caught each number of crabs here. Perhaps too simple but maybe realistic 
        cpue_2 = np.float32([
            np.sum(obs[0:5]) / (self.cpue_normalization * np.sum(action_natural_units)),
            np.sum(obs[5:]) / (self.cpue_normalization * np.sum(action_natural_units))
        ])
        return cpue_2
        
        