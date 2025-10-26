import gymnasium as gym
import logging
import numpy as np
from numpy.random import default_rng
import random

from gymnasium import spaces
from gymnasium.spaces import Tuple, Box, Discrete, Dict
from scipy.stats import norm
from rl4greencrab.envs.green_crab_env_2act import greenCrabMonthEnvTwoAct

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

"""
Taken from IPM_202040117.ipynb, modified minor aspects to be able to interface
with ts_model.py
"""

class greenCrabMonthEnvTwoActSize(greenCrabMonthEnvTwoAct):

    def __init__(self, config={}):
        super().__init__(config=config)
        
        # Observation space with month observation feature
        self.observation_space = spaces.Dict({
               "crabs": spaces.Box(
                np.zeros(shape=self.nsize, dtype=np.float32),
                self.max_obs * np.ones(shape=self.nsize, dtype=np.float32),
                dtype=np.float32,
            ),
            "months": spaces.Discrete(12, start=1)
        })

    def step(self,action):
        #size selective harvest rate, given action
        harvest_rate = (
            1 - np.exp( -(
                self.size_sel_norm()*action[0] 
                + self.size_sel_log(self.trapf_pmax, self.trapf_midpoint, self.trapf_k)*action[1]
            ))
        )
        removed = np.zeros(shape=(self.nsize,1),dtype='object')
        size_freq = np.zeros(shape=(self.nsize,1),dtype='object')
        if self.curr_month == 3:
            #add pop at t=1
            size_freq[:,0] = self.state
            #create array to store # removed
            #calculate removed and record observation at month = 3
            removed[:,0] = [self.np_random.binomial(size_freq[k,0], harvest_rate[k]) for k in range(self.nsize)]
            self.action_stacks = []
        else:
            size_freq[:] = [self.np_random.binomial(n=self.monthly_size[k].tolist(), p=self.pmort) for k in range(self.nsize)]
            removed[:] = [self.np_random.binomial(size_freq[k].tolist(), harvest_rate[k]) for k in range(self.nsize)]
        self.monthly_size = self.gm_ker@(size_freq[:] - removed[:]) # calculate for greencrab pop for next month

        # update actions stacks
        normalized_action = action / self.max_action * 2 - 1
        self.action_stacks.append(normalized_action)
        
        #update observation space
        biomass = np.sum(self.get_biomass_size() * removed[:,0]) # get biomass
        crab_counts = np.sum(removed[:,0])
        mean_biomass = biomass/crab_counts if crab_counts != 0 else 0
        
        self.observations = {"crabs": np.array(removed[:,0], dtype=np.float32),
                            "months": self.curr_month}
        
        self.state = self.monthly_size.reshape(21,) # calculate crab popluation after remove crab caught

        #calculate reward
        self.reward = self.reward_func(action)
        self.month_passed += 1
        self.curr_month += 1

        #calculate new adult population after overwinter mortality, how do we deal with for single month? 
        if self.curr_month > 11: 
            new_adults = [self.np_random.binomial(size_freq[k,0],self.w_mort_exp[k]) for k in range(self.nsize) ]

            #simulate new recruits for next year
            local_recruits = self.np_random.normal(self.dd_growth(size_freq[:]),self.env_stoch)
            
            nonlocal_recruits = self.non_localrecurit(size_freq)
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
    
    def reset(self, *, seed=None, options=None):
        if not hasattr(self, "total_episodes_seen"):
            self.total_episodes_seen = 0
        else:
            self.total_episodes_seen += 1 
        
        self.state = self.init_state()
        self.month_passed = 0
        self.curr_month = 3
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
    
        self.observations = {"crabs":  np.zeros(self.nsize, dtype=np.float32),
                            "months":  self.curr_month} # potentially start with end of previous year
        self.non_local_crabs = []
        return self.observations, {}

class greenCrabMonthEnvTwoActSizeNormalized(greenCrabMonthEnvTwoActSize):
    def __init__(self, config={}):
        super().__init__(config=config)

        self.observation_space = spaces.Dict({
            "crabs": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.nsize,),
                    dtype=np.float32
                ),
            "months": spaces.Discrete(12, start=1)
        })
        
        self.action_space = spaces.Box(
            np.float32([-1, -1]),
            np.float32([1, 1]),
            dtype=np.float32,
        )
        self.max_action = config.get('max_action', 3000) # ad hoc based on previous values, prev = 2000
        self.cpue_normalization = config.get('cpue_normalization', 100)
        self.observation = {"crabs": np.array([-1.0] * self.nsize, dtype=np.float32),
                           "months": spaces.Discrete(12, start=1)}
        
    def step(self, action):
        action_natural_units = np.maximum(self.max_action * (1 + action)/2 , 0.) #convert to normal action
        obs, rew, term, trunc, info = super().step(
            np.float32(action_natural_units)
        )
        normalized_cpue = 2 * self.cpue_2(obs['crabs'], action_natural_units) - 1
        self.observation = {"crabs": normalized_cpue,  "months": self.curr_month}
        return self.observation, rew, term, trunc, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.observation = {"crabs": np.array([-1.0] * self.nsize, dtype=np.float32),
                           "months": 1}
        # completely new  obs
        return self.observation, info
    
    def cpue_2(self, obs, action_natural_units):
        # If you don't set traps, the catch-per-effort is 0/0.  Should be NaN, but we call it 0
        if np.sum(action_natural_units) <= 0:
            return np.array([0] * self.nsize, dtype=np.float32)
        # can't tell which traps caught each number of crabs here. Perhaps too simple but maybe realistic 
        cpue_2 = np.float32(
            obs / (self.cpue_normalization * np.sum(action_natural_units)),
        )
        return cpue_2
    