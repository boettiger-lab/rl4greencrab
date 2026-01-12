import gymnasium as gym
import logging
import numpy as np

from gymnasium import spaces
from scipy.stats import norm
from rl4greencrab.envs.twoAction_cutomize import twoActEnv

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

class TwoActNormalized(twoActEnv):
    def __init__(self, config={}):
        super().__init__(config=config)
        
        self.observation_space = self.get_observations_space()
        
        self.action_space = spaces.Box(
            np.float32([-1, -1]),
            np.float32([1, 1]),
            dtype=np.float32,
        )
        self.max_action = config.get('max_action', 3000) # ad hoc based on previous values, prev = 2000
        self.cpue_normalization = config.get('cpue_normalization', 100)
        self.observation = self.initial_observation()
        
    def step(self, action):
        action_natural_units = np.maximum(self.max_action * (1 + action)/2 , 0.) #convert to normal action
        obs, rew, term, trunc, info = super().step(
            np.float32(action_natural_units)
        )
        if 'size' in self.observation_type:
            normalized_cpue = 2 * self.cpue_2_size(obs['crabs'], action_natural_units) - 1
        else:
            normalized_cpue = 2 * self.cpue_2_total(obs['crabs'], action_natural_units) - 1
        mean_biomass = obs["crabs"][1]
        normal_biomass = self.normalize_biomass(mean_biomass)
        
        self.observation = self.update_observation_norm(normalized_cpue, normal_biomass)
        # rew = 10 * rew # use larger rewards, possibly makes trainer easier?
        return self.observation, rew, term, trunc, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.observation = self.initial_observation()
        # completely new  obs
        return self.observation, info

    ###Helper Function###
    def get_observations_space(self):
        if self.observation_type == 'count-biomass-time':
            return spaces.Dict({
                "crabs": spaces.Box(
                    low=np.array([-1, -1]),  # Lower bounds: original obs (0), month (1)
                    high=np.array([1, 1]),  # Upper bounds: obs max, month max (12)
                    dtype=np.float32
                ), 
                "months": spaces.Discrete(12, start=1)
            })
        elif self.observation_type == 'size-time':
            return  spaces.Dict({
            "crabs": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.nsize,),
                    dtype=np.float32
                ),
            "months": spaces.Discrete(12, start=1)
        })
        elif self.observation_type == 'size':
            return  spaces.Dict({
                "crabs": spaces.Box(
                        low=-1.0,
                        high=1.0,
                        shape=(self.nsize,),
                        dtype=np.float32
                    )
            })
        elif self.observation_type == 'count-biomass':
            return spaces.Dict({
                "crabs": spaces.Box(
                    low=np.array([-1, -1]),  # Lower bounds: original obs (0), month (1)
                    high=np.array([1, 1]),  # Upper bounds: obs max, month max (12)
                    dtype=np.float32
                )
            })
    def update_observation_norm(self, normalized_cpue, normal_biomass):
        if self.observation_type == 'count-biomass-time':
            return {"crabs": np.array([normalized_cpue[0], normal_biomass], dtype=np.float32), "months": self.curr_month}
        if self.observation_type == 'size-time':
            return {"crabs": normalized_cpue,  "months": self.curr_month}
        if self.observation_type == 'size':
            return {"crabs": normalized_cpue}
        if self.observation_type == 'count-biomass':
            return {"crabs": np.array([normalized_cpue[0], normal_biomass], dtype=np.float32)}

    def initial_observation(self):
        if self.observation_type == 'count-biomass-time':
            return {"crabs": np.array([-1, -1], dtype=np.float32), "months": self.curr_month}
        if self.observation_type == 'size-time':
            return {"crabs": np.array([-1.0] * self.nsize, dtype=np.float32), "months": self.curr_month}
        if self.observation_type == 'size':
            return {"crabs": np.array([-1.0] * self.nsize, dtype=np.float32)}
        if self.observation_type == 'count-biomass':
            return {"crabs": np.array([-1, -1], dtype=np.float32)}
    
    def normalize_biomass(self, mean_biomass):
        biomass_sizes = self.get_biomass_size()
        b0 =  biomass_sizes[0]
        b20 = biomass_sizes[-1]
        return -1 + 2 * (mean_biomass - b0)/(b20 - b0)
    
    def cpue_2_total(self, obs, action_natural_units):
        # If you don't set traps, the catch-per-effort is 0/0.  Should be NaN, but we call it 0
        if np.sum(action_natural_units) <= 0:
            return np.float32([0])
        # can't tell which traps caught each number of crabs here. Perhaps too simple but maybe realistic 
        cpue_2 = np.float32([
            np.sum(obs[0]) / (self.cpue_normalization * np.sum(action_natural_units)),
        ])
        return cpue_2
        
    def cpue_2_size(self, obs, action_natural_units):
        # If you don't set traps, the catch-per-effort is 0/0.  Should be NaN, but we call it 0
        if np.sum(action_natural_units) <= 0:
            return np.array([0] * self.nsize, dtype=np.float32)
        # can't tell which traps caught each number of crabs here. Perhaps too simple but maybe realistic 
        cpue_2 = np.float32(
            obs / (self.cpue_normalization * np.sum(action_natural_units)),
        )
        return cpue_2