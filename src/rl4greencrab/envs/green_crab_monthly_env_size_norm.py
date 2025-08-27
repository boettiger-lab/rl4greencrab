import gymnasium as gym
import logging
import numpy as np

from gymnasium import spaces
from scipy.stats import norm
from rl4greencrab.envs.green_crab_monthly_env_size import greenCrabMonthEnvSize

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

class greenCrabMonthEnvSizeNormalized(greenCrabMonthEnvSize):
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
            np.float32([-1, -1, -1]),
            np.float32([1, 1, 1]),
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