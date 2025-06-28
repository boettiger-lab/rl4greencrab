import gymnasium as gym
import logging
import numpy as np

from gymnasium import spaces
from scipy.stats import norm
from rl4greencrab.envs.green_crab_monthly_env import greenCrabMonthEnv

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

class greenCrabMonthEnvNormalized(greenCrabMonthEnv):
    def __init__(self, config={}):
        super().__init__(config=config)
        # self.observation_space = spaces.Box(
        #     np.array([-1], dtype=np.float32),
        #     np.array([1], dtype=np.float32),
        #     dtype=np.float32,
        # )
        self.observation_space = spaces.Dict({
            "crabs": spaces.Box(
                low=np.array([-1, -1]),  # Lower bounds: original obs (0), month (1)
                high=np.array([1, 1]),  # Upper bounds: obs max, month max (12)
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
        
    def step(self, action):
        action_natural_units = np.maximum(self.max_action * (1 + action)/2 , 0.) #convert to normal action
        obs, rew, term, trunc, info = super().step(
            np.float32(action_natural_units)
        )
        normalized_cpue = 2 * self.cpue_2(obs['crabs'], action_natural_units) - 1
        mean_biomass = obs["crabs"][1]
        normal_biomass = self.normalize_biomass(mean_biomass)
        # TODO: normalize biomass
        self.observation = {"crabs": np.array([normalized_cpue[0], normal_biomass], dtype=np.float32), "months": obs['months']}
        # rew = 10 * rew # use larger rewards, possibly makes trainer easier?
        return self.observation, rew, term, trunc, info

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)

        # completely new  obs
        return {"crabs":np.array([-1, -1], dtype=np.float32), "months":1}, info

    def normalize_biomass(self, mean_biomass):
        biomass_sizes = self.get_biomass_size()
        b0 =  biomass_sizes[0]
        b20 = biomass_sizes[-1]
        return -1 + 2 * (mean_biomass - b0)/(b20 - b0)
    
    def cpue_2(self, obs, action_natural_units):
        # If you don't set traps, the catch-per-effort is 0/0.  Should be NaN, but we call it 0
        if np.sum(action_natural_units) <= 0:
            return np.float32([0])
        # can't tell which traps caught each number of crabs here. Perhaps too simple but maybe realistic 
        cpue_2 = np.float32([
            np.sum(obs[0]) / (self.cpue_normalization * np.sum(action_natural_units)),
        ])
        return cpue_2