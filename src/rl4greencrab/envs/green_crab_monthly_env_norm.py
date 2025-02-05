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
        self.observation_space = spaces.Box(
            np.array([-1], dtype=np.float32),
            np.array([1], dtype=np.float32),
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
        action_natural_units = np.maximum(self.max_action * (1 + action)/2 , 0.) #convert to normal action
        obs, rew, term, trunc, info = super().step(
            np.float32(action_natural_units)
        )
        normalized_cpue = 2 * self.cpue_2(obs, action_natural_units) - 1
        # observation = np.float32(np.append(normalized_cpue, action))
        observation = normalized_cpue
        # rew = 10 * rew # use larger rewards, possibly makes trainer easier?
        return observation, rew, term, trunc, info

    def reset(self, *, seed=42, options=None):
        _, info = super().reset(seed=seed, options=options)

        # completely new  obs
        return - np.ones(shape=self.observation_space.shape, dtype=np.float32), info
        
    def cpue_2(self, obs, action_natural_units):
        # If you don't set traps, the catch-per-effort is 0/0.  Should be NaN, but we call it 0
        if np.sum(action_natural_units) <= 0:
            return np.float32([0])
        # can't tell which traps caught each number of crabs here. Perhaps too simple but maybe realistic 
        cpue_2 = np.float32([
            np.sum(obs[0]) / (self.cpue_normalization * np.sum(action_natural_units)),
        ])
        return cpue_2