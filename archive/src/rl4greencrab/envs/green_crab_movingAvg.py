import gymnasium as gym
import logging
import numpy as np
import random

from gymnasium import spaces
from gymnasium.spaces import Tuple, Box, Discrete, Dict
from scipy.stats import norm
from rl4greencrab import greenCrabMonthEnvNormalized

class  greenCrabMonthNormalizedMoving(greenCrabMonthEnvNormalized):
    def __init__(self, config={}):
        super().__init__(config=config)
        self.n_obs = config.get('n_obs', 3)
        sample = self.observation_space.sample()
        zero_obs = {k: np.zeros_like(v) for k, v in sample.items()}
        self.obs_stacks = [zero_obs for _ in range(self.n_obs)]
        self.t = 1

    def reset(self, *, seed=None, options=None):
        init_state, init_info = super().reset(seed=seed, options=options)
        sample = self.observation_space.sample()
        zero_obs = {k: np.zeros_like(v) for k, v in sample.items()}
        self.obs_stacks = [zero_obs for _ in range(self.n_obs)]
        return init_state, init_info

    def step(self, action):
        new_state, reward, terminated, truncated, info = super().step(action)
        self.obs_stacks.pop(0)
        self.obs_stacks.append(new_state)
        obs_stacks = self._stacked_obs()
        new_obs = {}
        new_obs['crabs'] = np.average(obs_stacks['crabs'], axis=0)
        new_obs['months'] = obs_stacks['months'][-1]
        if self.t <  self.n_obs:
            new_obs['crabs'] = np.average(obs_stacks['crabs'][-self.t:], axis=0)
            new_obs['months'] = obs_stacks['months'][-1]
            self.t+=1
        return new_obs, reward, terminated, truncated, info
    
    def _stacked_obs(self):
        # Turn list of Dicts → Dict of stacked arrays
        return {
            k: np.stack([o[k] for o in self.obs_stacks], axis=0)
            for k in self.obs_stacks[0]
        }