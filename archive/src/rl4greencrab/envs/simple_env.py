import gymnasium as gym
import logging
import numpy as np
import random

from gymnasium import spaces
from gymnasium.spaces import Tuple, Box, Discrete, Dict
from scipy.stats import norm

# A simple harvest model to test RL ability
# x_{t+1} = x_t + R x_t (1 - x_t / K) - F x_t
class SimpleEnv(gym.Env):
    def __init__(self, config=None):
        config=config or {}
        self.r = float(config.get("r", 0.01)) # TODO: randomize latter
        self.K = float(config.get("K", 100))
        
        self.action_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        ) # F harvest fraction

        self.observation_space = spaces.Box(
            low=np.array([0]), high=np.array([self.K]), shape=(1,), dtype=np.float32
        )
        self.timestep = 0
        self.Tmax = config.get("Tmax", 1500) # 100 years
        self.init_pop = config.get("init_pop", 10)
        self.state = self.init_pop
        self.reward = 0
        self.alpha = config.get("alpha", 1)
        self.randomize = config.get("randomize", False)

    def step(self, action):
        x_t = self.state
        harvest = action[0] * x_t
        self.state = np.max(x_t + self.r * x_t *(1 - x_t / self.K) - harvest, 0)
        # TODO: add noise
        observation = np.array([self.state], dtype=np.float32) 
        
        # calculate reward
        self.reward = self.reward_func(action, harvest)
        terminated = bool(self.timestep > self.Tmax)
        if self.state <= 0:
            terminated = True
        self.timestep += 1
        info = self. _get_info()
        return observation, self.reward, terminated, False, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.state = self.init_pop
        observation = self._get_obs()
        info = self. _get_info()
        self.timestep = 0
        if self.randomize:
            self.r = self.np_random.uniform(0, 1)
        return observation, info

    def _get_obs(self):
        return np.array([self.state], dtype=np.float32) 

    def _get_info(self):
        return {}

    def reward_func(self, action, harvest):
        # x_t = self.state
        # optimal_biomass = self.K / 2
        # penalty = self.alpha * (x_t - optimal_biomass)**2
        if self.state == 0:
            return -1
        return harvest
        
    