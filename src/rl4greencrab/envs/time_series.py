import gymnasium as gym
from gymnasium import spaces
import numpy as np


class timeSeriesEnv(gym.Env):
    """
    takes an environment env and produces an new environemtn timeSeriesEnv(env)
    whose observations are timeseries of the env environment.
    """
    def __init__(self, config = {}):
        self.N_mem = config.get('N_mem', 3)
        if 'base_env' in config:
            self.base_env = config['base_env']
        else:
            from rl4greencrab import greenCrabSimplifiedEnv
            self.base_env = greenCrabSimplifiedEnv()
    
        self.action_space = self.base_env.action_space
        ones_shape = np.ones(
            shape = (self.N_mem, *self.base_env.observation_space.shape), 
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-ones_shape, +ones_shape)
        self.Tmax = self.base_env.Tmax
        #
        # [[state t], [state t-1], [state t-2], ..., [state t - (N_mem-1)]]
        # where each [state i] is a vector
        #
        _ = self.reset()
    
    def reset(self, *, seed=42, options=None):
        init_state, init_info = self.base_env.reset(seed=seed, options=options)
        empty_heap = - np.ones(shape = self.observation_space.shape, dtype=np.float32)
        self.heap = np.insert(
            empty_heap[0:-1],
            0,
            init_state,
            axis=0,
        )
        return self.heap, init_info
    
    def step(self, action):
        new_state, reward, terminated, truncated, info = self.base_env.step(action)
        self.heap = np.insert(
            self.heap[0:-1],
            0,
            new_state,
            axis=0,
        )
        return self.heap, reward, terminated, truncated, info
    
    def pop_to_state(self, pop):
        return self.base_env.pop_to_state(pop)
    
    def state_to_pop(self, state):
        return self.base_env.state_to_pop(state)




