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
        self.configuration = config.get('config', {})
        self.is_dict = False
        if 'base_env' in config:
            self.base_env = config['base_env']
        else:
            from rl4greencrab import greenCrabSimplifiedEnv
            self.base_env = greenCrabSimplifiedEnv(self.configuration)
    
        self.action_space = self.base_env.action_space
        
        if self.base_env.observation_space.shape == None:
            self.is_dict = True
            self.observation_space = timeSeriesEnv.stack_dict_space(
                self.base_env.observation_space, n_mem=self.N_mem
            ) 
        else:
            ones_shape = np.ones(
                shape = (self.N_mem, *self.base_env.observation_space.shape), 
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(-ones_shape, +ones_shape)
        
        self.Tmax = self.base_env.Tmax
        sample = self.base_env.observation_space.sample()
        zero_obs = {k: np.zeros_like(v) for k, v in sample.items()}
        self._buf = [zero_obs for _ in range(self.N_mem)]
        #
        # [[state t], [state t-1], [state t-2], ..., [state t - (N_mem-1)]]
        # where each [state i] is a vector
        #
        _ = self.reset()
    
    def reset(self, *, seed=42, options=None):
        init_state, init_info = self.base_env.reset(seed=seed, options=options)
        if self.is_dict:
            self._buf = [init_state] * self.N_mem       # fill buffer
            return self._stacked_obs(), init_info
        else:
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
        if self.is_dict:
            self._buf.pop(0); self._buf.append(new_state)  # roll buffer
            return self._stacked_obs(), reward, terminated, truncated, info
        else:
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

    def _stacked_obs(self):
        # Turn list of Dicts → Dict of stacked arrays
        return {
            k: np.stack([o[k] for o in self._buf], axis=0)
            for k in self._buf[0]
        }

    def stack_dict_space(base_space: spaces.Dict, n_mem: int) -> spaces.Dict:
        """
        Build a new Dict space whose values are (n_mem, …) stacks of
        the arrays contained in `base_space`.
    
        * Box  →  Box with an extra leading axis  (low/high repeated)
        * Discrete →  Box[int] of shape (n_mem,) covering same integer range
        (Add more `elif` branches if your env ever returns other space types.)
        """
        assert isinstance(base_space, spaces.Dict), \
            f"Expected Dict, got {type(base_space)}"
    
        stacked = {}
    
        for key, space in base_space.spaces.items():
            if isinstance(space, spaces.Box):
                # Repeat low/high along a new first dimension
                low  = np.repeat(space.low[np.newaxis, ...],  n_mem, axis=0)
                high = np.repeat(space.high[np.newaxis, ...], n_mem, axis=0)
                stacked[key] = spaces.Box(low=low, high=high, dtype=space.dtype)
    
            elif isinstance(space, spaces.Discrete):
                # Convert to a 1-D integer Box so SB3 & friends are happy
                low  = np.full((n_mem,), space.start,               dtype=np.int32)
                high = np.full((n_mem,), space.start + space.n - 1, dtype=np.int32)
                stacked[key] = spaces.Box(low=low, high=high, dtype=np.int32)
    
            else:
                raise NotImplementedError(
                    f"Stacking not implemented for {type(space)} (key='{key}')")
    
        return spaces.Dict(stacked)




