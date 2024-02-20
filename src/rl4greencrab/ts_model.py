import gymnasium as gym
import numpy as np

from gymnasium import spaces

class ts_env_v1(gym.Env):
	"""
	takes an environment env and produces an new environemtn ts_env
	whose observations are timeseries of the env environment.

	v1: the timeseries includes only past observations of base_env,
			not past actions.
	"""
	def __init__(self, config = {}):
		self.N_mem = config.get('N_mem', 5)
		if 'base_env_cls' not in config:
			raise Warning(
				"ts_env initializer needs to have a base environment "
				"out of whose dynamics the time-series will be built! "
				"Try: ts_env(config = {'base_env': <<your env>>, ...}). \n\n"
				"(Here, <<your env>> should be a class, not an instance!)"
				)
		if 'base_env_cfg' not in config:
			raise Warning(
				"ts_env initializer needs to have a base environment "
				"config!"
				"Try: ts_env(config = {'base_env_cfg': <<your config dict (possibly empty)>>, ...}). \n\n"
				)
		self.base_env_cls = config['base_env_cls']
		self.base_env_cfg = config['base_env_cfg']
		self.base_env = self.base_env_cls(config=self.base_env_cfg)

		self.action_space = self.base_env.action_space
		self.observation_space = spaces.Box(
			np.array(
				[ 
				- np.ones(shape=self.base_env.observation_space.shape, dtype=np.float32)
				for _ in range(self.N_mem)
				]
			),
			np.array(
				[ 
				+ np.ones(shape=self.base_env.observation_space.shape, dtype=np.float32)
				for _ in range(self.N_mem)
				]
			),
		)
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






