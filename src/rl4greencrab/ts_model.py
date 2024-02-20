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


class ts_env_v2(gym.Env):
	"""
	UNDER CONSTRUCTION - BROKEN

	takes an environment env and produces an new environemtn ts_env
	whose observations are timeseries of the env environment.

	v2: the timeseries includes past observations AND past actions
	"""
	def __init__(self, config = {}):
		self.N_mem = config.get('N_mem', 5)
		if 'base_env' not in config:
			raise Warning(
				"ts_env initializer needs to have a base environment "
				"out of whose dynamics the time-series will be built! "
				"Try: ts_env(config = {'base_env': <<your env>>, ...})."
				)
		self.base_env = config['base_env']

		self.action_space = spaces.Box(
			np.float32([-1]),
			np.float32([+1]),
			)
		self.observation_space = spaces.Box(
			np.float32([ [-1] * self.N_mem ] * 2),
			np.float32([ [+1] * self.N_mem ] * 2),
			)
		#
		# [[obs  t, obs t-1, obs t-2, ..., obs t - (N_mem-1)]
		#  [act  t, act t-1, act t-2, ..., act t - (N_mem-1)]]
		#
		# here 'obs t' is the observation at the end of last timestep
		# and 'act t' is the action taken during the last timestep
		#
		_ = self.reset()

	def reset(self, *, seed=42, options=None):
		init_state, init_info = self.base_env.reset(seed=seed, options=options)
		empty_heap = np.float32([-1] * self.N_heap)
		obs_heap = np.insert(
			empty_heap[0:-1],
			0,
			init_state,
		)
		act_heap = empty_heap.copy()
		self.heap = np.array(
			[obs_heap, act_heap]
		)
		return self.heap, init_info

	def step(self, action):
		new_state, reward, terminated, truncated, info = self.base_env.step(action)

		# update the heap timeseries
		[obs_heap, act_heap] = self.heap
		obs_heap = np.insert(
			obs_heap[0:-1],
			0,
			new_state,
		)
		act_heap = np.insert(
			act_heap[0:-1],
			0,
			action,
		)
		self.heap = np.float32([obs_heap, act_heap])

		return self.heap, reward, terminated, truncated, info

	def pop_to_state(self, pop):
		return self.base_env.pop_to_state(pop)

	def state_to_pop(self, state):
		return self.base_env.state_to_pop(state)



