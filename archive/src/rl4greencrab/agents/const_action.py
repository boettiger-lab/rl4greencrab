import gymnasium as gym
import numpy as np

class constAction:
    def __init__(self, trap_type: "1, 2, or 3", action: float, env: gym.Env, **kwargs):
        self.env = env
        # self.action = action * np.ones(self.env.action_space.shape)
        self.action = np.zeros(self.env.action_space.shape)
        self.action[trap_type] = action

    def predict(self, observation, **kwargs):
        return self.action, {}

class multiConstAction:
    def __init__(self, action: np.ndarray, env: gym.Env, **kwargs):
        self.env = env
        # self.action = action * np.ones(self.env.action_space.shape)
        self.action = action

    def predict(self, observation, **kwargs):
        return self.action, {}

class constActionNatUnits:
    def __init__(self, ntraps=0, env = None, **kwargs):
        self.ntraps = ntraps
        self.action = ntraps
        self.env = env

    def predict(self, observation, **kwargs):
        return self.action, {}


        