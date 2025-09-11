from itertools import product
import json
import os
import numpy as np
import polars as pl
from tqdm import tqdm
from .unit_interface import unitInterface

from rl4fisheries.agents.common import isVecObs

class PrecautionaryPolicy:
    def __init__(self, env, x1=0, x2=1, y2=1, **kwargs):
        self.env = env
        
        self.x1 = x1
        self.x2 = x2
        self.y2 = y2
    
    def predict(self, observation, **kwargs):
        crabs_obs = observation['crabs'][0]
        if crabs_obs < self.x1:
            return np.float32([-1])
        elif self.x1 <= crabs_obs <= self.x2:
            return np.float32([
                -1 + 2 * self.y1 * (crabs_obs - self.x1) / (self.x2 - self.x1)
            ])
        else:
            return np.float32([self.y1])



