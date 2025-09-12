from rl4greencrab import greenCrabMonthEnvSimple

import numpy as np

class LinearPolicy:
    """
    Policy for which action is linear function of observation.

    Assumes env generating the observation is of type: 
        rl4greencrab.greenCrabMonthEnvSimple

        ie. 
        
        observation = {
            'crabs': np.array([ N_crabs ])
        }
    """
    def __init__(
        self, 
        slope: np.float32, 
        **kwargs
    ):
        self.slope = slope
        

    def predict(self, observation, **kwargs):
        # action in [-1, -1 + slope]
        obs = observation['crabs'][0]
        action = -1 + self.slope * (obs + 1) / 2
        return np.float32([action]), {}