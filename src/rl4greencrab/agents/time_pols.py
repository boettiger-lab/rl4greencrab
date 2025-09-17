from rl4greencrab import greenCrabMonthEnvSizeNormalized
from typing import Annotated, Float, Int
import numpy as np

PositiveFloat = Annotated[Float, "Must be positive"]
TimeOfYear = Annotated[Float, "> 0 and <= 12"]
MonthOfYear = Annotated[Int, ">= 1 and <= 12"]
Phase = Annotated[Float, ">=0 and < pi"]


class TimeQuadPolicy:
    """
    Quadratic action as a function of month of year.

    Assumes env generating the observation is of type: 
        rl4greencrab.greenCrabMonthEnvSizeNormalized

        ie. 
        
        observation = {
            'crabs': np.array([ N_crabs ]),
            'months': N_month
        }
    """
    def __init__(
        self,
        # coefficient: PositiveFloat,
        offset: TimeOfYear,
        **kwargs,
    ):
        # self.coefficient = coefficient # -> factors out due to normalization!
        self.offset = offset

    def predict(self, observation, **kwargs):
        month = observation['months']
        return np.float32([
            -1 + 2 * ((month - self.offset)  ** 2) 
            / np.max( 
                # in order to normalize between -1 and 1
                [
                    (12 - self.offset) ** 2,
                    self.offset ** 2,
                ]
            )
        ])

class TimePulsePolicy:
    """
    Pulsed action one month of the year.

    Assumes env generating the observation is of type: 
        rl4greencrab.greenCrabMonthEnvSizeNormalized

        ie. 
        
        observation = {
            'crabs': np.array([ N_crabs ]),
            'months': N_month
        }
    """
    def __init__(
        self,
        pulse_height: PositiveFloat,
        pulse_month: MonthOfYear,
        **kwargs,
    ):
        self.pulse_height = pulse_height
        self.pulse_month = pulse_month

    def predict(self, observation, **kwargs): 
        month = observation['months']
        if month == pulse_month:
            return np.float32([
                -1 + self.pulse_height
            ])
        else:
            return np.float32([-1])


class TimeLinPolicy:
    """
    Linear action as a fn of month of the year.

    Assumes env generating the observation is of type: 
        rl4greencrab.greenCrabMonthEnvSizeNormalized

        ie. 
        
        observation = {
            'crabs': np.array([ N_crabs ]),
            'months': N_month
        }
    """
    def __init__(
        self,
        slope: Float,
        y_offset: PositiveFloat,
        **kwargs,
    ):
        self.slope = slope
        self.y_offset = y_offset

    def predict(self, observation, **kwargs): 
        month = observation['months']
        action = -1 + self.y_offset + self.slope * month / 12
        return np.float32([action])


class TimeSinPolicy:
    """
    Sinusoidal action as a fn of month of the year.

    Assumes env generating the observation is of type: 
        rl4greencrab.greenCrabMonthEnvSizeNormalized

        ie. 
        
        observation = {
            'crabs': np.array([ N_crabs ]),
            'months': N_month
        }
    """
    def __init__(
        self,
        sine_height: PositiveFloat,
        phase: Phase,
        **kwargs,
    ):
        self.sine_height = sine_height
        self.phase = phase

    def predict(self, observation, **kwargs): 
        month = observation['months']
        sine_sq = self.sine_height * np.sin(
            self.phase + np.pi * month / 12
        ) ** 2
        return np.float32([
            -1 + sine_sq
        ])



        