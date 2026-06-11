import numpy as np
import pandas as pd


def find_closest_actions(CPUE, biomass, month, centroids):
    results = {}
    for action in ['act0', 'act1']:
        subset = centroids[(centroids['month'] == month) & (centroids['action'] == action)].copy()
        subset['dist'] = ((subset['CPUE'] - CPUE)**2 + (subset['biomass'] - biomass)**2)**0.5
        closest = subset.loc[subset['dist'].idxmin()]
        results[action] = closest[action]
    return results['act0'], results['act1']

# example usage:
# act0, act1 = find_closest_actions(CPUE=-0.97, biomass=-0.5, month=5, centroids=centroids)

class CentroidAgent:
    """Agent that selects actions by finding the nearest centroid for the current observation.

    Expects observations of the form {"crabs": np.array([CPUE, biomass]), "months": month},
    i.e. observation_type='count-biomass-time'.
    """
    def __init__(self, centroids: pd.DataFrame, env):
        self.centroids = centroids
        self.env = env

    def predict(self, observation, **kwargs):
        CPUE = float(observation["crabs"][0])
        biomass = float(observation["crabs"][1])
        month = int(observation["months"])
        act0, act1 = find_closest_actions(CPUE, biomass, month, self.centroids)
        return np.array([act0, act1], dtype=np.float32), {}