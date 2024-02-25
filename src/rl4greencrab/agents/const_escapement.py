class constEsc:
    def __init__(self, escapement, env = None):
        self.escapement = escapement
        self.env = env
        self.bound = 1
        if self.env is not None:
            self.bound = self.env.bound

    def predict(self, observation):
        obs_nat_units = self.bound * self.to_01(observation)
        if obs_nat_units <= self.escapement or obs_nat_units =< 0:
            return -1
        mortality = (obs_nat_units - self.escapement) / self.escapement
        return self.to_pm1(mortality)   
    
    def to_01(self, val):
        return (val + 1 ) / 2

    def to_pm1(self, val):
        return 2 * val - 1


        