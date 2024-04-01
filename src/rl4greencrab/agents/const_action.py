class constAction:
    def __init__(self, action=-1, env = None, **kwargs):
        self.action = action
        self.env = env

    def predict(self, observation, **kwargs):
        return self.action, {}

class constActionNatUnits:
    def __init__(self, ntraps=0, env = None, **kwargs):
        self.ntraps = ntraps
        self.action = ntraps
        self.env = env

    def predict(self, observation, **kwargs):
        return self.action, {}


        