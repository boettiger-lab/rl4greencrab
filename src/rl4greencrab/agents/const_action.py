class constAction:
    def __init__(self, mortality=0, env = None, **kwargs):
        self.mortality = mortality
        self.action = 2 * self.mortality - 1
        self.env = env

    def predict(self, observation, **kwargs):
        return self.action, {}


        