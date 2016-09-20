class Bot(object):
    """Base class for all bots."""

    def __init__(self, env):
        self.env = env
        self.state = None

    def reset(self, destination=None):
        pass

    def update(self, t):
        pass

    def get_state(self):
        return self.state