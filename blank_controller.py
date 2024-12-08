import numpy as np

class BlankController:
    def __init__(self):
        self.use_for_training = False
        pass

    def initialize_controller(self, env):
        pass

    def reset_state(self, initial_state, opp_state):
        pass
    
    def make_step(self, timestamp, initial_state):
        return np.array([[0.0], [0.0]])
