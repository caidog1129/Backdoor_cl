from sim_policies.base import SimPolicy

import numpy as np


class UniformSimPolicy(SimPolicy):

    def select(self, state, actions, action_probs=None):
        idx = np.random.randint(low=0, high=len(actions))
        return int(idx)