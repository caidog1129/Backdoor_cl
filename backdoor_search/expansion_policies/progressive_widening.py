from expansion_policies.base import ExpansionPolicy

import math


class PWExpansionPolicy(ExpansionPolicy):
    def __init__(self, expansion_type="uniform", widening_root=2):
        super().__init__(expansion_type=expansion_type, widening_root=widening_root)

    def is_expanded(self, node, env):
        # print("progwide: ",
        #       node.N, node.N_prev,
        #       self.num_children(node.N), self.num_children(node.N_prev),
        #       len(node.children), len(env.action_space), node.depth,
        #       self.num_children(node.N) == self.num_children(node.N_prev),
        #       len(node.children) >= len(env.action_space) - node.depth)
        return self.num_children(node.N) == self.num_children(node.N_prev) or \
               len(node.children) >= len(env.action_space) - node.depth

    def num_children(self, N):
        return math.floor(N ** (1.0 / self.widening_root)) if N >= 0 else -1