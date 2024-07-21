from tree_policies.base import TreePolicy

import numpy as np


class UCTreePolicy(TreePolicy):
    def __init__(self, expansion_policy, track_finished, c=1.0/np.sqrt(2), variance=True, rave_weight=0.1):
        super().__init__(expansion_policy=expansion_policy, track_finished=track_finished)

        self.c = c
        self.variance = variance
        self.rave_weight = rave_weight

    def next_child(self, node, rave_scores):
        best_node = None
        best_score = 0
        for child in node.children:
            if child.is_finished:
                continue
            sqrt_term = (np.log(node.N) / child.N)
            if self.variance:
                sqrt_term = np.sqrt(sqrt_term * np.min([0.25, np.var(node.rewards) + np.sqrt(2*sqrt_term)]))
            else:
                sqrt_term = np.sqrt(sqrt_term)

            # todo: make Q backdoor-size dependent, Q[10] is the avg. reward of size-10 backdoor simulations
            if rave_scores is None:
                score = child.Q + self.c * sqrt_term
            else:
                score = (1-self.rave_weight) * (child.Q + self.c * sqrt_term) + self.rave_weight * rave_scores[child.a]

            # if len(node.state) == 0:
            #     print("score =", score, "no_rave =", child.Q + self.c * sqrt_term, "self.c * sqrt_term =", self.c * sqrt_term, "rave_score =", rave_scores[child.a])
            best_node, best_score = (child, score) if ((score > best_score) or (best_node is None)) else (best_node, best_score)

        return best_node