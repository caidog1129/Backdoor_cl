import numpy as np
from anytree import AnyNode


def new_child(actions_untried, expansion_type, scores=None):
    idx = -1
    if expansion_type == "uniform" or scores is None:
        idx = np.random.randint(low=0, high=len(actions_untried))
    elif expansion_type == "best_score" and scores is not None:
        idx = np.argmax(scores[actions_untried])
    return idx


class ExpansionPolicy:
    def __init__(self, expansion_type="uniform", widening_root=2):
        self.expansion_type = expansion_type
        self.widening_root = float(widening_root)

    def __call__(self, node, env, scores=None):
        actions = env.get_actions(state=node.state)
        actions_tried = [child.a for child in node.children]
        actions_untried = list(set(actions) - set(actions_tried))
        assert(len(actions_untried) > 0)

        idx = new_child(actions_untried=actions_untried, expansion_type=self.expansion_type, scores=scores)
        assert(idx != -1)

        node.N_prev = node.N
        action = actions_untried[idx]
        child = AnyNode(id='',
                        parent=node,
                        depth=node.depth + 1,
                        state=node.state + [action],
                        a=action,
                        Q=0,
                        N=0,
                        N_prev=-1,
                        best_child=None,
                        best_child_updated=1,
                        is_finished=False, num_finished_children=0,
                        rewards=[])

        return child

    def is_expanded(self, node, env):
        raise NotImplementedError