import utils


class BackupPolicy:
    def __init__(self, best_child_criterion, track_finished):
        self.best_child_criterion = best_child_criterion
        self.track_finished = track_finished

    def __call__(self, node, reward, max_backdoor, num_actions):
        finished_children_increment = 0
        while node is not None:
            # node.N_prev = node.N
            update_stats(node=node, reward=reward)
            self.propagate(node=node, reward=reward)

            """ Update best child for current node """
            utils.set_best_child(node=node, best_child_criterion=self.best_child_criterion)

            """ tag node if it is fully expanded """
            if self.track_finished:
                node.num_finished_children += finished_children_increment
                finished_children_increment = 0
                if (node.num_finished_children == num_actions - node.depth) or (node.depth == max_backdoor):
                    print("finished_children_increment", node.state)
                    finished_children_increment = 1
                    node.is_finished = True

            node = node.parent

    def propagate(self, node, reward):
        raise NotImplementedError


def update_stats(node, reward):
    node.N += 1
    node.rewards += [reward]