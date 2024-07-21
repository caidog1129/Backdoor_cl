class TreePolicy:
    def __init__(self, expansion_policy, track_finished):
        self.expansion_policy = expansion_policy
        self.track_finished = track_finished

    def __call__(self, node, env, rave_scores=None):
        v = node
        # todo: if a state is_terminal, we shouldn't even arrive there at all!
        # todo: if a subtree is fully expanded, we shouldn't go there!
        while not env.is_terminal(state=v.state):
            prev_state = v.state
            expand_bool = not self.expansion_policy.is_expanded(node=v, env=env)
            expand_bool = expand_bool or (self.track_finished and v.depth == env.max_backdoor-1)
            # v.N_prev = v.N
            if expand_bool:
                # v.N_prev = v.N
                return self.expansion_policy(node=v, env=env, scores=rave_scores)
            else:
                # len_v_children = len(v.children)
                # v_num_finished_children = v.num_finished_children
                # v_state = v.state
                num_children_active = len(v.children) - self.track_finished * v.num_finished_children
                # if len(v.children) > 0:
                if num_children_active > 0:
                    v_prev = v
                    v = self.next_child(node=v, rave_scores=rave_scores)
                    # if v is None:
                    #     from anytree import RenderTree
                    #     print(v_state, num_children_active, len_v_children, v_num_finished_children)
                    #     print(RenderTree(v_prev))
                else:
                    return v

        return v

    def next_child(self, node, rave_scores):
        raise NotImplementedError