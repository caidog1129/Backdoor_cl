class SimPolicy:
    def __init__(self, count=1):
        self.count = count

    def __call__(self, state, env):
        r = []
        for _ in range(self.count):
            s = state.copy()
            # print("STATE POL: ", s)
            is_initial_state_terminal = True
            while not env.is_terminal(state=s):
                is_initial_state_terminal = False
                actions = env.get_actions(state=s)
                idx = self.select(state=s, actions=actions)
                s += [actions[idx]]
                # print("STATE POL: ", s)

            r += [s]
            if is_initial_state_terminal:
                break

        return r

    def select(self, state, actions, action_probs=None):
        raise NotImplementedError