import cplex as cpx


class Solver:
    def __init__(self, limits_init=None):
        self.max_time_solver = \
            -1.0 if limits_init is None or 'max_time_solver' not in limits_init \
            else limits_init['max_time_solver']

        self.cpx_threads = limits_init['cpx_threads']

        dummy_cpx = cpx.Cplex()
        self.int_tol = dummy_cpx.parameters.mip.tolerances.integrality.get()

    def __call__(self, instance_cpx, backdoor_candidate, *args, **kwargs):
        raise NotImplementedError