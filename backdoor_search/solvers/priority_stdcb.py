from solvers.base import Solver
from mip_utils.solution import Solution

from cplex.callbacks import BranchCallback, SolveCallback, NodeCallback
import math
import numpy as np
from sortedcontainers import SortedList


def set_callback_data(callback, backdoor_candidate, int_tol, max_lp_sols=-1):
    callback.int_tol = int_tol
    callback.backdoor_candidate = backdoor_candidate
    callback.backdoor_candidate_priorities = {var: len(backdoor_candidate) - idx for idx, var in
                                              enumerate(backdoor_candidate)}
    callback.calls = 0  # How often was the callback invoked?
    callback.branches = 0  # How many branches did the callback create?
    callback.pruned = 0

    callback.best_bound = math.inf
    callback.tree_weight = 1.0

    callback.lp_sol_store = SortedList()
    callback.max_lp_sols = max_lp_sols

    callback.pseudocosts = []
    callback.pseudocosts_varset = set()

    callback.branched_on = set()

    callback.forbidden_nodes = set()


class PrioritySolverStandard(Solver):
    def __init__(self, limits_init=None, warmstart_root=False):
        super().__init__(limits_init=limits_init)
        self.root_lp = None
        self.incumbent = Solution()
        self.incumbent_updated = False
        self.lp_sol_store_last = None

        self.warmstart_root = warmstart_root

    def __call__(self, instance, backdoor_candidate, max_lp_sols=-1):
        self.incumbent_updated = False
        order_tuples = []
        max_priority = len(backdoor_candidate)
        if max_lp_sols > -2:
            for idx, var in enumerate(backdoor_candidate):
                cur_priority = max_priority if self.ignore_ordering or max_lp_sols != -1 else max_priority - idx
                order_tuples += [(var, cur_priority, instance.order.branch_direction.up)]
                # order_tuples += [(var, max_priority - idx, instance.order.branch_direction.up)]
                # order_tuples += [(var, max_priority, instance.order.branch_direction.up)]

            if max_priority > 0:
                instance.order.set(order_tuples)

        # todo I think this is true: We don't need to add the incumbent if reuse_instance=True
        # if self.incumbent.solution is not None:
        #     instance.MIP_starts.add(
        #         [list(range(len(self.incumbent.solution))), self.incumbent.solution],
        #         instance.MIP_starts.effort_level.auto)

        cb = instance.register_callback(BranchCallbackStandard)
        set_callback_data(cb,
                          backdoor_candidate=backdoor_candidate,
                          int_tol=self.int_tol,
                          max_lp_sols=max_lp_sols)

        if self.warmstart_root:
            cb_solve = instance.register_callback(SolveCallbackStandard)
            cb_solve.root_lp = self.root_lp

            heuristicfreq_val = instance.parameters.mip.strategy.heuristicfreq.get()
            instance.parameters.mip.limits.nodes.set(1)
            instance.parameters.mip.strategy.heuristicfreq.set(-1)

            instance.solve()

            instance.parameters.mip.limits.nodes.set(1.0e9)
            instance.parameters.mip.strategy.heuristicfreq.set(heuristicfreq_val)

            instance.parameters.advance.set(1)

        if max_lp_sols <= -2:
            cb_nodesel = instance.register_callback(NodeCallbackStandard)
            cb_nodesel.forbidden_nodes = cb.forbidden_nodes
            cb_nodesel.max_lp_sols = max_lp_sols
            # cb_nodesel.num_nodes_processed = 0

        instance.solve()

        if max_lp_sols <= -2:
            self.num_nodes_processed = cb_nodesel.num_nodes_processed
            instance.unregister_callback(BranchCallbackStandard)
            instance.unregister_callback(NodeCallbackStandard)
            return

        if instance.solution.is_primal_feasible():
            obj_val = instance.solution.get_objective_value()
            # print("obj_val=", obj_val)
            sense = instance.objective.sense[instance.objective.get_sense()]
            if (self.incumbent.solution is None) \
                    or (sense == "minimize" and obj_val < self.incumbent.objective) \
                    or (sense == "maximize" and obj_val > self.incumbent.objective):
                self.incumbent.objective = obj_val
                self.incumbent.solution = instance.solution.get_values()
                self.incumbent_updated = True

        self.lp_sol_store_last = cb.lp_sol_store

        self.pseudocosts = cb.pseudocosts
        self.pseudocosts_varset = cb.pseudocosts_varset
        self.branched_on = cb.branched_on

        return cb.best_bound, cb.pruned, cb.tree_weight


class SolveCallbackStandard(SolveCallback):

    def __call__(self):
        if self.get_num_nodes() == 0 and self.root_lp is not None:
            primal = (list(range(len(self.root_lp['primal'].solution))), self.root_lp['primal'].solution)
            dual = (list(range(len(self.root_lp['dual'].solution))), self.root_lp['dual'].solution)
            self.set_start(primal=primal, dual=dual)
            self.solve()  # (self.method.dual)
            self.use_solution()


class BranchCallbackStandard(BranchCallback):

    def __call__(self):
        if self.get_num_branches() == 0:
            return

        self.calls += 1

        # Get the objective value of the current relaxation. We use this as estimate for the new children to create.
        self.best_bound = self.get_best_objective_value()
        lpsol = self.get_values()
        obj = self.get_objective_value()
        self.pseudocosts = self.get_pseudo_costs()

        node_count = self.get_num_nodes()
        # if node_count == 0:
        # self.root_lp_primal = Solution(lpsol, obj)

        best_var = -1

        _, var_info = self.get_branch(0)
        branching_var_idx = var_info[0][0]
        best_var = branching_var_idx if branching_var_idx in self.backdoor_candidate else -1
        if best_var != -1:
            self.branched_on.add(best_var)
        # note: is this check needed? why not query CPLEX's chosen variable and check if it's in backdoor?
        # lbs = self.get_lower_bounds()
        # ubs = self.get_upper_bounds()
        # for idx, var in enumerate(self.backdoor_candidate):
        #     if lbs[var] == ubs[var] or abs(lpsol[var] - np.round(lpsol[var])) < self.int_tol:
        #         continue
        #     best_var = var
        #     break

        depth = self.get_current_node_depth()

        if best_var == -1:
            # print("prune!")
            self.pruned += 1

            if self.max_lp_sols == -1:
                # todo: extend tree weight to integer variables
                self.tree_weight -= (2 ** -depth)
                self.prune()
                return
            elif self.max_lp_sols <= -2:
                for branch_idx in range(self.get_num_branches()):
                    node_seqnum = self.make_cplex_branch(branch_idx)
                    self.forbidden_nodes.add(node_seqnum)
                    # print("self.forbidden_nodes = ", self.forbidden_nodes)
            else:
                self.pseudocosts_varset.add(branching_var_idx)
                if len(self.lp_sol_store) < self.max_lp_sols:
                    self.lp_sol_store.add(Solution(lpsol, obj))
                    if len(self.lp_sol_store) == self.max_lp_sols:
                        self.abort()
                        print("ABORTING!!")
                        return
                else:
                    self.tree_weight -= (2 ** -depth)
                    self.prune()
                    return
                    # self.abort()

        self.branches += 1


class NodeCallbackStandard(NodeCallback):

    def __call__(self):
        if self.get_num_nodes() == 0:
            return

        best_node = None
        best_bound = math.inf
        for node_idx in range(self.get_num_remaining_nodes()):
            node_seqnum = self.get_node_ID(node_idx)[0]
            # print("node_seqnum = ", node_seqnum)
            # print("self.forbidden_nodes2", self.forbidden_nodes)
            if node_seqnum not in self.forbidden_nodes:
                node_idx = self.get_node_number((node_seqnum,))
                node_bound = self.get_objective_value(node_idx)
                if node_bound < best_bound:
                    best_node = node_seqnum
                    best_bound = node_bound

        if best_node is None:
            self.num_nodes_processed = self.get_num_nodes()
            if self.max_lp_sols == -2:
                self.abort()
            return

        node_idx = self.get_node_number((best_node,))
        self.select_node(node_idx)