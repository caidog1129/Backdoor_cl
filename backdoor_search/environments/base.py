import utils
from mip_utils.solution import Solution

import numpy as np
import cplex as cpx
import os
import subprocess
import time


class Env:
    def __init__(self, *,
                 instance_path,
                 solver_maker,
                 reward_function,
                 max_backdoor,
                 cpx_time,
                 seed,
                 cpx_variableselect=-1,
                 presolve_flag=False,
                 presolved_dir_path=None,
                 presolved_overwrite=False,
                 initializeopt_flag=True,
                 initializeopt_dir_path="./SOL/",
                 exclude_root_int=False,
                 get_root_lp=False,
                 root_lp=None,
                 reuse_instance=False,
                 cpx_threads=1, cpx_mem=1000, cpx_display=False, sol_file="",
                 cpx_heuristicfreq=0, ignore_ordering=False):

        self.instance_name = ""
        self.instance_path = instance_path
        self.solver_maker = solver_maker
        self.solver = solver_maker()
        self.reward_function = reward_function
        self.max_backdoor = max_backdoor
        self.cpx_variableselect = cpx_variableselect
        self.initializeopt_flag = initializeopt_flag
        self.initializeopt_dir_path = initializeopt_dir_path
        self.reuse_instance = reuse_instance
        self.cpx_threads = cpx_threads
        self.cpx_mem = cpx_mem
        self.cpx_time = cpx_time
        self.cpx_display = cpx_display
        self.cpx_heuristicfreq = cpx_heuristicfreq
        self.sol_file = sol_file

        self.instance_cpx = cpx.Cplex()
        self.instance_cpx.parameters.randomseed.set(seed)
        if presolve_flag:
            self.set_instance_path(presolved_dir_path=presolved_dir_path, presolved_overwrite=presolved_overwrite)

        self.root_lp = root_lp
        self.solver.root_lp = root_lp

        self.ignore_ordering = ignore_ordering
        self.solver.ignore_ordering = ignore_ordering

        self.int_vars = None
        self.reset(disable_output=not self.cpx_display, get_root_lp=get_root_lp, initializeopt_flag=initializeopt_flag)
        if self.reuse_instance and get_root_lp:
            self.reset(disable_output=not self.cpx_display, get_root_lp=False, initializeopt_flag=initializeopt_flag)

        self.num_vars = self.instance_cpx.variables.get_num()
        self.action_space = self.build_actions(exclude_root_int=exclude_root_int)
        self.max_backdoor = min([self.max_backdoor, len(self.action_space)])

    def build_actions(self, exclude_root_int):
        assert (self.int_vars is not None)
        action_space = []
        for int_var in self.int_vars:
            if exclude_root_int and \
                    abs(self.root_lp['primal'].solution[int_var] - np.round(
                        self.root_lp['primal'].solution[int_var])) < self.solver.int_tol:
                continue
            action_space += [int_var]

        return action_space

    def set_instance_path(self, presolved_dir_path, presolved_overwrite):
        assert (presolved_dir_path is not None)
        instance_name = os.path.basename(self.instance_path)
        instance_name = instance_name[:instance_name.find('.')]
        self.instance_name = instance_name
        presolved_instance_path = "%s/%s_proc.mps" % (presolved_dir_path, instance_name)
        if presolved_overwrite or not os.path.exists(presolved_instance_path):
            cmd = './get_cuts %s %s' % (self.instance_path, presolved_instance_path)
            subprocess.call(cmd, shell=True)
        self.instance_path = presolved_instance_path

    def reset(self, disable_output=True, get_root_lp=False, initializeopt_flag=True):
        """ disable all cplex output """
        if disable_output:
            utils.disable_output_cpx(self.instance_cpx)

        self.instance_cpx.read(self.instance_path)

        self.instance_cpx.parameters.threads.set(self.cpx_threads)
        self.instance_cpx.parameters.workmem.set(self.cpx_mem)
        var_types = self.instance_cpx.variables.get_types()
        self.int_vars = np.sort(np.concatenate((np.where(np.array(var_types) == 'B')[0],
                                                np.where(np.array(var_types) == 'I')[0]))).tolist()

        if get_root_lp:
            """ get root lp if needed """

            continuous_type = self.instance_cpx.variables.type.continuous
            idx_type_tuples = [(idx, continuous_type) for idx in self.int_vars]
            self.instance_cpx.variables.set_types(idx_type_tuples)
            self.instance_cpx.set_problem_type(self.instance_cpx.problem_type.LP)

            self.instance_cpx.solve()
            root_primal = Solution(self.instance_cpx.solution.get_values(),
                                   self.instance_cpx.solution.get_objective_value())
            root_dual = Solution(self.instance_cpx.solution.get_dual_values(),
                                 self.instance_cpx.solution.get_objective_value())
            self.root_lp = {'primal': root_primal, 'dual': root_dual}
            self.solver.root_lp = self.root_lp

        else:
            self.instance_cpx.parameters.preprocessing.presolve.set(0)
            self.instance_cpx.parameters.mip.strategy.variableselect.set(self.cpx_variableselect)
            self.instance_cpx.parameters.mip.limits.cutpasses.set(-1)
            self.instance_cpx.parameters.mip.strategy.heuristicfreq.set(self.cpx_heuristicfreq)

            # if self.solver.max_time_solver >= 0:
            #     self.instance_cpx.parameters.timelimit.set(self.cpx_time)

            if initializeopt_flag:
                optsol_path = self.initializeopt_dir_path  # "%s/%s.sol" % (initializeopt_dir_path, self.instance_name)
                if optsol_path is not None and os.path.exists(optsol_path):
                    self.instance_cpx.MIP_starts.read(optsol_path)

    def is_terminal(self, state):
        return len(state) == self.max_backdoor

    def get_actions(self, state):
        return list(set(self.action_space) - set(state))

    def get_pseudocosts(self):
        return [self.solver.pseudocosts]

    def get_pseudocosts_varset(self):
        return [self.solver.pseudocosts_varset]

    def get_branched_on(self):
        return [self.solver.branched_on]

    def get_reward(self, state, max_lp_sols=-1):
        # https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.10.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/AdvInd.html
        if self.reuse_instance:
            self.instance_cpx.parameters.advance.set(2)
        else:
            self.reset(initializeopt_flag=self.initializeopt_flag, disable_output=not self.cpx_display)

        self.instance_cpx.parameters.timelimit.set(self.cpx_time)

        solver_time = time.time()
        best_bound, num_nodes_remaining, tree_weight = self.solver(self.instance_cpx, state[0], max_lp_sols)
        solver_time = time.time() - solver_time

        # set tree_weight to 0 if solver timed out: https://www.ibm.com/support/knowledgecenter/SSSA5P_20.1.0/ilog.odms.cplex.help/refcallablelibrary/macros/Solution_status_codes.html
        status = self.instance_cpx.solution.get_status()
        tree_weight = 0.0 if status >= 104 else tree_weight

        # todo best_bound should be in a dictionary and passed to reward_function as kwargs
        ret = self.reward_function(self.instance_cpx,
                                   best_bound=best_bound,
                                   num_nodes_remaining=num_nodes_remaining,
                                   tree_weight=tree_weight)

        # if self.solver.incumbent_updated:
        #     sol_file = '%s_%.3f.sol' % (self.sol_file, self.solver.incumbent.objective)
        #     self.instance_cpx.solution.write(sol_file)

        return [ret], [solver_time]