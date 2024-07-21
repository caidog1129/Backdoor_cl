from environments.base import Env
from mip_utils.solution import Solution

import ray
import os
import psutil
from ray.util import ActorPool
import time
import numpy as np
import cplex as cpx


@ray.remote
class EnvWorker:
    def __init__(self,
                 **kwargs):
        self.env = Env(**kwargs)

    def getattr(self, attr):
        return getattr(self.env, attr)

    def max_backdoor(self, max_backdoor):
        self.env.max_backdoor = max_backdoor

    def cpx_time(self, cpx_time):
        self.env.cpx_time = cpx_time

    def work(self, incumbent, state_list, write_incumbent_bool=False, max_lp_sols=-1):
        process = psutil.Process(os.getpid())
        last_times = process.cpu_times()

        if (self.env.reuse_instance and incumbent.solution): # a new incumbent is sent from the master env
            self.env.solver.incumbent = incumbent
            if write_incumbent_bool and len(self.env.sol_file) > 0:
                self.env.instance_cpx.MIP_starts.add(
                    [list(range(len(incumbent.solution))), incumbent.solution],
                    self.env.instance_cpx.MIP_starts.effort_level.auto)

        reward, solver_time = self.env.get_reward([state_list], max_lp_sols)
        reward, solver_time = reward[0], solver_time[0]
        pc = self.env.solver.pseudocosts
        pcvs = self.env.solver.pseudocosts_varset
        branched_on = self.env.solver.branched_on
        incumbent = self.env.solver.incumbent
        sense = self.env.instance_cpx.objective.sense[self.env.instance_cpx.objective.get_sense()]

        # if write_incumbent_bool and len(self.env.sol_file) > 0:
        #     self.env.instance_cpx.solution.write(self.env.sol_file)

        times = process.cpu_times()
        usage = sum(times) - sum(last_times)
        return {"reward": reward,
                "solver_time": solver_time,
                "pc": pc,
                "pcvs": pcvs,
                "branched_on": branched_on,
                "incumbent": incumbent,
                "sense": sense,
                "time": usage}


class ParallelEnv:
    def __init__(self, *,
                 num_envs,
                 **kwargs):
            ray.init(include_dashboard=False)

            self._max_backdoor = kwargs["max_backdoor"]
            self._cpx_time = kwargs["cpx_time"]

            self.incumbent = Solution()
            self.pseudocosts = []
            self.pseudocosts_varset = set()
            self.branched_on = []

            self.incumbent_updated_bool = False
            self.best_reward_global = 0

            # Do root LP computation here once and send it to all solvers.
            if kwargs["get_root_lp"]:
                cpx_threads_true = kwargs["cpx_threads"]
                kwargs["cpx_threads"] = num_envs
                temp_worker = Env(**kwargs)
                kwargs["root_lp"] = temp_worker.root_lp
                kwargs["get_root_lp"] = False
                kwargs["cpx_threads"] = cpx_threads_true

            self.worker_envs = [EnvWorker.remote(**kwargs) for _ in range(num_envs)]

    def __getattr__(self, name):
        return ray.get(self.worker_envs[0].getattr.remote(name))

    @property
    def max_backdoor(self):
        return self._max_backdoor

    @max_backdoor.setter
    def max_backdoor(self, max_backdoor):
        self._max_backdoor = max_backdoor
        ray.get([env.max_backdoor.remote(max_backdoor) for env in self.worker_envs])

    @property
    def cpx_time(self):
        return self._cpx_time

    @cpx_time.setter
    def cpx_time(self, cpx_time):
        self._cpx_time = cpx_time
        ray.get([env.cpx_time.remote(cpx_time) for env in self.worker_envs])

    def is_terminal(self, state):
        return len(state) == self._max_backdoor

    def get_actions(self, state):
        return list(set(self.action_space) - set(state))

    def get_pseudocosts(self):
        return self.pseudocosts

    def get_pseudocosts_varset(self):
        return self.pseudocosts_varset

    def get_branched_on(self):
        return self.branched_on

    def get_reward(self, state_list, max_lp_sols=-1):
        assert(len(state_list) > 0)

        w_time = time.time()

        pool = ActorPool(self.worker_envs)

        state_incumbent_list = [(state_list[i], self.incumbent_updated_bool and i == 0) for i in range(len(state_list))]
        results = list(pool.map(
            lambda env, state: env.work.remote(self.incumbent, state[0], state[1], max_lp_sols), state_incumbent_list))
        rewards, solver_times, pcs, pcvss, branched_on, times = [], [], [], [], [], []

        self.incumbent_updated_bool = False
        for i, r in enumerate(results):
            #print("r[reward] = ", r["reward"])
            rewards.append(r["reward"])
            solver_times.append(r["solver_time"])
            pcs.append(r["pc"])
            pcvss.append(r["pcvs"])
            branched_on.append(r["branched_on"])
            times.append(r["time"])

            # Check if any of the workers incumbents is better than ours
            sense = r["sense"]
            worker_incumbent = r["incumbent"]
            obj_val = worker_incumbent.objective
            if (obj_val): # if the worker incumbent is NOT primal feasible, then its incumbent would be Solution() which has a None objective
                if (self.incumbent.solution is None) \
                        or (sense == "minimize" and obj_val < self.incumbent.objective) \
                        or (sense == "maximize" and obj_val > self.incumbent.objective):
                    self.incumbent.objective = obj_val
                    self.incumbent.solution = worker_incumbent.solution
                    self.incumbent_updated_bool = True

        tag = os.path.basename(self.instance_path)
        # if self.incumbent.objective is not None:
        #     self.s_writer.add_scalar(f"{tag}/incumb_obj", self.incumbent.objective)

        self.pseudocosts = pcs
        self.pseudocosts_varset = pcvss

        self.best_reward_global = max([self.best_reward_global, np.max(rewards)])

        # self.s_writer.add_scalar(f"{tag}/median_reward_local", np.median(rewards))
        # self.s_writer.add_scalar(f"{tag}/best_reward_local", np.max(rewards))
        # self.s_writer.add_scalar(f"{tag}/best_reward_global2", self.best_reward_global)

        print(f"[get_reward()] Workers  (CPU-time): {np.mean(times):.2f}s/{np.max(times):.2f}s/{np.sum(times):.2f}s (mean/max/sum)")
        print(f"               Master (wall-clock): {(time.time() - w_time):.2f}s")
        return rewards, solver_times