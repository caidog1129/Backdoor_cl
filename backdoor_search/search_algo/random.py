import sys
import numpy as np
import time
from cplex.exceptions import CplexSolverError


class RandomSampler:
    def __init__(self, sim_cnt, env, strategy, seed):
        self.strategy = strategy
        self.sim_cnt = sim_cnt
        self.seed = seed

        self.p = None
        if self.strategy == 'biased':
            eps = env.solver.int_tol
            lpsol = np.array(env.root_lp['primal'].solution)[env.action_space]
            p = 0.5 - np.abs(0.5 - (lpsol - np.floor(lpsol)))
            p[p == 0] = eps
            self.p = p / np.sum(p)

        self.rng = np.random.default_rng(seed=self.seed)

    def __call__(self, env, cache_backdoors, max_backdoor):
        #TODO (Minor): Can/should we make max_attempts more accurate? Like P(env.action_space, max_backdoor) - len(cached_backdoors)), where P=Permutation
        new_backdoors = dict()
        max_attempts, attempt = 10000, 0

        while len(new_backdoors) != self.sim_cnt and attempt < max_attempts:
            # todo: for "random", env.action_space may be rstricted to root-LP fraction variables. This is not the case
                # in Dilkina et al.
            backdoor = self.rng.choice(a=env.action_space, size=max_backdoor, p=self.p, replace=False).tolist()
            backdoor_str = str(backdoor)

            if (backdoor_str not in cache_backdoors and
                backdoor_str not in new_backdoors):
                new_backdoors[backdoor_str] = backdoor
                attempt = 0
            else:
                attempt += 1

        return list(new_backdoors.values())

class RandomAlgorithm:

    def __init__(self, sampler):
        self.sampler = sampler

    def __call__(self, env, limits_init, max_backdoor, backdoor_file=None, backdoor_list=None):
        time_start = time.time()

        max_iter = limits_init['max_iter']
        max_time = limits_init['max_time']
        goodenough_reward = limits_init['goodenough_reward']  # 0.9
        patience_iter = limits_init['patience_iter']  # 500
        patience_zeroreward = limits_init['patience_zeroreward']

        best_backdoor = []
        best_reward = 0.0
        best_iter = -1
        best_iter_prev = -1
        best_simulationcounter = 0

        cache_backdoors = dict()
        reward_list = []
        simulation_counter = 0

        iteration = 0
        while iteration < max_iter and time.time() - time_start < max_time:
            sys.stdout.flush()
            print('------------------------------------------')
            print("iteration ", iteration)

            new_backdoors = self.sampler(env=env, cache_backdoors=cache_backdoors, max_backdoor=max_backdoor)
            if (len(new_backdoors) == 0):
                print(f"Tried all backdoors of size {max_backdoor}.")
                break

            try:
                rewards, new_solver_times = env.get_reward(new_backdoors)
                # rewards += new_rewards
                # rewards = env.get_reward(new_backdoors)

                if np.mean(new_solver_times) >= 0.9 * env.cpx_time:
                    env.cpx_time = 2 * env.cpx_time
                    print("UPDATING TIME to %g" % env.cpx_time)

            except CplexSolverError as e:
                print("CplexSolverError: ", str(e))
                continue

            reward_list += [np.max(rewards)]
            print("reward_list =", ', '.join(map(str, reward_list)))

            simulation_counter += len(rewards)
            found_new_incumbent = False

            for (backdoor, reward) in zip(new_backdoors, rewards):
                cache_backdoors[str(backdoor)] = reward

                print("reward = %g" % reward)
                print("----", backdoor)

                if backdoor_list is not None:
                    with open(backdoor_list, "a") as text_file:
                        print(str(reward) + ";" + str(backdoor), file=text_file)

                """ Found better (full) backdoor? """
                if reward > best_reward:
                    found_new_incumbent = True
                    best_reward = reward
                    best_backdoor = backdoor
                    best_iter_prev = best_iter
                    best_iter = len(reward_list) - 1
                    best_simulationcounter = simulation_counter

            if (found_new_incumbent):
                print("New Incumbent Backdoor!")

                # dump results to file...
                if backdoor_file is not None:
                    with open(backdoor_file, "a") as text_file:
                        print(iteration, time.time(), best_reward, len(best_backdoor), best_backdoor, file=text_file)

                # todo add OR condition for when best_reward is MUCH better than median rewards over at least T rewards
                """ Reward good enough? QUIT! """
                if best_reward >= goodenough_reward:
                    print("Reward good enough")
                    break

            """ Terminate early? """
            # early termination if no progress in last patience_iter simulations
            terminate_early = \
                (best_iter - best_iter_prev >= patience_iter) \
                or \
                (best_iter - best_iter_prev >= patience_zeroreward and best_reward == 0.0)
            if terminate_early:
                print("iteration %d: Terminate early, best_iter = %d, best_simulationcounter = %d, simulation_counter = %d, best_reward = %g"
                      % (iteration, best_iter, best_simulationcounter, simulation_counter, best_reward))
                break

            print("time elapsed = %g" % (time.time() - time_start))
            iteration += len(new_backdoors)

        return best_backdoor, best_iter, reward_list