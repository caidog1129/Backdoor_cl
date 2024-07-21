import utils

from anytree import RenderTree, AnyNode

import sys
import numpy as np
import time
from cplex.exceptions import CplexSolverError


class MCTSAlgorithm:

    def __init__(self,
                 sim_policy,
                 tree_policy,
                 backup_policy):
        self.sim_policy = sim_policy
        self.tree_policy = tree_policy
        self.backup_policy = backup_policy

    def __call__(self, root, env, limits_init, max_backdoor,
                 dig_deeper=True, backdoor_file = None, backdoor_list=None):
        time_start = time.time()

        max_iter = limits_init['max_iter']
        max_time = limits_init['max_time']

        goodenough_reward = limits_init['goodenough_reward']  # 0.9
        patience_iter = limits_init['patience_iter']  # 500
        patience_deeper = limits_init['patience_deeper']  # 100
        patience_zeroreward = limits_init['patience_zeroreward']

        cur_root = root

        best_backdoor = []
        best_reward = 0.0
        best_iter = -1
        best_simulationcounter = 0

        cache_backdoors = dict()
        reward_list = []
        simulation_counter = 0

        pseudocosts_avg = np.array(root.pseudocosts_avg)
        pseudocosts_count = np.array(root.pseudocosts_count)

        dig_iter = 0
        iteration = 0
        while iteration < max_iter and time.time() - time_start < max_time \
                and cur_root.depth < max_backdoor:
            sys.stdout.flush()
            print('------------------------------------------')
            print("iteration ", iteration)

            pseudocosts_avg_max = np.max(pseudocosts_avg[env.int_vars])
            pseudocosts_avg_scaled = pseudocosts_avg / pseudocosts_avg_max
            v = self.tree_policy(node=cur_root, env=env, rave_scores=pseudocosts_avg_scaled)

            if v.id == "root":
                print(RenderTree(root))
                print("Tree is complete, aborting")
                break

            print("v.state = ", v.state)

            # todo: deal with repeated choices
            rewards, new_leafs = [], []
            sims = self.sim_policy(state=v.state.copy(), env=env)
            # remove duplicate sims
            sims = [list(x) for x in set(tuple(x) for x in sims)]
            for state_leaf in sims:
                if env.ignore_ordering:
                    state_leaf.sort()
                if str(state_leaf) in cache_backdoors:
                    # rewards += [cache_backdoors[str(state_leaf)]]
                    rewards += [cache_backdoors[str(state_leaf)]]
                else:
                    if len(new_leafs) == 0:
                        print("new leafs: ", state_leaf)
                    else:
                        print("           ", state_leaf)

                    new_leafs += [state_leaf]

            try:
                if len(new_leafs) != 0:
                    new_rewards, new_solver_times = env.get_reward(new_leafs)
                    rewards += new_rewards

                    for i, pseudocosts in enumerate(env.get_pseudocosts()):
                        if len(pseudocosts) > 0:
                            utils.update_pseudocosts(pseudocosts, pseudocosts_avg, pseudocosts_count, new_leafs[i])
                    root.pseudocosts_avg, root.pseudocosts_count = pseudocosts_avg.tolist(), pseudocosts_count.tolist()

                    for i, branched_on in enumerate(env.get_branched_on()):
                        if len(branched_on) != len(new_leafs[i]):
                            print("did not branch on all backdoor variables!", new_leafs[i], branched_on)
                            new_leafs[i] = [var for var in branched_on]

                    if np.mean(new_solver_times) >= 0.9 * env.cpx_time:
                        env.cpx_time = 2 * env.cpx_time
                        print("UPDATING TIME to %g" % env.cpx_time)

            except CplexSolverError:
                # if v is a newly expanded node + CPLEX failed, delete v
                print("Exception")
                print(v)
                if v.N == 0:
                    print("v.N == 0")
                    print(RenderTree(root))
                    v.parent = None
                    print(RenderTree(root))
                continue

            print(f"rewards (mean: {np.mean(rewards)}) =", ', '.join(map(str, rewards)))
            # TODO: Shall we still backup the similar traces?
            for reward in rewards:
                print("reward = ", reward)
                self.backup_policy(node=v, reward=reward, max_backdoor=max_backdoor, num_actions=len(env.action_space))

            reward_list += [np.max(rewards)]

            good_reward = np.max(rewards)

            if iteration % 100 == 0:
                # print(RenderTree(root))
                # if tree_file is not None:
                #     utils.write_tree(root, tree_file)

                print("reward_list =", ', '.join(map(str, reward_list)))

            simulation_counter += len(rewards)
            found_new_incumbent = False
            for (state_leaf, reward) in zip(new_leafs, new_rewards):
                if reward == good_reward:
                    if backdoor_list is not None:
                        with open(backdoor_list, "a") as text_file:
                            print(str(reward) + ";" + str(state_leaf), file=text_file)
                # cache_backdoors[str(state_leaf)] = reward
                cache_backdoors[str(state_leaf)] = reward

                """ Found better (full) backdoor? """
                # take a max of rewards from processes
                if reward > best_reward:
                    found_new_incumbent = True
                    best_reward = reward
                    best_backdoor = state_leaf
                    best_iter = iteration
                    best_simulationcounter = simulation_counter

            if (found_new_incumbent):
                print("New Incumbent Backdoor!")
                # dump results to file...
                # todo should consider get_variable_names(best_backdoor) and storing that instead of just indices
                if backdoor_file is not None:
                    with open(backdoor_file, "a") as text_file:
                        print(iteration, time.time(), best_reward, len(best_backdoor), best_backdoor, file=text_file)

                """ Reward good enough? QUIT! """
                # if best_reward >= goodenough_reward:
                #     print("Reward good enough")
                #     break

            """ Terminate early? """
            # early termination if no progress in last patience_iter simulations
            terminate_early = \
                (iteration - best_iter >= patience_iter) \
                or \
                (iteration - best_iter >= patience_zeroreward and best_reward == 0.0)
            if terminate_early:
                print("iteration %d: Terminate early, best_iter = %d, best_simulationcounter = %d, simulation_counter = %d, best_reward = %g"
                      % (iteration, best_iter, best_simulationcounter, simulation_counter, best_reward))
                # break

            """ Dig deeper? """
            if dig_deeper and \
                    iteration - max([dig_iter, best_iter]) >= patience_deeper and \
                    cur_root.depth < max_backdoor - 1 and \
                    best_reward > 0.0:
                dig_iter = iteration
                # temp_root = utils.get_best_child_known(cur_root, cur_root.best_child)
                # todo: revisit this decision
                temp_root = utils.get_best_child_known(cur_root, best_backdoor[cur_root.depth])
                cur_root = temp_root if temp_root is not None else cur_root

                if temp_root is not None:
                    print("Dig deeper: %d" % cur_root.depth)
                    print("---- ", cur_root.state)
                    print("best_backdoor was ", best_backdoor)

            print("time elapsed = %g" % (time.time() - time_start))
            iteration += 1

            # "consolidate" new incumbent by creating nodes in the tree for it
            if found_new_incumbent:
                print("consolidating...")
                v = root
                created_bool = False
                for var_idx, var in enumerate(best_backdoor):
                    exists_bool = False
                    for child in v.children:
                        if child.a == var:
                            v = child
                            exists_bool = True
                            break
                    if exists_bool:
                        continue
                    created_bool = True
                    v = AnyNode(id='',
                                parent=v,
                                depth=v.depth + 1,
                                state=v.state + [var],
                                a=var,
                                Q=0,
                                N=0,
                                N_prev=-1,
                                best_child=None,
                                best_child_updated=1,
                                is_finished=False, num_finished_children=0,
                                rewards=[])
                if created_bool:
                    self.backup_policy(node=v, reward=best_reward, max_backdoor=max_backdoor, num_actions=len(env.action_space))

        return best_backdoor, best_iter, reward_list, root