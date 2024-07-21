from search_algo.mcts import MCTSAlgorithm
from search_algo.random import RandomAlgorithm, RandomSampler
# from search_algo.setcover import SetCoverAlgorithm

from environments.base import Env
from environments.parallel_env import ParallelEnv

from sim_policies.uniform import UniformSimPolicy

# from expansion_policies.uniform import UniformExpansionPolicy
from expansion_policies.progressive_widening import PWExpansionPolicy

from tree_policies.uct import UCTreePolicy

# from backup_policies.sum import SumBackupPolicy
from backup_policies.max import MaxBackupPolicy

from solvers.priority_stdcb import PrioritySolverStandard

# from reward_functions.gap import GapReward
# from reward_functions.leaf_freq import LeafFreqReward
from reward_functions.tree_weight import TreeWeightReward

import numpy as np
# from tensorboardX import GlobalSummaryWriter
import time
from anytree import AnyNode, RenderTree
import argparse
import math
import os
import configargparse
import glob
from datetime import datetime


def main(instance_dir,
         tf_dir,
         config='configs/debug.ini',
         seed=0,
         cpx_mem=1024,
         parallel=2,
         max_time=1e9):

    parser = configargparse.ArgParser(default_config_files=[config])

    parser.add('--config', default=config, required=False, is_config_file=True, help='config file path')

    # common parameters for all methods
    parser.add("--method", type=str, help="mcts, biased, random, setcover")
    parser.add("--max_iter", type=int, default=100)

    # backdoor size parameters
    parser.add("--max_backdoor", type=int, default=-1)
    parser.add("--max_backdoor_method", type=str, default="fixed")
    parser.add("--size_search_strategy", type=str, default="sequential", help="binarysearch or sequential")
    parser.add("--min_actions", type=int, default=-1)
    parser.add("--backdoor_lowerbd", type=int, default=2)
    parser.add("--backdoor_upperbd", type=int, default=5)
    parser.add("--backdoor_frac", type=float, default=0.1, help='if max_backdoor_method=relative, this is active')
    parser.add("--exclude_root_int", default=False, action='store_true')

    # warmstarting or early termination parameters
    parser.add("--acceptable_reward", type=float, default=(1 - 1e-6))
    parser.add("--goodenough_reward", type=float, default=0.9)
    parser.add("--patience_iter_frac", type=float, default=0.3)
    parser.add("--patience_zeroreward", type=float, default=0.05)
    parser.add("--patience_deeper", type=float, default=0.1)
    parser.add("--warmstart_sol", default=False, action='store_true')
    parser.add("--warmstart_root", default=False, action='store_true')

    # mcts-specific
    # todo add parameters to reward (make sure all rewards are in [0,1], higher is better)
    # todo add parameters for tree and backup policies!
    parser.add("--mcts_backup", type=str, default="max")
    parser.add("--mcts_uct_variance", default=False, action='store_true')
    parser.add("--mcts_uct_exploration_c", type=str, default="np.sqrt(2)")
    parser.add("--mcts_uct_raveweight", type=float, default=0.1)
    parser.add("--mcts_dig_deeper", default=False, action='store_true')
    parser.add("--mcts_best_child_criterion", type=str, default="count", help="count or value")
    parser.add("--mcts_expansion_type", type=str, default="best_score")
    parser.add("--mcts_ignore_ordering", default=False, action='store_true')

    # setcover-specific
    parser.add("--setcover_maxlpsols", type=int, default=10)

    # solver parameters, common to all methods
    parser.add("--cpx_threads", type=int, default=1)
    parser.add("--cpx_time", type=float, default=600) # todo make it instance-specific based on statistical estimates
    parser.add("--cpx_display", type=int, default=0)
    parser.add("--cpx_heuristicfreq", type=int, default=0)
    parser.add("--reuse_instance", default=False, action='store_true')
    parser.add('--presolve', default=False, action='store_true')
    parser.add('--presolve_overwrite', default=False, action='store_true')
    parser.add('--get_root_lp', default=False, action='store_true')

    # parallelism-related
    parser.add("--sim_count", type=int, default=parallel)

    # tensorboard
    parser.add('--tens_board', default=False, action='store_true')

    args, unknown = parser.parse_known_args()
    print(args)
    print(parser.format_values())

    print('instance_dir = %s' % instance_dir)

    config_filename = os.path.splitext(os.path.basename(args.config))[0]

    # paths for inputs
    if glob.glob('%s/*.mps*' % instance_dir):
        instance_mps_path = glob.glob('%s/*.mps*' % instance_dir)[0]
    else:
        instance_mps_path = glob.glob('%s/*.lp*' % instance_dir)[0]
    instance_warmstart_path = glob.glob('%s/*.sol' % instance_dir)
    instance_warmstart_path = None if not args.warmstart_sol or len(instance_warmstart_path) == 0 else instance_warmstart_path[0]
    instance_mps_filename = os.path.basename(instance_mps_path)

    # paths for outputs
    # instance_output_dir = '%s/%s/%s/%d' % (instance_dir, args.method, config_filename, seed)
    # os.makedirs(instance_output_dir, exist_ok=True)
    # instance_tree_path = '%s/mcts.tree' % instance_output_dir
    instance_bkd_path = '%s/backdoor.bkd' % instance_dir
    # instance_incb_path = '%s/incumbent' % instance_output_dir
    backdoor_list = '%s/backdoor.csv' % instance_dir

    # set up tensorboard directory
    # time_str = datetime.now().strftime("%d%m%Y-%H%M")
    # tf_dir = f"../tf_events/{args.method}/{config_filename}/{seed}/"
    # s_writer = GlobalSummaryWriter(tf_dir,
    #                                write_to_disk=args.tens_board, flush_secs=5)

    # set numpy random seed globally
    np.random.seed(seed)

    # set solver branching strategy: default (0) if setcover, mininfeasible (-1) otherwise
    cpx_variableselect = 0 if args.method == 'setcover' else -1

    limits_init = {'backdoor_frac': args.backdoor_frac,
                   'min_actions': args.min_actions,
                   'backdoor_lowerbd': args.backdoor_lowerbd, 'backdoor_upperbd': args.backdoor_upperbd,
                   'max_iter': args.max_iter, 'max_time': max_time, 'max_time_solver': args.cpx_time,
                   'goodenough_reward': args.goodenough_reward if args.max_backdoor == -1 else args.acceptable_reward,
                   'patience_iter': max([10, int(args.patience_iter_frac*args.max_iter)]),
                   'patience_zeroreward': max([50, int(args.patience_zeroreward*args.max_iter)]),
                   'patience_deeper': int(args.patience_deeper*args.max_iter),
                   'max_lp_sols': args.setcover_maxlpsols, 'cpx_threads': args.cpx_threads, 'cpx_mem': cpx_mem}
    print(limits_init)
    limits_cur = dict(limits_init)

    solver_maker = lambda: PrioritySolverStandard(limits_init=limits_init, warmstart_root=args.warmstart_root)
    env_args = dict(instance_path=instance_mps_path, max_backdoor=args.max_backdoor,
                    seed=seed,
                    cpx_time=args.cpx_time,
                    cpx_variableselect=cpx_variableselect,
                    solver_maker=solver_maker, reward_function=TreeWeightReward(),
                    presolve_flag=args.presolve, presolved_dir_path=instance_dir, presolved_overwrite=args.presolve_overwrite,
                    initializeopt_flag=args.warmstart_sol, initializeopt_dir_path=instance_warmstart_path,
                    get_root_lp=args.get_root_lp, root_lp=None,
                    exclude_root_int=args.exclude_root_int, reuse_instance=args.reuse_instance,
                    cpx_threads=args.cpx_threads, cpx_mem=cpx_mem, cpx_display=args.cpx_display,
                    cpx_heuristicfreq=args.cpx_heuristicfreq, ignore_ordering=args.mcts_ignore_ordering)

    if parallel > 1:
        # # todo: if sim_count = 1, use Env even if parallel > 1
        if args.method in ["mcts", "random", "biased"]:
            print(f"Running with Parallel Env ({parallel} workers).")
            env_args.update(dict(num_envs=parallel))
            env = ParallelEnv(**env_args)
        elif args.method == 'setcover':
            print(f"Warning: Parallel Env is not supported for {args.method}. Running with Base Env (sim_count=1).")
            env_args['cpx_threads'] = max([parallel, args.cpx_threads])
            env = Env(**env_args)
        else:
            print(f"Warning: {args.method} is invalid. ")
            exit()
    else:
        parallel = 1
        if args.sim_count > 1:
            print("Base Env cannot work with sim_count > 1. Use Parallel Env instead. Setting sim_count=1.")
            # exit(1)
        args.sim_count = 1

        print("Running with Base Env.")
        env = Env(**env_args)

    print("num_vars =", env.num_vars)
    print("int_vars =", len(env.int_vars))
    print("num actions =", len(env.action_space))

    num_actions = len(env.action_space)
    num_int = len(env.int_vars)

    if num_actions <= limits_init['min_actions']:
        print("too easy")
        exit()

    time_start = time.time()
    with open(instance_bkd_path, "w") as text_file:
        print(time.time(), file=text_file)
        print(seed, file=text_file)
        print(args, file=text_file)
        print(limits_init, file=text_file)
        print(num_int, file=text_file)
        print(num_actions, file=text_file)

    with open(backdoor_list, "w") as text_file:
        print("reward;backdoor_list", file=text_file)

    best_reward_global = 0.0
    meta_iteration_counter = 0
    goodenough_reward_new = limits_init['goodenough_reward']
    terminate_bool = False
    if args.method != 'setcover':
        # todo: even if best_reward_global = 1, should try to look for smaller backdoors of same quality
        while best_reward_global <= args.acceptable_reward and ((args.max_backdoor == -1) or (meta_iteration_counter == 0)) and (terminate_bool == False):
            outer_iteration_counter = 0

            limits_cur['goodenough_reward'] = goodenough_reward_new
            print("meta_iteration = %d, limits_cur['goodenough_reward'] = %g"
                  % (meta_iteration_counter, limits_cur['goodenough_reward']))
            while not terminate_bool:
                iteration_binarysearch = 0
                cur_backdoor_size = -1

                if args.max_backdoor == -1:
                    min_backdoor_size = limits_cur['backdoor_lowerbd']
                    max_backdoor_size = limits_cur['backdoor_upperbd']
                    if args.max_backdoor_method == "relative":
                        max_backdoor_size = max([min_backdoor_size + 1, min([max_backdoor_size,
                                                                             int(math.floor(
                                                                                 limits_cur[
                                                                                     'backdoor_frac'] * num_actions))])])
                    elif args.max_backdoor_method == "fixed":
                        max_backdoor_size = min([limits_cur['backdoor_upperbd'], num_actions])
                else:
                    min_backdoor_size, max_backdoor_size = args.max_backdoor, args.max_backdoor
                    terminate_bool = True
                terminate_bool = True

                # limit_names = ['max_iter', 'max_time', 'max_time_solver', 'patience_iter', 'patience_zeroreward', 'patience_deeper']
                limit_names = ['max_iter', 'max_time', 'patience_iter', 'patience_zeroreward', 'patience_deeper']
                for limit_name in limit_names:
                    limits_cur[limit_name] *= (2 - ((iteration_binarysearch + outer_iteration_counter + meta_iteration_counter) == 0))

                print("outer_iteration_counter", outer_iteration_counter, "Doubled iteration/time limits to ",
                      limits_cur['max_iter'], limits_cur['max_time'], limits_cur['max_time_solver'],
                      limits_cur['patience_iter'], limits_cur['patience_zeroreward'])

                while min_backdoor_size <= max_backdoor_size:
                    print("---------------------")
                    print("outer_iteration_counter =", outer_iteration_counter)
                    print("iteration_binarysearch =", iteration_binarysearch)

                    cur_backdoor_size_prev = cur_backdoor_size

                    if args.max_backdoor != -1:
                        cur_backdoor_size = args.max_backdoor
                    elif args.size_search_strategy == 'binarysearch':
                        cur_backdoor_size = int(math.floor(((max_backdoor_size + min_backdoor_size) / 2)))
                    elif args.size_search_strategy == 'sequential':
                        cur_backdoor_size = min([min_backdoor_size, max_backdoor_size])

                    print("%d in [%d, %d]; prev was %d"
                          % (cur_backdoor_size, min_backdoor_size, max_backdoor_size, cur_backdoor_size_prev))

                    env.max_backdoor = cur_backdoor_size
                    # env.cpx_time = limits_cur['max_time_solver']

                    reward_list = []
                    if args.method == 'mcts':
                        if iteration_binarysearch + outer_iteration_counter + meta_iteration_counter == 0:
                            sim_policy = UniformSimPolicy(count=args.sim_count)
                            expansion_policy = PWExpansionPolicy(expansion_type=args.mcts_expansion_type)
                            tree_policy = UCTreePolicy(expansion_policy=expansion_policy,
                                                       variance=args.mcts_uct_variance,
                                                       rave_weight=args.mcts_uct_raveweight,
                                                       track_finished=(args.max_backdoor != -1),
                                                       c=eval(args.mcts_uct_exploration_c))
                            if args.mcts_backup == 'sum':
                                pass
                                # backup_policy = SumBackupPolicy(best_child_criterion=args.mcts_best_child_criterion,
                                #                                 track_finished=(args.max_backdoor != -1))
                            elif args.mcts_backup == 'max':
                                backup_policy = MaxBackupPolicy(best_child_criterion=args.mcts_best_child_criterion,
                                                                track_finished=(args.max_backdoor != -1))
                            else:
                                print("Backup policy does not exist, aborting...")
                                exit()

                            searcher = MCTSAlgorithm(sim_policy, tree_policy, backup_policy)
                            mcts_root = AnyNode(id="root", parent=None, depth=0, state=[], a=None, Q=0, N=0, N_prev=-1,
                                                best_child=None, best_child_updated=1, rewards=[],
                                                is_finished=False, num_finished_children=0,
                                                pseudocosts_avg=[1e-9]*env.num_vars,
                                                pseudocosts_count=[0]*env.num_vars)

                        best_backdoor, best_iter, reward_list, mcts_root = searcher(root=mcts_root,
                                                                                    env=env,
                                                                                    limits_init=limits_cur,
                                                                                    max_backdoor=cur_backdoor_size,
                                                                                    dig_deeper=args.mcts_dig_deeper,
                                                                                    backdoor_file=instance_bkd_path,
                                                                                    backdoor_list=backdoor_list)
                        # print(RenderTree(mcts_root))

                    else:
                        if iteration_binarysearch + outer_iteration_counter + meta_iteration_counter == 0:
                            sampler = RandomSampler(sim_cnt=parallel, env=env, strategy=args.method, seed=seed)

                            searcher = RandomAlgorithm(sampler)

                        best_backdoor, best_iter, reward_list = searcher(env=env,
                                                                         limits_init=limits_cur,
                                                                         max_backdoor=cur_backdoor_size,
                                                                         backdoor_file=instance_bkd_path,
                                                                         backdoor_list=backdoor_list)

                    best_reward = reward_list[best_iter] if best_iter >= 0 else 0.0
                    best_reward_global = max([best_reward_global, best_reward])
                    # track: best_reward, best_reward_global, cur_backdoor_size
                    # s_writer.add_scalar(f"{instance_mps_filename}/best_reward", best_reward)
                    # s_writer.add_scalar(f"{instance_mps_filename}/best_reward_global", best_reward_global)
                    # s_writer.add_scalar(f"{instance_mps_filename}/cur_backdoor_size", cur_backdoor_size)

                    print("************")
                    print("Results from iteration_binarysearch %i/%i/%i" %
                          (meta_iteration_counter, outer_iteration_counter, iteration_binarysearch))
                    print(iteration_binarysearch,
                          best_reward,
                          cur_backdoor_size,
                          int(reward_list[best_iter] >= limits_cur['goodenough_reward']))
                    print("best_backdoor_final =", ','.join(map(str, best_backdoor)))
                    print("reward_list_final =", ','.join(map(str, reward_list)))
                    print("total time elapsed =", time.time() - time_start)
                    print("************")

                    if args.max_backdoor != -1:
                        break

                    # terminate_bool = (best_reward_global >= limits_cur['goodenough_reward'])
                    # if terminate_bool:
                    #     goodenough_reward_new = best_reward_global + (1-best_reward_global)/2
                    #     cur_backdoor_size = len(best_backdoor)

                    if args.size_search_strategy == 'binarysearch':
                        if best_reward >= limits_cur['goodenough_reward']:
                            max_backdoor_size = cur_backdoor_size - 1
                        else:
                            min_backdoor_size = cur_backdoor_size + 1
                    elif args.size_search_strategy == 'sequential':
                        if best_reward >= limits_cur['goodenough_reward']:
                            max_backdoor_size = 0
                        else:
                            min_backdoor_size = cur_backdoor_size + 1

                    iteration_binarysearch += 1
                outer_iteration_counter += 1
            meta_iteration_counter += 1

    elif args.method == 'setcover':
        pass
        # searcher = SetCoverAlgorithm()

        # max_backdoor_setcover = limits_init['backdoor_upperbd'] if args.max_backdoor == -1 else args.max_backdoor
        # best_backdoor, reward_list = searcher(env=env,
        #                                       limits_init=limits_init,
        #                                       max_backdoor=max_backdoor_setcover,
        #                                       backdoor_file=instance_bkd_path)
        # best_iter = len(reward_list)

        # print("************")
        # print("Results from iteration_binarysearch %i" % 0)
        # print(0, 0, len(best_backdoor), 1)
        # print("best_backdoor_final =", ','.join(map(str, best_backdoor)))
        # print("reward_list_final =", ','.join(map(str, reward_list)))
        # print("total time elapsed =", time.time() - time_start)
        # print("************")

    else:
        print("Method does not exist")

    # s_writer.close()
    print("SEARCH DONE")

    with open(instance_bkd_path, "a") as text_file:
        print("terminated at %d" % (time.time()), file=text_file)


if __name__ == '__main__':
    parser_main = argparse.ArgumentParser()

    parser_main.add_argument("--instance_dir", type=str, default='../OUTPUT_debug/50v-10')
    parser_main.add_argument("--tf_dir", type=str, default='../tf_events_debug/%s' % (datetime.now().strftime("%d%m%Y-%H%M")))
    parser_main.add_argument("--config", type=str, default='configs/debug.ini')
    parser_main.add_argument("--seed", type=int, default=0)
    parser_main.add_argument("--cpx_mem", type=int, default=1024)
    parser_main.add_argument("--parallel", type=int, default=10)
    parser_main.add_argument("--max_time", type=float, default=1e6)

    args_main = parser_main.parse_args()
    print(args_main)

    main(instance_dir=args_main.instance_dir,
         tf_dir=args_main.tf_dir,
         config=args_main.config,
         seed=args_main.seed,
         cpx_mem=args_main.cpx_mem,
         parallel=args_main.parallel,
         max_time=args_main.max_time)