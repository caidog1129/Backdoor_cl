import numpy as np
import yaml
from anytree.exporter import DictExporter
from anytree.importer import DictImporter


# assumes sol_file exists
def read_sol_miplib(sol_file):
    with open(sol_file, "r") as file:
        lines = file.readlines()

    sol_dict = dict()
    for line in lines[2:]:
        line_list = line.strip().split(" ")
        sol_dict[line_list[0]] = float(line_list[1])
    return sol_dict


def write_tree(root, tree_file):
    tree_out = DictExporter().export(root)
    with open(tree_file, "w") as file:
        yaml.dump(tree_out, file)


def read_tree(tree_file):
    with open(tree_file, "r") as file:
        tree_in = yaml.load(file.read())
    root = DictImporter().import_(tree_in)
    return root


def update_pseudocosts(pseudocosts_new_cpx, pseudocosts_avg, pseudocosts_count, backdoor):
    pc_down = np.array([pc[0] for pc in pseudocosts_new_cpx])
    pc_up = np.array([pc[1] for pc in pseudocosts_new_cpx])

    pseudocosts_avg[backdoor] = (pseudocosts_avg[backdoor] * pseudocosts_count[backdoor] +
                                 pc_down[backdoor] + pc_up[backdoor]) / (pseudocosts_count[backdoor] + 1)

    pseudocosts_count[backdoor] += 1


def is_integer(values, int_vars=None, int_tol=1e-5):
    int_vars = np.arange(len(values)) if int_vars is None else int_vars
    values_np = np.array(values)[int_vars]
    values_int_bool = np.abs(values_np - np.round(values_np)) < int_tol
    values_int_idx = np.where(values_int_bool)[0]
    values_frac_idx = np.where(~values_int_bool)[0]

    return np.array(int_vars)[values_frac_idx], np.array(int_vars)[values_int_idx], len(values_int_idx) == len(int_vars)


def get_best_child_known(node, best_child_action):
    for child in node.children:
        if child.a == best_child_action:
            return child
    return None


def get_best_child(node, criterion):
    best_score = -1
    best_score2 = -1
    best_child = None
    for child in node.children:
        if criterion == 'count':
            child_score = child.N
            child_score2 = child.Q
        elif criterion == 'value':
            child_score = child.Q
            child_score2 = child.N
        better_bool = (child_score > best_score) or (child_score == best_score and child_score2 > best_score2)
        if better_bool:
            best_child = child.a
            best_score = child_score
            best_score2 = child_score2

    return best_child, best_score


def set_best_child(node, best_child_criterion):
    best_child, best_score = get_best_child(node, best_child_criterion)
    if node.best_child is None or node.best_child != best_child:
        node.best_child_updated = node.N
    node.best_child = best_child


def get_best_leaf(root, criterion):
    cur_node = root
    while len(cur_node.children) > 0:
        cur_node, _ = get_best_child(node=cur_node, criterion=criterion)

    return cur_node


def disable_output_cpx(instance_cpx):
    instance_cpx.set_log_stream(None)
    instance_cpx.set_error_stream(None)
    instance_cpx.set_warning_stream(None)
    instance_cpx.set_results_stream(None)