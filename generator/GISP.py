import numpy as np
import networkx as nx
import fire
import gurobipy as grb
import os

def partition_edges(edges, alpha, seed=1):
    np.random.seed(seed)
    E1 = set()
    E2 = set()
    for e in edges:
        if np.random.rand() <= alpha:
            E2.add(e)
        else:
            E1.add(e)
    return E1, E2

def generate_MIP(nodes, E1, E2, node_weight, edge_cost):
    m = grb.Model()
    node_vars = {i: m.addVar(obj=-node_weight,vtype=grb.GRB.BINARY) for i in nodes}

    edge_vars = {(i,j): m.addVar(obj=edge_cost, vtype=grb.GRB.BINARY) for (i,j) in E2}

    m.setAttr("modelSense", grb.GRB.MINIMIZE)
    m.update()
    for (i,j) in E1:
        m.addConstr(node_vars[i]+node_vars[j], grb.GRB.LESS_EQUAL, 1)
    for (i,j) in E2:
        m.addConstr(node_vars[i]+node_vars[j]-edge_vars[(i,j)], grb.GRB.LESS_EQUAL, 1)
    
    m.update()

    return m


def generate(filename, seed=1, nodes=30, edge_prob=0.7, edge_cost=1, node_weight=100, alpha=0.75):
    """
    Generate a generalized independent set instance

bibtex


    Saves it as a CPLEX LP file.

    Parameters
    ----------
    filename : str
        Path to the file to save.
    """

    rng = np.random.RandomState(seed)
    graph_seed = rng.randint(2**31)
    # generate graph (or collect from dimacs)
    graph = nx.erdos_renyi_graph(nodes, edge_prob, seed=graph_seed)

    # generate mip
    partiton_seed = rng.randint(2**31)
    E1, E2 = partition_edges(graph.edges, alpha, partiton_seed)
    problem = generate_MIP(graph.nodes, E1, E2, node_weight, edge_cost)
    problem.write(filename)

def main():
    # 1. Create a folder called 'ca'
    if not os.path.exists('gisp'):
        os.mkdir('gisp')

    # # 2. Create a subfolder called 'train' and call the generate function 200 times
    # train_folder = os.path.join('ca', 'train')
    # if not os.path.exists(train_folder):
    #     os.mkdir(train_folder)

    # for i in range(1, 201):
    #     filename = os.path.join(train_folder, f'ca_{i}.lp')
    #     generate(filename, seed=i+10000, n_items=200, n_bids=1000)

    # 3. Create a subfolder called 'test' and call the generate function 100 times
    # test_folder = os.path.join('gisp_h', 'test')
    # if not os.path.exists(test_folder):
    #     os.mkdir(test_folder)

    for i in range(1, 1124):
        filename = os.path.join("gisp", f'gisp_{i}.lp')
        generate(filename, seed=i+5433, nodes=150)

if __name__ == '__main__':
    main()