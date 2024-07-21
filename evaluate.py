import os
import math
from pathlib import Path

import pandas as pd

import time

import numpy as np
import gurobipy as grb
from gurobipy import GRB

import argparse

import network.MIPDataset as MIPDataset
import network.utils as utils
from network.gnn_policy import GNNPolicy_ranking, GNNPolicy_cl
import glob
import torch

import copy
import pickle

def softmax(vector):
 e = np.exp(vector)
 return e / e.sum()

gurobi_log = {"baseline":[], "cl":[]}

def mycallback(model, where):

    if where == GRB.Callback.MIPSOL:

        # Access solution values using the custom attribute model._vars
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        log_entry_name = model.Params.LogFile

        log_entry = []
        log_entry.append(obj)
        log_entry.append(model.cbGet(GRB.Callback.RUNTIME))
        gurobi_log[log_entry_name].append(log_entry)

        # Print the solution with variable names
        #print('**** New solution of obj %g '%obj)
        # print("New solution", obj, "Runtime", runtime)
        #for i in range(len(model._vars)):
        #   print(f"{model._vars[i].VarName}: {sol[i]}")

def eval_backdoor_model(instance_dir, seed=0):
    env = grb.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.setParam("Threads",1)
    env.start()
    torch.set_num_threads(1)
    if glob.glob('%s/*.mps*' % instance_dir):
        instance_path = glob.glob('%s/*.mps*' % instance_dir)[0]
    else:
        instance_path = glob.glob('%s/*.lp*' % instance_dir)[0]
    m = grb.read(instance_path, env = env)

    m.optimize()
    baseline = m.Runtime

    start_time = time.time()
    # restore model file
    model_file = "model_cl.pt"
    # load predictive model
    saved_dict = torch.load(model_file, map_location=torch.device('cpu'))
    # args = saved_dict['args']
    # model = ValueNet()
    model = GNNPolicy_cl()
    model.load_state_dict(saved_dict)
    model.eval()

    mip_data = MIPDataset.compute_mip_representation(instance_path)
    # mip_data = MIPDataset.get_BG_from_GRB(instance_path)
    mip_data.x_vars_batch = torch.zeros(mip_data.x_vars.shape[0]).long()

    out = model(mip_data.x_cons,
          mip_data.edge_index_cons_to_vars,
          mip_data.edge_attr,
          mip_data.x_vars,
          mip_data.x_vars_batch)
    values, indices = torch.topk(out["output"], 8)

    # model_file = "model_classifier.pt"
    # # load predictive model
    # saved_dict = torch.load(model_file, map_location=torch.device('cpu'))
    # # args = saved_dict['args']
    # # model = ValueNet()
    # model = GNNPolicy_ranking()
    # model.load_state_dict(saved_dict)
    # model.eval()

    # one_hot_backdoor = np.zeros(mip_data.x_vars.shape[0])
    # for var_ind in indices:
    #     one_hot_backdoor[int(var_ind)] = 1
    # backdoor = torch.tensor(one_hot_backdoor)
    # backdoor = backdoor.float()

    # out = model(mip_data.x_cons,
    #                     mip_data.edge_index_cons_to_vars,
    #                     mip_data.edge_attr,
    #                     torch.hstack([mip_data.x_vars, backdoor.unsqueeze(-1)]),
    #                     mip_data.x_vars_batch)

    m = grb.read(instance_path, env = env)
    for i in range(len(m.getVars())):
        if i in indices:
            m.getVars()[i].BranchPriority = 2
        else:
            m.getVars()[i].BranchPriority = 1
    m.update()
    m.optimize()
    total_time_cl = time.time() - start_time


    start_time = time.time()
    # restore model file
    model_file = "model_PaS_GAT_1.pth"
    # load predictive model
    saved_dict = torch.load(model_file, map_location=torch.device('cpu'))
    # args = saved_dict['args']
    # model = ValueNet()
    model = GNNPolicy_cl()
    model.load_state_dict(saved_dict)
    model.eval()

    mip_data = MIPDataset.compute_mip_representation(instance_path)
    # mip_data = MIPDataset.get_BG_from_GRB(instance_path)
    mip_data.x_vars_batch = torch.zeros(mip_data.x_vars.shape[0]).long()

    out = model(mip_data.x_cons,
          mip_data.edge_index_cons_to_vars,
          mip_data.edge_attr,
          mip_data.x_vars,
          mip_data.x_vars_batch)
    values, indices = torch.topk(out["output"], 8)
    m = grb.read(instance_path, env = env)
    for i in range(len(m.getVars())):
        if i in indices:
            m.getVars()[i].BranchPriority = 2
        else:
            m.getVars()[i].BranchPriority = 1
    m.update()
    m.optimize()
    total_time_cl_2 = time.time() - start_time

    # score = out["output"].detach().numpy() 
    

    # start_time = time.time()
    # # restore model file
    # model_file = "model_ranking.pt"
    # # load predictive model
    # saved_dict = torch.load(model_file, map_location=torch.device('cpu'))
    # # args = saved_dict['args']
    # # model = ValueNet()
    # model = GNNPolicy_ranking()
    # model.load_state_dict(saved_dict)
    # model.eval()

    # mip_data = MIPDataset.compute_mip_representation(instance_path)
    # mip_data.x_vars_batch = torch.zeros(mip_data.x_vars.shape[0]).long()
    
    # sampling_eps = 0.001
    # sampling_weights = utils.to_numpy(mip_data.x_vars[:, 8])
    # sampling_weights = np.maximum(sampling_weights, sampling_eps)
    # sampling_weights[~mip_data.discrete_var_mask] = 0
    
    # sampling_weights /= sampling_weights.sum()

    # worst_backdoor = None
    # worst_backdoor_val = np.inf

    # num_bd_samples = 50
    # backdoor_size = 8
    # for i in range(num_bd_samples):
    # #     # decision variables
    #     candidate_backdoor = np.random.choice(
    #         range(len(m.getVars())), backdoor_size, replace=False, p=sampling_weights)
    #     backdoor_indicators = np.zeros(len(m.getVars()))
    #     backdoor_indicators[candidate_backdoor] = 1

    #     out = model(mip_data.x_cons,
    #                                   mip_data.edge_index_cons_to_vars,
    #                                   mip_data.edge_attr,
    #                                   torch.hstack([mip_data.x_vars, torch.tensor(backdoor_indicators).float().unsqueeze(-1)]),
    #                                   mip_data.x_vars_batch)
    #     estimated_value = utils.to_numpy(out["output"])[0]
    #     if estimated_value < worst_backdoor_val:
    #         worst_backdoor = backdoor_indicators.copy()
    #         worst_backdoor_val = estimated_value
    # m = grb.read(instance_path, env = env)
    # for i in range(len(m.getVars())):
    #     if worst_backdoor[i] == 1:
    #         m.getVars()[i].BranchPriority = 2
    #     else:
    #         m.getVars()[i].BranchPriority = 1
    # m.update()
    # m.optimize()
    # total_time_ranking = time.time() - start_time

    # start_time = time.time()
    # model_file = "model_cl.pt"
    # # load predictive model
    # saved_dict = torch.load(model_file, map_location=torch.device('cpu'))
    # # args = saved_dict['args']
    # # model = ValueNet()
    # model = GNNPolicy_cl()
    # model.load_state_dict(saved_dict)
    # model.eval()

    # mip_data = MIPDataset.compute_mip_representation(instance_path)
    # mip_data.x_vars_batch = torch.zeros(mip_data.x_vars.shape[0]).long()

    # out = model(mip_data.x_cons,
    #       mip_data.edge_index_cons_to_vars,
    #       mip_data.edge_attr,
    #       mip_data.x_vars,
    #       mip_data.x_vars_batch)
    # sampling_weights = np.array(out["output"].detach())
    # sampling_weights = softmax(sampling_weights)
    # sorted_array = np.sort(sampling_weights)
    # threshold = sorted_array[-20]
    # sampling_weights[sampling_weights < threshold] = 0
    # sampling_weights = softmax(sampling_weights)

    # worst_backdoor = None
    # worst_backdoor_val = np.inf

    # num_bd_samples = 50
    # backdoor_size = 8
    # model_file = "model_ranking.pt"
    # # load predictive model
    # saved_dict = torch.load(model_file, map_location=torch.device('cpu'))
    # # args = saved_dict['args']
    # # model = ValueNet()
    # model = GNNPolicy_ranking()
    # model.load_state_dict(saved_dict)
    # model.eval()
    # for i in range(num_bd_samples):
    #     # decision variables
    #     candidate_backdoor = np.random.choice(
    #         range(len(m.getVars())), backdoor_size, replace=False, p=sampling_weights)
    #     backdoor_indicators = np.zeros(len(m.getVars()))
    #     backdoor_indicators[candidate_backdoor] = 1

    #     out = model(mip_data.x_cons,
    #                                   mip_data.edge_index_cons_to_vars,
    #                                   mip_data.edge_attr,
    #                                   torch.hstack([mip_data.x_vars, torch.tensor(backdoor_indicators).float().unsqueeze(-1)]),
    #                                   mip_data.x_vars_batch)
    #     estimated_value = utils.to_numpy(out["output"])[0]
    #     if estimated_value < worst_backdoor_val:
    #         worst_backdoor = backdoor_indicators.copy()
    #         worst_backdoor_val = estimated_value
    # m = grb.read(instance_path, env = env)      
    # for i in range(len(m.getVars())):
    #     if worst_backdoor[i] == 1:
    #         m.getVars()[i].BranchPriority = 2
    #     else:
    #         m.getVars()[i].BranchPriority = 1
    # m.update()
    # m.optimize()
    # total_time_cl_ranking = time.time() - start_time
    
    # df = pd.DataFrame(columns = ['baseline', 'total_time_cl', 'total_time_ranking', 'total_time_cl_ranking'])
    # df = df._append({'baseline': baseline, 'total_time_cl': total_time_cl, 'total_time_ranking': total_time_ranking, 'total_time_cl_ranking': total_time_cl_ranking}, ignore_index=True)
    
    # df = pd.DataFrame(columns = ['baseline', 'total_time_cl', 'total_time_ranking'])
    # df = df._append({'baseline': baseline, 'total_time_cl': total_time_cl, 'total_time_ranking': total_time_ranking}, ignore_index=True)

    df = pd.DataFrame(columns = ['baseline', 'total_time_cl', 'total_time_cl_b'])
    df = df._append({'baseline': baseline, 'total_time_cl': total_time_cl, 'total_time_cl_b': total_time_cl_2}, ignore_index=True)

    # df = pd.DataFrame(columns = ['baseline', 'total_time_cl_ranking'])
    # df = df._append({'baseline': baseline, 'total_time_cl_ranking': total_time_cl_ranking}, ignore_index=True)

    df.to_csv(os.path.join(instance_dir, "backdoor_finetune.csv"), index=False)


if __name__ == "__main__":
    parser_main = argparse.ArgumentParser()

    parser_main.add_argument("--instance_dir", type=str)
    parser_main.add_argument("--seed", type=str, default = 0)
    args_main = parser_main.parse_args()

    eval_backdoor_model(instance_dir = args_main.instance_dir)
    
