import os
import datetime as dt
from pathlib import Path
import logging

from tqdm import tqdm
import pandas as pd
import sklearn
import sklearn.metrics
import numpy as np
import gurobipy as grb
import pickle
import torch
import torch_geometric as tg
from ast import literal_eval
from copy import deepcopy
import re

from network.gnn_policy import GNNPolicy_ranking, GNNPolicy_cl
from network import utils

import heapq


variable_feature_names = [
    "type_CONTINUOUS",
    "type_BINARY",
    "type_INTEGER",
    "coef",
    "has_lb",
    "has_ub",
    "sol_is_at_lb",
    "sol_is_at_ub",
    "sol_frac",
    "basis_status_BASIC",
    "basis_status_NONBASIC_LOWER",
    "basis_status_NONBASIC_UPPER",
    "basis_status_SUPERBASIC",
    "reduced_cost",
    "sol_val"
    ]

constraint_feature_names = [
    "obj_cos_sim",
    "bias",
    "is_tight",
    "dualsol_val",
]

num_var_features = len(variable_feature_names)
num_con_features = len(constraint_feature_names)
env = grb.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

def calculate_overlap(list1, list2):
    """Calculate the number of overlapping elements between two lists."""
    set1 = set(list1)
    set2 = set(list2)
    return len(set1.intersection(set2))

def top_overlapping_lists_indices(reference_list, lists, top_n=5):
    """Find the indices of the top 'n' lists with the highest overlap with the reference list."""
    # Calculate overlap for each list
    overlap_counts = [(index, calculate_overlap(reference_list, lst)) for index, lst in enumerate(lists)]

    # Sort the lists based on the overlap count in descending order
    sorted_indices = sorted(overlap_counts, key=lambda x: x[1], reverse=True)

    # Select the indices of the top 'n' lists
    return sorted_indices[:top_n]

def normalize_features(data):
    # data.x_vars = # normalize this
    # data.x_cons
    var_scaler = sklearn.preprocessing.StandardScaler()
    con_scaler = sklearn.preprocessing.StandardScaler()
    data.x_vars = torch.tensor(var_scaler.fit_transform(data.x_vars.numpy()))
    data.x_cons = torch.tensor(con_scaler.fit_transform(data.x_cons.numpy()))
    return data

def compute_fractionality(mip_file, seed, presolve=False):
    """Given a mip instance and a presolve flag,
       return the indices to integer vars in the (presolved) model,
       fractionality of the integer variables,
       (presolved) model."""

    model = grb.read(mip_file)

    if presolve:
        model = model.presolve()

    model.setParam("Seed", seed)

    relaxation = model.relax()
    relaxation.optimize()

    fractions = []
    discrete_vars = [i for i, x in enumerate(model.getVars())
                     if x.vType !=grb.GRB.CONTINUOUS]
    all_relax_vars = relaxation.getVars()

    assert(len(model.getVars()) == len(all_relax_vars))

    for i in discrete_vars:
        relax_var = all_relax_vars[i]
        fractions.append(abs(round(relax_var.x) - relax_var.x))

    return discrete_vars, np.array(fractions), model

def compute_mip_representation(mip_file):
    """given a mip instance, compute features and return pytorch_geometric data instance describing variable constraint graph

    Args:
        mip_file (str): mip instance file location, readable by gurobi
    """
    model = grb.read(str(mip_file), env=env)
    A = model.getA()
    objective_coefficients = np.array([x.Obj for x in model.getVars()])
    relaxation = model.relax()
    relaxation.optimize()

    discrete_var_mask = torch.zeros(len(model.getVars()), dtype=torch.bool)

    # compute variable features
    # collect into list of variable features i.e. num_vars x num_var_features sized matrix X_v
    variable_features = []
    for var_ind, (decision_var, relax_var) in enumerate(zip(model.getVars(), relaxation.getVars())):
        feature_vector = [
            decision_var.VType == grb.GRB.CONTINUOUS,
            decision_var.VType == grb.GRB.BINARY,
            decision_var.VType == grb.GRB.INTEGER,
            decision_var.Obj,
            decision_var.LB > -grb.GRB.INFINITY,
            decision_var.UB <= grb.GRB.INFINITY,
            relax_var.x == relax_var.LB,
            relax_var.x == relax_var.UB,
            abs(round(relax_var.x) - relax_var.x),
            relax_var.VBasis == grb.GRB.BASIC,
            relax_var.VBasis == grb.GRB.NONBASIC_LOWER,
            relax_var.VBasis == grb.GRB.NONBASIC_UPPER,
            relax_var.VBasis == grb.GRB.SUPERBASIC,
            relax_var.RC,
            relax_var.x
        ]
        discrete_var_mask[var_ind] = decision_var.VType != grb.GRB.CONTINUOUS
        variable_features.append(feature_vector)

    # compute constraint features
    # collect into list of constraint features i.e. num_cons x num_con_features sized matrix X_c
    cosine_sims =sklearn.metrics.pairwise.cosine_similarity(A, objective_coefficients.reshape(1, -1))
    constraint_features = []
    for con_ind, (con, relax_con) in enumerate(zip(model.getConstrs(), relaxation.getConstrs())):
        feature_vector = [
            cosine_sims[con_ind, 0],
            con.RHS,
            relax_con.Slack == 0,
            relax_con.Pi
        ]
        constraint_features.append(feature_vector)
    # compute edge features
    # collect list of edge features i.e. num_edges x num_edge_features
    edge_features = []
    edge_indices = np.array(A.nonzero())
    # length num edges long vector of features, just contains the nonzero coeffs for now
    edge_features = A[edge_indices[0], edge_indices[1]].T

    # get mip features in graph
    data = BackdoorData(
        x_vars = torch.tensor(np.array(variable_features), dtype=torch.float),
        x_var_names = variable_feature_names,
        x_cons = torch.tensor(np.array(constraint_features), dtype=torch.float),
        x_con_names = constraint_feature_names,
        edge_index_cons_to_vars = torch.tensor(edge_indices, dtype=torch.long),
        edge_index_var_to_cons = torch.tensor(edge_indices[::-1].copy(), dtype=torch.long),
        edge_attr = torch.tensor(edge_features, dtype=torch.float),
        # edge_index = torch.tensor(edge_indices, dtype=torch.long),
        # edge_attr = torch.tensor(edge_features),
        discrete_var_mask = discrete_var_mask
    )

    return data

class BackdoorData(tg.data.Data):

    def __inc__(self, key, value):
        if key == 'edge_index_cons_to_vars':
            return torch.tensor([[self.x_cons.size(0)], [self.x_vars.size(0)]])
        elif key == "edge_index_var_to_cons":
            return torch.tensor([[self.x_vars.size(0)], [self.x_cons.size(0)]])
        else:
            return super(BackdoorData, self).__inc__(key, value)


class BackdoorDatasetCL(tg.data.InMemoryDataset):
    def __init__(self, root, instance_list, mip_distribution_name, transform=None, pre_transform=None):
        """FullBackdoorDataset dataset for a given mip distribution, maintains backdoors and lists
        uses all mip instances in mip_distribution_dir and collects the corresponding backdoors from wandb

        Args:
            root (str): directory where data is stored
            mip_distribution_dir (str): mip distribution directory containing lp files
            mip_distribution_name (str): name of mip distribution such as cauctions or similar
            mip_distribution_partition (str): train/val/test/transfer split of mip distribution
            transform (pytorch geometric transformation function, optional): for use with pytorch geometric. Defaults to None.
            pre_transform (pytorch geometric transformation, optional): for use with pytorch geometric. Defaults to None.
        """

        self.instance_list = instance_list
        self.mip_distribution_name = mip_distribution_name
        super(BackdoorDatasetCL, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def processed_file_names(self):
        return [f"{self.mip_distribution_name}_cl_all_bd.pt"]

    def process(self):
        logging.debug(f'processing {self.mip_distribution_name}')
        data_list = []

        # for each line in the instance file, get the instance, backdoor dataframes
        with open(self.instance_list, "r") as f:
            for line in f:
                instance_dir = line.strip()
                # instance file in the instance_dir has a postfix either .mps or .lp
                if list(Path(instance_dir).glob("*.mps")):
                    instance_file = list(Path(instance_dir).glob("*.mps"))[0]
                elif list(Path(instance_dir).glob("*.lp")):
                    instance_file = list(Path(instance_dir).glob("*.lp"))[0]

                org_data = compute_mip_representation(instance_file)
                org_data.machine_filepath = str(instance_file)
                org_data.mip_distribution_name = self.mip_distribution_name

                # get backdoor data in the instance_dir has a prefix backdoor_evaluate
                backdoor_file = list(Path(instance_dir).glob("backdoor_evaluate*.csv"))[0]
                
                backdoor_df = pd.read_csv(backdoor_file)

                # count = 0
                # for i, row in backdoor_df.iterrows():
                #     data = deepcopy(org_data)

                #     backdoor = row["backdoor_list"]
                #     one_hot_backdoor = np.zeros(data.x_vars.shape[0])
                #     for var_ind in literal_eval(backdoor):
                #         one_hot_backdoor[int(var_ind)] = 1
                #     data.candidate_backdoor = torch.tensor(one_hot_backdoor)
                #     data.pos_sample = []
                #     data.neg_sample = []
                #     for j in range(5):
                #         if j != count:
                #             one_hot_backdoor = np.zeros(data.x_vars.shape[0])
                #             for var_ind in literal_eval(backdoor_df["backdoor_list"].iloc[j]):
                #                 one_hot_backdoor[int(var_ind)] = 1
                #             data.pos_sample.append(one_hot_backdoor)

                #     for j in range(15,50):
                #         one_hot_backdoor = np.zeros(data.x_vars.shape[0])
                #         for var_ind in literal_eval(backdoor_df["backdoor_list"].iloc[j]):
                #             one_hot_backdoor[int(var_ind)] = 1
                #         data.neg_sample.append(one_hot_backdoor)

                #     data_list.append(data)
                #     if count >= 14:
                #         break
                #     else:
                #         count += 1

                for i, row in backdoor_df.iterrows():
                    if i >= 5 and i < 45:
                        data = deepcopy(org_data)

                        backdoor = row["backdoor_list"]
                        one_hot_backdoor = np.zeros(data.x_vars.shape[0])
                        for var_ind in literal_eval(backdoor):
                            one_hot_backdoor[int(var_ind)] = 1
                        data.candidate_backdoor = torch.tensor(one_hot_backdoor)
                        data.pos_sample = []
                        data.neg_sample = []
                        for j in range(5):
                            one_hot_backdoor = np.zeros(data.x_vars.shape[0])
                            for var_ind in literal_eval(backdoor_df["backdoor_list"].iloc[j]):
                                one_hot_backdoor[int(var_ind)] = 1
                            data.pos_sample.append(one_hot_backdoor)
    
                        for j in range(45,50):
                            one_hot_backdoor = np.zeros(data.x_vars.shape[0])
                            for var_ind in literal_eval(backdoor_df["backdoor_list"].iloc[j]):
                                one_hot_backdoor[int(var_ind)] = 1
                            data.neg_sample.append(one_hot_backdoor)
                        data_list.append(data)

                # for i, row in backdoor_df.iterrows():
                #     if i < 15:
                #         data = deepcopy(org_data)
                        
                #         reference_list = literal_eval(row["backdoor_list"])
                #         lists = []
                #         for j in range(15, 50):
                #             lists.append(literal_eval(backdoor_df["backdoor_list"].iloc[j]))
                #         top_list_indices = top_overlapping_lists_indices(reference_list, lists)
                #         idx_lst = []
                #         for index, overlap in top_list_indices:
                #             idx_lst.append(index)
                #         data.pos_sample = []
                #         data.neg_sample = []
                #         one_hot_backdoor = np.zeros(data.x_vars.shape[0])
                #         for var_ind in reference_list:
                #             one_hot_backdoor[int(var_ind)] = 1
                #         data.pos_sample.append(one_hot_backdoor)

                #         for j in idx_lst:
                #             one_hot_backdoor = np.zeros(data.x_vars.shape[0])
                #             for var_ind in literal_eval(backdoor_df["backdoor_list"].iloc[j]):
                #                 one_hot_backdoor[int(var_ind)] = 1
                #             data.neg_sample.append(one_hot_backdoor)
                #         data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        logging.debug(f'collating {self.mip_distribution_name}')
        data, slices = self.collate(data_list)
        logging.debug(f'saving {self.mip_distribution_name}')
        torch.save((data, slices), self.processed_paths[0])
        logging.debug(f'done processing {self.mip_distribution_name}')

class BackdoorDatasetRanking(tg.data.InMemoryDataset):
    def __init__(self, root, instance_list, mip_distribution_name, transform=None, pre_transform=None):
        """FullBackdoorDataset dataset for a given mip distribution, maintains backdoors and lists
        uses all mip instances in mip_distribution_dir and collects the corresponding backdoors from wandb

        Args:
            root (str): directory where data is stored
            mip_distribution_dir (str): mip distribution directory containing lp files
            mip_distribution_name (str): name of mip distribution such as cauctions or similar
            mip_distribution_partition (str): train/val/test/transfer split of mip distribution
            transform (pytorch geometric transformation function, optional): for use with pytorch geometric. Defaults to None.
            pre_transform (pytorch geometric transformation, optional): for use with pytorch geometric. Defaults to None.
        """

        self.instance_list = instance_list
        self.mip_distribution_name = mip_distribution_name
        super(BackdoorDatasetRanking, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"{self.mip_distribution_name}_ranking_all_bd.pt"]

    def process(self):
        logging.debug(f'processing {self.mip_distribution_name}')
        data_list = []

        # for each line in the instance file, get the instance, backdoor dataframes
        with open(self.instance_list, "r") as f:
            for line in f:
                instance_dir = line.strip()
                # instance file in the instance_dir has a postfix either .mps or .lp
                if list(Path(instance_dir).glob("*.mps")):
                    instance_file = list(Path(instance_dir).glob("*.mps"))[0]
                elif list(Path(instance_dir).glob("*.lp")):
                    instance_file = list(Path(instance_dir).glob("*.lp"))[0]
                org_data = compute_mip_representation(instance_file)
                org_data.machine_filepath = str(instance_file)
                org_data.mip_distribution_name = self.mip_distribution_name


                # get backdoor data in the instance_dir has a prefix backdoor_evaluate
                # backdoor_file = list(Path(instance_dir).glob("backdoor_evaluate*.csv"))[0]
                
                # match = re.search(r"backdoor_evaluate_([\d.]+).csv", str(backdoor_file))

                backdoor_file = list(Path(instance_dir).glob("backdoor_evaluate*.csv"))[0]
                
                match = re.search(r"backdoor_cl_evaluate_([\d.]+).csv", str(backdoor_file))
                gurobi_runtime = match.group(1)
                gurobi_runtime = round(float(gurobi_runtime), 2)
                
                backdoor_df = pd.read_csv(backdoor_file)
                count = 0
                backdoor_df["run_time"] = round(backdoor_df["run_time"], 2)
                backdoor_df["normalized_run_time"] = backdoor_df["run_time"] / gurobi_runtime
                for i, row in backdoor_df.sample(frac=1).iterrows():
                    data = deepcopy(org_data)

                    backdoor = row["backdoor_list"]
                    one_hot_backdoor = np.zeros(data.x_vars.shape[0])
                    
                    # for var_ind in literal_eval(backdoor):
                    #     one_hot_backdoor[int(var_ind)] = 1
                    numbers = re.findall(r'\d+', backdoor)
                    numbers = [int(num) for num in numbers]
                    for var_ind in numbers:
                        one_hot_backdoor[int(var_ind)] = 1
                        
                    data.candidate_backdoor = torch.tensor(one_hot_backdoor)
                    data.solve_time = torch.tensor([row["normalized_run_time"]]).float()

                    data_list.append(data)

                # model_file = "model_cl.pt"
                # # load predictive model
                # saved_dict = torch.load(model_file, map_location=torch.device('cpu'))
                # # args = saved_dict['args']
                # # model = ValueNet()
                # model = GNNPolicy_cl()
                # model.load_state_dict(saved_dict)
                # model.eval()

                # org_data.x_vars_batch = torch.zeros(org_data.x_vars.shape[0]).long()

                # out = model(org_data.x_cons,
                #       org_data.edge_index_cons_to_vars,
                #       org_data.edge_attr,
                #       org_data.x_vars,
                #       org_data.x_vars_batch)
                
                # cl_score = []
                # for i, row in backdoor_df.iterrows():
                #     backdoor = row["backdoor_list"]
                #     sum = 0
                #     for var_ind in literal_eval(backdoor):
                #         sum += out["output"][var_ind]
                #     cl_score.append(sum)
                # top_values = heapq.nlargest(20, cl_score)

                # # Finding the indices of these top k elements
                # top_indices = [cl_score.index(value) for value in top_values]

                # for index in top_indices:
                #     row = backdoor_df.iloc[index]
                #     data = deepcopy(org_data)

                #     backdoor = row["backdoor_list"]
                #     one_hot_backdoor = np.zeros(data.x_vars.shape[0])
                #     for var_ind in literal_eval(backdoor):
                #         one_hot_backdoor[int(var_ind)] = 1
                #     data.candidate_backdoor = torch.tensor(one_hot_backdoor)
                #     data.solve_time = torch.tensor([row["normalized_run_time"]]).float()

                #     data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        logging.debug(f'collating {self.mip_distribution_name}')
        data, slices = self.collate(data_list)
        logging.debug(f'saving {self.mip_distribution_name}')
        torch.save((data, slices), self.processed_paths[0])
        logging.debug(f'done processing {self.mip_distribution_name}')

class RankingBackdoorDataset(torch.utils.data.Dataset):
    def __init__(self, full_backdoor_dataset: BackdoorDatasetRanking):
        """
        Creates a ranking dataset from a full_backdoor_dataset
        """
        self.full_backdoor_dataset = full_backdoor_dataset
        # gets list of mip_instances for each element in dataset
        self.mip_instances = np.array([data.machine_filepath for data in full_backdoor_dataset])
        # gets list of all pairs of indices into full_backdoor_dataset that match mip_instances
        self.matching_pairs = []

        for i in tqdm(range(len(self.mip_instances)), desc="load dataset"):
            for j in range(i+1, len(self.mip_instances)):
                if self.mip_instances[i] == self.mip_instances[j]:
                    self.matching_pairs.append((i, j))

        self.matching_pairs = np.array(self.matching_pairs)
        # self.matching_pairs = np.array([(i, j) for i, elt in enumerate(self.mip_instances) for j in np.where(self.mip_instances==elt)[0]])


    def __len__(self):
        return len(self.matching_pairs)

    def __getitem__(self, index: int):
        r"""
        Returns the pair given the pair index
        """
        # Return data sample
        pair_ind = self.matching_pairs[index]
        return (self.full_backdoor_dataset[pair_ind[0]], self.full_backdoor_dataset[pair_ind[1]])
    
class BackdoorDatasetClassifier(tg.data.InMemoryDataset):
    def __init__(self, root, instance_list, mip_distribution_name, transform=None, pre_transform=None):
        """FullBackdoorDataset dataset for a given mip distribution, maintains backdoors and lists
        uses all mip instances in mip_distribution_dir and collects the corresponding backdoors from wandb

        Args:
            root (str): directory where data is stored
            mip_distribution_dir (str): mip distribution directory containing lp files
            mip_distribution_name (str): name of mip distribution such as cauctions or similar
            mip_distribution_partition (str): train/val/test/transfer split of mip distribution
            transform (pytorch geometric transformation function, optional): for use with pytorch geometric. Defaults to None.
            pre_transform (pytorch geometric transformation, optional): for use with pytorch geometric. Defaults to None.
        """

        self.instance_list = instance_list
        self.mip_distribution_name = mip_distribution_name
        super(BackdoorDatasetClassifier, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"{self.mip_distribution_name}_classifer_all_bd.pt"]

    def process(self):
        logging.debug(f'processing {self.mip_distribution_name}')
        data_list = []

        # for each line in the instance file, get the instance, backdoor dataframes
        with open(self.instance_list, "r") as f:
            for line in f:
                instance_dir = line.strip()
                # instance file in the instance_dir has a postfix either .mps or .lp
                if list(Path(instance_dir).glob("*.mps")):
                    instance_file = list(Path(instance_dir).glob("*.mps"))[0]
                elif list(Path(instance_dir).glob("*.lp")):
                    instance_file = list(Path(instance_dir).glob("*.lp"))[0]
                data = compute_mip_representation(instance_file)
                data.machine_filepath = str(instance_file)
                data.mip_distribution_name = self.mip_distribution_name
                data.x_vars_batch = torch.zeros(data.x_vars.shape[0]).long()

                # for scorer
                # # get backdoor data in the instance_dir has a prefix backdoor_evaluate
                # backdoor_file = list(Path(instance_dir).glob("backdoor_evaluate*.csv"))[0]
                
                # match = re.search(r"backdoor_evaluate_([\d.]+).csv", str(backdoor_file))
                # gurobi_runtime = match.group(1)
                # gurobi_runtime = round(float(gurobi_runtime), 2)
                
                # backdoor_df = pd.read_csv(backdoor_file)
                # backdoor_df["run_time"] = round(backdoor_df["run_time"], 2)
                # backdoor_df["normalized_run_time"] = backdoor_df["run_time"] / gurobi_runtime
                # model_file = "model_ranking.pt"
                # # load predictive model
                # saved_dict = torch.load(model_file, map_location=torch.device('cpu'))
                # # args = saved_dict['args']
                # # model = ValueNet()
                # model = GNNPolicy_ranking()
                # model.load_state_dict(saved_dict)
                # model.eval()
                # lowest_val = 999
                # lowest_runtime = 0
                # lowest_backdoor = []
                # m = grb.read(str(instance_file))
                
                # for i, row in backdoor_df.iterrows():
                #     candidate_backdoor = literal_eval(row["backdoor_list"])
                #     backdoor_indicators = np.zeros(len(m.getVars()))
                #     backdoor_indicators[candidate_backdoor] = 1
            
                #     out = model(data.x_cons,
                #                       data.edge_index_cons_to_vars,
                #                       data.edge_attr,
                #                       torch.hstack([data.x_vars, torch.tensor(backdoor_indicators).float().unsqueeze(-1)]),
                #                       data.x_vars_batch)
                #     estimated_value = utils.to_numpy(out["output"])[0]

                #     if estimated_value < lowest_val:
                #         lowest_val = estimated_value
                #         lowest_runtime = row["normalized_run_time"]
                #         lowest_backdoor = torch.tensor(backdoor_indicators)
                
                # if lowest_runtime <= 1:
                #     data.use = 1
                # else:
                #     data.use = 0
                # data.candidate_backdoor = lowest_backdoor
                # data_list.append(data)
                backdoor_file = list(Path(instance_dir).glob("backdoor_results.csv"))[0]
                backdoor_df = pd.read_csv(backdoor_file)
                if backdoor_df.iloc[0]["baseline"] < backdoor_df.iloc[0]["total_time_cl"]:
                    data.use = 0
                else:
                    data.use = 1
                numbers = literal_eval(backdoor_df.iloc[0]["backdoor"][7:][:-1])
                one_hot_backdoor = np.zeros(data.x_vars.shape[0])
                for var_ind in numbers:
                    one_hot_backdoor[int(var_ind)] = 1
                data.candidate_backdoor = torch.tensor(one_hot_backdoor)
                data_list.append(data)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        logging.debug(f'collating {self.mip_distribution_name}')
        data, slices = self.collate(data_list)
        logging.debug(f'saving {self.mip_distribution_name}')
        torch.save((data, slices), self.processed_paths[0])
        logging.debug(f'done processing {self.mip_distribution_name}')

# def get_BG_from_GRB(ins_name):
#     #vars:  [obj coeff, norm_coeff, degree, max coeff, min coeff, Bin?]
    
#     m=grb.read(str(ins_name))
#     ori_start=6
#     emb_num=15
    
#     mvars=m.getVars()
#     mvars.sort(key=lambda v:v.VarName)

#     v_map={}
#     for indx,v in enumerate(mvars):
#         v_map[v.VarName]=indx

#     nvars=len(mvars)
    
#     v_nodes=[]
#     b_vars=[]
#     for i in range(len(mvars)):
#         tp=[0]*ori_start
#         tp[3]=0
#         tp[4]=1e+20
#         #tp=[0,0,0,0,0]
#         if mvars[i].VType=='B':
#             tp[ori_start-1]=1
#             b_vars.append(i)
            
#         v_nodes.append(tp)
#     obj=m.getObjective()
#     obj_cons=[0]*(nvars+2)
#     obj_node=[0,0,0,0]
    
#     nobjs=obj.size()
#     for i in range(nobjs):
#         vnm=obj.getVar(i).VarName
#         v=obj.getCoeff(i)
#         v_indx=v_map[vnm]
#         obj_cons[v_indx]=v
#         v_nodes[v_indx][0]=v
#         obj_node[0]+=v
#         obj_node[1]+=1
#     obj_node[0]/=obj_node[1]
#     cons=m.getConstrs()
#     ncons=len(cons)
#     lcons=ncons
#     c_nodes=[]
    
#     A=[]
#     for i in range(ncons):
#         A.append([])
#         for j in range(nvars+2):
#             A[i].append(0)
#     A.append(obj_cons)
#     for i in range(ncons):
#         tmp_v=[]
#         tmp_c=[]
        
#         sense=cons[i].Sense
#         rhs=cons[i].RHS
#         nzs=0
        
#         if sense=='<':
#             sense=0
#         elif sense=='>':
#             sense=1
#         elif sense=='=':
#             sense=2
#         tmp_c=[0,0,rhs,sense]
#         summation=0
#         tmp_v=[0,0,0,0,0]
#         for v in mvars:
#             v_indx=v_map[v.VarName]
#             ce=m.getCoeff(cons[i],v)
            
#             if ce!=0:
#                 nzs+=1
#                 summation+=ce
#                 A[i][v_indx]=1
#                 A[i][-1]+=1
            
#         if nzs==0:
#             continue
#         tmp_c[0]=summation/nzs
#         tmp_c[1]=nzs
#         c_nodes.append(tmp_c)
#         for v in mvars:
#             v_indx=v_map[v.VarName]
#             ce=m.getCoeff(cons[i],v)
            
#             if ce!=0:            
#                 v_nodes[v_indx][2]+=1
#                 v_nodes[v_indx][1]+=ce/lcons
#                 v_nodes[v_indx][3]=max(v_nodes[v_indx][3],ce)
#                 v_nodes[v_indx][4]=min(v_nodes[v_indx][4],ce)
    
#     c_nodes.append(obj_node)
#     v_nodes=torch.as_tensor(v_nodes,dtype=torch.float32)   
#     c_nodes=torch.as_tensor(c_nodes,dtype=torch.float32)
#     b_vars=torch.as_tensor(b_vars,dtype=torch.int32)   

#     A=np.array(A,dtype=np.float32)
#     A=A[:,:-2]
#     A=torch.as_tensor(A).to_sparse()
#     clip_max=[20000,1,torch.max(v_nodes,0)[0][2].item()]
#     clip_min=[0,-1,0]
    
#     v_nodes[:,0]=torch.clamp(v_nodes[:,0],clip_min[0],clip_max[0])
    
    
#     maxs=torch.max(v_nodes,0)[0]
#     mins=torch.min(v_nodes,0)[0]
#     diff=maxs-mins
#     for ks in range(diff.shape[0]):
#         if diff[ks]==0:
#             diff[ks]=1
#     v_nodes=v_nodes-mins
#     v_nodes=v_nodes/diff
#     v_nodes=torch.clamp(v_nodes,1e-5,1)
#     #v_nodes=position_get_ordered(v_nodes)
#     # v_nodes=position_get_ordered_flt(v_nodes)
    
    
#     maxs=torch.max(c_nodes,0)[0]
#     mins=torch.min(c_nodes,0)[0]
#     diff=maxs-mins
#     for ks in range(diff.shape[0]):
#         if diff[ks]==0:
#             diff[ks]=1
#     c_nodes=c_nodes-mins
#     c_nodes=c_nodes/diff
#     c_nodes=torch.clamp(c_nodes,1e-5,1)

#     edge_indices = A._indices()
#     edge_features =A._values().unsqueeze(1)
#     edge_features=torch.ones(edge_features.shape)
    
#     data = BackdoorData(
#         x_vars = v_nodes,
#         x_cons = c_nodes,
#         edge_index_cons_to_vars = torch.tensor(edge_indices, dtype=torch.long),
#         edge_attr = torch.tensor(edge_features, dtype=torch.float)
#     )

#     return data 

# class BackdoorDatasetCL2(tg.data.InMemoryDataset):
#     def __init__(self, root, instance_list, mip_distribution_name, transform=None, pre_transform=None):
#         """FullBackdoorDataset dataset for a given mip distribution, maintains backdoors and lists
#         uses all mip instances in mip_distribution_dir and collects the corresponding backdoors from wandb

#         Args:
#             root (str): directory where data is stored
#             mip_distribution_dir (str): mip distribution directory containing lp files
#             mip_distribution_name (str): name of mip distribution such as cauctions or similar
#             mip_distribution_partition (str): train/val/test/transfer split of mip distribution
#             transform (pytorch geometric transformation function, optional): for use with pytorch geometric. Defaults to None.
#             pre_transform (pytorch geometric transformation, optional): for use with pytorch geometric. Defaults to None.
#         """

#         self.instance_list = instance_list
#         self.mip_distribution_name = mip_distribution_name
#         super(BackdoorDatasetCL2, self).__init__(root, transform, pre_transform)

#         self.data, self.slices = torch.load(self.processed_paths[0])
        
#     @property
#     def processed_file_names(self):
#         return [f"{self.mip_distribution_name}_cl_all_bd.pt"]

#     def process(self):
#         logging.debug(f'processing {self.mip_distribution_name}')
#         data_list = []

#         # for each line in the instance file, get the instance, backdoor dataframes
#         with open(self.instance_list, "r") as f:
#             for line in f:
#                 instance_dir = line.strip()
#                 # instance file in the instance_dir has a postfix either .mps or .lp
#                 if list(Path(instance_dir).glob("*.mps")):
#                     instance_file = list(Path(instance_dir).glob("*.mps"))[0]
#                 elif list(Path(instance_dir).glob("*.lp")):
#                     instance_file = list(Path(instance_dir).glob("*.lp"))[0]

#                 org_data = get_BG_from_GRB(instance_file)
#                 org_data.machine_filepath = str(instance_file)
#                 org_data.mip_distribution_name = self.mip_distribution_name
#                 org_data.x_vars_batch = torch.zeros(org_data.x_vars.shape[0]).long()

#                 # get backdoor data in the instance_dir has a prefix backdoor_evaluate
#                 backdoor_file = list(Path(instance_dir).glob("backdoor_evaluate*.csv"))[0]
                
#                 backdoor_df = pd.read_csv(backdoor_file)


#                 for i, row in backdoor_df.iterrows():
#                     if i >= 5 and i < 45:
#                         data = deepcopy(org_data)

#                         backdoor = row["backdoor_list"]
#                         one_hot_backdoor = np.zeros(data.x_vars.shape[0])
#                         for var_ind in literal_eval(backdoor):
#                             one_hot_backdoor[int(var_ind)] = 1
#                         data.candidate_backdoor = torch.tensor(one_hot_backdoor)
#                         data.pos_sample = []
#                         data.neg_sample = []
#                         for j in range(5):
#                             one_hot_backdoor = np.zeros(data.x_vars.shape[0])
#                             for var_ind in literal_eval(backdoor_df["backdoor_list"].iloc[j]):
#                                 one_hot_backdoor[int(var_ind)] = 1
#                             data.pos_sample.append(one_hot_backdoor)
    
#                         for j in range(45,50):
#                             one_hot_backdoor = np.zeros(data.x_vars.shape[0])
#                             for var_ind in literal_eval(backdoor_df["backdoor_list"].iloc[j]):
#                                 one_hot_backdoor[int(var_ind)] = 1
#                             data.neg_sample.append(one_hot_backdoor)
#                         data_list.append(data)

#         if self.pre_filter is not None:
#             data_list = [data for data in data_list if self.pre_filter(data)]

#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]

#         logging.debug(f'collating {self.mip_distribution_name}')
#         data, slices = self.collate(data_list)
#         logging.debug(f'saving {self.mip_distribution_name}')
#         torch.save((data, slices), self.processed_paths[0])
#         logging.debug(f'done processing {self.mip_distribution_name}')

# def position_get_ordered_flt(variable_features):
    
#     lens=variable_features.shape[0]
#     feature_widh=20 #max length 4095
#     sorter=variable_features[:,1]
#     position=torch.argsort(sorter)
#     position=position/float(lens)
    
#     position_feature=torch.zeros(lens,feature_widh)
    
#     for row in range(position.shape[0]):
#         flt_indx=position[row]
#         divider=1.0
#         for ks in range(feature_widh):
#             if divider<=flt_indx:
#                 position_feature[row][ks]=1
#                 flt_indx-=divider
#             divider/=2.0 
#         #print(row,position[row],position_feature[row])
#     position_feature=position_feature
#     variable_features=variable_features
#     v=torch.concat([variable_features,position_feature],dim=1)
#     return v   
        