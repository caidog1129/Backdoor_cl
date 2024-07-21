import torch
import torch.nn.init as init
import torch_geometric as tg
import torch.nn.functional as F

from network import prenorm 
from network import MIPDataset
from network import gat_convolution

# Implements the branching policy described in
# https://papers.nips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html
class GNNPolicy_cl(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = MIPDataset.num_con_features
        edge_nfeats = 1
        var_nfeats = MIPDataset.num_var_features

        # Constraint embedding
        self.cons_norm = prenorm.Prenorm(cons_nfeats)
        self.cons_embedding = torch.nn.Sequential(
            self.cons_norm,
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # Edge embedding
        self.edge_norm = prenorm.Prenorm(edge_nfeats)
        self.edge_embedding = torch.nn.Sequential(
            self.edge_norm,
            torch.nn.Linear(edge_nfeats, emb_size),
        )

        # Variable embedding
        self.var_norm = prenorm.Prenorm(var_nfeats, preserve_features=[2])
        self.var_embedding = torch.nn.Sequential(
            self.var_norm,
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = gat_convolution.GATConvolution()
        self.conv_c_to_v = gat_convolution.GATConvolution()  

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )
        self.reset_parameters()


    def reset_parameters(self):
        for t in self.parameters():
            if len(t.shape) == 2:
                init.orthogonal_(t)
            else:
                init.normal_(t)

    def freeze_normalization(self):
        if not self.cons_norm.frozen:
            self.cons_norm.freeze_normalization()
            self.edge_norm.freeze_normalization()
            self.var_norm.freeze_normalization()
            self.conv_v_to_c.reset_normalization()
            self.conv_c_to_v.reset_normalization()
            return False
        if not self.conv_v_to_c.frozen:
            self.conv_v_to_c.freeze_normalization()
            self.conv_c_to_v.reset_normalization()
            return False
        if not self.conv_c_to_v.frozen:
            self.conv_c_to_v.freeze_normalization()
            return False
        return True


    def forward(self, constraint_features, edge_indices, edge_features, variable_features, variable_features_batch):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        
        # Two half convolutions
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)
 
        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)

        out = {}
        out["output"] = output
        return out

class GNNPolicy_ranking(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = MIPDataset.num_con_features
        edge_nfeats = 1
        var_nfeats = MIPDataset.num_var_features + 1

        # Constraint embedding
        self.cons_norm = prenorm.Prenorm(cons_nfeats)
        self.cons_embedding = torch.nn.Sequential(
            self.cons_norm,
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # Edge embedding
        self.edge_norm = prenorm.Prenorm(edge_nfeats)
        self.edge_embedding = torch.nn.Sequential(
            self.edge_norm,
            torch.nn.Linear(edge_nfeats, emb_size),
        )

        # Variable embedding
        self.var_norm = prenorm.Prenorm(var_nfeats, preserve_features=[2])
        self.var_embedding = torch.nn.Sequential(
            self.var_norm,
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = gat_convolution.GATConvolution()
        self.conv_c_to_v = gat_convolution.GATConvolution()  
        
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )
        self.reset_parameters()


    def reset_parameters(self):
        for t in self.parameters():
            if len(t.shape) == 2:
                init.orthogonal_(t)
            else:
                init.normal_(t)

    def freeze_normalization(self):
        if not self.cons_norm.frozen:
            self.cons_norm.freeze_normalization()
            self.edge_norm.freeze_normalization()
            self.var_norm.freeze_normalization()
            self.conv_v_to_c.reset_normalization()
            self.conv_c_to_v.reset_normalization()
            return False
        if not self.conv_v_to_c.frozen:
            self.conv_v_to_c.freeze_normalization()
            self.conv_c_to_v.reset_normalization()
            return False
        if not self.conv_c_to_v.frozen:
            self.conv_c_to_v.freeze_normalization()
            return False
        return True


    def forward(self, constraint_features, edge_indices, edge_features, variable_features, variable_features_batch):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        # Two half convolutions
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)
 
        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)

        output = tg.nn.global_mean_pool(output, variable_features_batch)
        
        out = {}
        out["output"] = output

        return out

# class MultiSequential(torch.nn.Sequential):
#     def forward(self, *xs):
#         for module in self._modules.values():
#             xs = module(*xs)
#         return xs
        
# class GeometricUnit(torch.nn.Module):
#     def __init__(self, var2con, con2var, activation='leaky_relu', last_layer=False, num_heads=1):
#         super(GeometricUnit, self).__init__()
#         self.var2con = var2con
#         self.con2var = con2var
#         self.last_layer = last_layer
#         self.activation = activation
#         self.num_heads = num_heads
#         if not self.last_layer:
#             self.con_batchnorm = tg.nn.BatchNorm(in_channels=self.var2con.out_channels * self.num_heads)
#             self.var_batchnorm = tg.nn.BatchNorm(in_channels=self.con2var.out_channels * self.num_heads)

#     def forward(self, var_embeddings, con_embeddings, edge_index_var2cons, edge_index_con2vars):
#         out_con_embeddings = self.var2con(x=(var_embeddings, con_embeddings), edge_index=edge_index_var2cons)
#         out_var_embeddings = self.con2var(x=(con_embeddings, var_embeddings), edge_index=edge_index_con2vars)
        
#         if not self.last_layer:
#             out_con_embeddings = self.con_batchnorm(out_con_embeddings)
#             out_var_embeddings = self.var_batchnorm(out_var_embeddings)
            
#             if self.activation == "leaky_relu":
#                 out_con_embeddings = F.leaky_relu(out_con_embeddings)
#                 out_var_embeddings = F.leaky_relu(out_var_embeddings)
#             elif self.activation == "relu":
#                 out_con_embeddings = F.relu(out_con_embeddings)
#                 out_var_embeddings = F.relu(out_var_embeddings)
#             else:
#                 raise NotImplemented(f"activation {self.activation} is not implemented")
#             # import IPython; import sys; IPython.embed(); sys.exit(1)

#             # out_con_embeddings = F.dropout(out_con_embeddings, p=0.5, training=self.training)
#             # out_var_embeddings = F.dropout(out_var_embeddings, p=0.5, training=self.training)
#         # else:
#         #     out_con_embeddings = F.relu(out_con_embeddings)
#         #     out_var_embeddings = F.relu(out_var_embeddings)
        

#         return out_var_embeddings, out_con_embeddings, edge_index_var2cons, edge_index_con2vars


# def simple_nn(feature_sizes, activate_last_layer=True):
#     layers = []
#     for layer_ind, (in_size, out_size) in enumerate(zip(feature_sizes[:-1], feature_sizes[1:])):
#         layers.append(torch.nn.Linear(in_size, out_size))
#         if activate_last_layer or (layer_ind < len(feature_sizes) - 2):
#             layers.append(torch.nn.BatchNorm1d(out_size))
#             layers.append(torch.nn.ReLU())
#             layers.append(torch.nn.Dropout())
#     return torch.nn.Sequential(*layers)
    
# class RunMLNet(torch.nn.Module):
#     """
#     Network module for predicting, given a mip and a candidate backdoor set, whether it's a backdoor or not
#     """
#     def __init__(self, activation="leaky_relu", **kwargs):
#         super(RunMLNet, self).__init__()
#         # sizes of variable embedding, and constraint embedding
#         # including initial features
        
#         # includes an additional feature (set inclusion)
#         feature_sizes = [
#             (MIPDataset.num_var_features+1, MIPDataset.num_con_features),
#             (64, 64),
#             (64, 64),
#             (64, 64),
#             ]
        
#         gate_nn_feature_sizes = [
#             feature_sizes[-1][0],
#             64,
#             1
#         ]

#         prediction_feature_sizes = [
#             feature_sizes[-1][0],
#             64,
#             1
#         ]

#         self.args = {
#             "class": "RunMLNet",
#             "activation": activation
#         }
        
#         is_last_layer = [False for i in range(len(feature_sizes))]
#         is_last_layer[-1] = True

#         # list of geometric units for use in sequential network
#         self.convs = [
#             GeometricUnit(
#                 var2con = tg.nn.GATConv(in_channels=(prev_var_feats, prev_con_feats), heads=1, out_channels=next_con_feats, add_self_loops=False),
#                 con2var = tg.nn.GATConv(in_channels=(prev_con_feats, prev_var_feats), heads=1, out_channels=next_var_feats, add_self_loops=False),
#                 last_layer=last_layer,
#                 activation=activation,
#                 num_heads=1
#             )
#             for (prev_var_feats, prev_con_feats), (next_var_feats, next_con_feats), last_layer  in zip(feature_sizes[:-1], feature_sizes[1:], is_last_layer[1:])
#         ]
#         self.model = MultiSequential(*self.convs)

#         # TODO probably use something other than this, maybe nnlib
#         # TODO figure out what nn should be here
#         gate_nn = simple_nn(gate_nn_feature_sizes, activate_last_layer=False)
#         self.pooling_module = tg.nn.GlobalAttention(gate_nn=gate_nn, nn=None)

#         self.prediction_module = simple_nn(prediction_feature_sizes, activate_last_layer=False)


#     def forward(self, data, set_indicators):
#         """
#         computes variable logits (constraint logits are thrown out but can be used if desired)
#         """
#         out = {}
#         var_embeddings, _, _, _ = self.model(torch.hstack([data.x_vars, set_indicators.unsqueeze(-1)]), data.x_cons, data.edge_index_var_to_cons, data.edge_index_cons_to_vars)
#         mip_representation = self.pooling_module(var_embeddings, batch=data.x_vars_batch)

#         mip_predictions = self.prediction_module(mip_representation)

#         out["var_embeddings"] = var_embeddings
#         out["mip_representation"] = mip_representation
#         out["mip_predictions"] = mip_predictions
        
#         return out

# class GNNPolicy(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         emb_size = 64
#         cons_nfeats = 4
#         edge_nfeats = 1
#         var_nfeats = 6

#         # CONSTRAINT EMBEDDING
#         self.cons_embedding = torch.nn.Sequential(
#             torch.nn.LayerNorm(cons_nfeats),
#             torch.nn.Linear(cons_nfeats, emb_size),
#             torch.nn.ReLU(),
#             torch.nn.Linear(emb_size, emb_size),
#             torch.nn.ReLU(),
#         )

#         # EDGE EMBEDDING
#         self.edge_embedding = torch.nn.Sequential(
#             torch.nn.LayerNorm(edge_nfeats),
#         )

#         # VARIABLE EMBEDDING
#         self.var_embedding = torch.nn.Sequential(
#             torch.nn.LayerNorm(var_nfeats),
#             torch.nn.Linear(var_nfeats, emb_size),
#             torch.nn.ReLU(),
#             torch.nn.Linear(emb_size, emb_size),
#             torch.nn.ReLU(),
#         )

#         self.conv_v_to_c = BipartiteGraphConvolution()
#         self.conv_c_to_v = BipartiteGraphConvolution()

#         self.conv_v_to_c2 = BipartiteGraphConvolution()
#         self.conv_c_to_v2 = BipartiteGraphConvolution()

#         self.output_module = torch.nn.Sequential(
#             torch.nn.Linear(emb_size, emb_size),
#             torch.nn.ReLU(),
#             torch.nn.Linear(emb_size, 1, bias=False),
#         )

#     def forward(
#         self, constraint_features, edge_indices, edge_features, variable_features
#     ):
#         reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

#         # First step: linear embedding layers to a common dimension (64)
#         constraint_features = self.cons_embedding(constraint_features)
#         edge_features = self.edge_embedding(edge_features)
#         variable_features = self.var_embedding(variable_features)

#         # Two half convolutions
#         constraint_features = self.conv_v_to_c(
#             variable_features, reversed_edge_indices, edge_features, constraint_features
#         )
#         variable_features = self.conv_c_to_v(
#             constraint_features, edge_indices, edge_features, variable_features
#         )

#         constraint_features = self.conv_v_to_c2(
#             variable_features, reversed_edge_indices, edge_features, constraint_features
#         )
#         variable_features = self.conv_c_to_v2(
#             constraint_features, edge_indices, edge_features, variable_features
#         )

#         # A final MLP on the variable features
#         output = self.output_module(variable_features).squeeze(-1)

#         return output

# class BipartiteGraphConvolution(tg.nn.MessagePassing):
#     """
#     The bipartite graph convolution is already provided by pytorch geometric and we merely need
#     to provide the exact form of the messages being passed.
#     """

#     def __init__(self):
#         super().__init__("add")
#         emb_size = 64

#         self.feature_module_left = torch.nn.Sequential(
#             torch.nn.Linear(emb_size, emb_size)
#         )
#         self.feature_module_edge = torch.nn.Sequential(
#             torch.nn.Linear(1, emb_size, bias=False)
#         )
#         self.feature_module_right = torch.nn.Sequential(
#             torch.nn.Linear(emb_size, emb_size, bias=False)
#         )
#         self.feature_module_final = torch.nn.Sequential(
#             torch.nn.LayerNorm(emb_size),
#             torch.nn.ReLU(),
#             torch.nn.Linear(emb_size, emb_size),
#         )

#         self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))


#         # output_layers
#         self.output_module = torch.nn.Sequential(
#             torch.nn.Linear(2 * emb_size, emb_size),
#             torch.nn.ReLU(),
#             torch.nn.Linear(emb_size, emb_size),
#         )

#     def forward(self, left_features, edge_indices, edge_features, right_features):
#         """
#         This method sends the messages, computed in the message method.
#         """


#         output = self.propagate(
#             edge_indices,
#             size=(left_features.shape[0], right_features.shape[0]),
#             node_features=(left_features, right_features),
#             edge_features=edge_features,
#         )
#         b=torch.cat([self.post_conv_module(output), right_features], dim=-1)
#         a=self.output_module(
#             torch.cat([self.post_conv_module(output), right_features], dim=-1)
#         )

#         return self.output_module(
#             torch.cat([self.post_conv_module(output), right_features], dim=-1)
#         )


#     def message(self, node_features_i, node_features_j, edge_features):
#         #node_features_i,the node to be aggregated
#         #node_features_j,the neighbors of the node i

#         # print("node_features_i:",node_features_i.shape)
#         # print("node_features_j",node_features_j.shape)
#         # print("edge_features:",edge_features.shape)

#         output = self.feature_module_final(
#             self.feature_module_left(node_features_i)
#             + self.feature_module_edge(edge_features)
#             + self.feature_module_right(node_features_j)
#         )

#         return output