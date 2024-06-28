import torch
import copy
import torch.nn as nn
from typing import List
from torch import Tensor
from Layer_GVP_optuna import SinGVP
from Layer_optuna import MLP


'''
Architecture of ProstaNet
'''
class StaGVP(nn.Module):
    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, num_MLP, inner, num_GVP, drop_rate, n_message, n_feedforward):
        super(StaGVP, self).__init__()

        print('Model Loaded')

        '''
        the graphs of wild-type and mutant are passed through GVP-GNN layers
        '''
        self.wild_conv1 = SinGVP(node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, drop_rate, num_GVP, n_message, n_feedforward)
        
        self.mutant_conv1 = SinGVP(node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, drop_rate, num_GVP, n_message, n_feedforward)
        '''
        node_in_dim: node dimensions (s, V) in input graph
        node_h_dim: node dimensions to use in GVP-GNN layers
        edge_in_dim: edge dimensions (s, V) in input graph
        edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers
        num_GVP: number of GVP-GNN layers
        drop_rate: rate to use in all dropout layers
        n_message: Number of GVPs to use in message function
        n_feedforward: Number of GVPs to use in feedforward function
        '''
        # concat outputs from GVPConvLayer(s)
        node_h_dim = (
            node_h_dim[0] * num_GVP,
            node_h_dim[1] * num_GVP,
        )
        ns, _ = node_h_dim

        self.MLP = MLP(2*ns, 1, inner*ns, num_MLP, drop_rate) #inner*ns is the dimension of inner layers

    def forward(self, wild_data, mutant_data):
        h_V_w = (wild_data.node_s, wild_data.node_v) #node features, node_s node scalar features, node_v node vector features
        h_E_w = (wild_data.edge_s, wild_data.edge_v) #edge features, edge_s edge scalar features, edge_v edge vector features
        edge_index_w = wild_data.edge_index
        batch_w = wild_data.batch

        h_V_m = (mutant_data.node_s, mutant_data.node_v)
        h_E_m = (mutant_data.edge_s, mutant_data.edge_v)
        edge_index_m = mutant_data.edge_index
        batch_m = mutant_data.batch
        
        x = self.wild_conv1(h_V_w, edge_index_w, h_E_w, batch_w)
        xt = self.mutant_conv1(h_V_m, edge_index_m, h_E_m, batch_m)

        xc = torch.cat((xt, x), 1) #concatenate two features vectors from two structure graphs

        out = self.MLP(xc) #pass the feature vector from graph pass through dense layers

        return out
