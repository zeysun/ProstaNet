import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from gvp.models import GVP, GVPConvLayer, LayerNorm
from torch_scatter import scatter_mean


'''
Architecture of single GVP-GNN layer 
'''
class SinGVP(nn.Module):
    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim,
                 drop_rate, num_layers, n_message, n_feedforward):
        '''
        node_in_dim: node dimensions in input graph
        node_h_dim: node dimensions to use in GVP-GNN layers
        edge_in_dim: edge dimensions in input graph
        edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers
        drop_rate: rate to use in all dropout layers
        num_layers: number of GVP-GNN layers in each of the encoder and decoder modules
        n_message: Number of GVPs to use in message function
        n_feedforward: Number of GVPs to use in feedforward function
        '''
        super(SinGVP, self).__init__()

        node_in_dim = (node_in_dim[0], node_in_dim[1])

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, n_message, n_feedforward, drop_rate) 
            for _ in range(num_layers))
        
        node_h_dim = (node_h_dim[0] * num_layers,
                      node_h_dim[1] * num_layers)
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

    def forward(self, h_V, edge_index, h_E, batch):
        '''
        h_V: tuple (s, V) of node embeddings
        edge_index: `torch.Tensor` of shape [2, num_edges]
        h_E: tuple (s, V) of edge embeddings
        '''
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        h_V_out = []
        h_V_in = h_V
        for layer in self.layers:
            h_V_out.append(layer(h_V_in, edge_index, h_E))
            h_V_in = h_V_out[-1]
        h_V_out = (
            torch.cat([h_V[0] for h_V in h_V_out], dim=-1),
            torch.cat([h_V[1] for h_V in h_V_out], dim=-2),
        )
        out = self.W_out(h_V_out)

        out = scatter_mean(out, batch, dim=0)

        return out
