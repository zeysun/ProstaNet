import torch
import torch.nn as nn
from Layer import GCN, MLP
from torch_geometric.nn import global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep, AttentionalAggregation
from torch_geometric.nn.norm import GraphNorm
from torch.nn import Linear
import torch.nn.functional as F


'''
Architecture of GCN-based method for cross-validation
'''
class staGCNN(nn.Module):
    def __init__(self, n_output=1,
                 num_features_pro=71,
                 inner_dim=71,
                 output_dim=71,
                 num_layers=2,
                 dropout=0.6):
        super(staGCNN, self).__init__()
        '''
        n_output: dimension of output
        num_features_pro: dimension of input features
        inner_dim: dimension of inner layers in the final dense layers
        output_dim: output dimension of dense layer after pooling layer
        num_dense_layers: number of last dense layers
        dropout: dropout rate
        '''
        print('Model Loaded')

        # For wild_type
        self.n_output = n_output
        self.wild_conv1 = GCN(num_features_pro, num_features_pro) #GCN layers processing wild-type protein graphs
        self.wild_conv2 = GCN(num_features_pro, num_features_pro)
        self.wild_fc1 = nn.Linear(num_features_pro, output_dim)
        self.wild_fc2 = nn.Linear(num_features_pro, num_features_pro)
        self.wild_ba1 = nn.BatchNorm1d(output_dim)

        # For mutant
        self.mutant_conv1 = GCN(num_features_pro, num_features_pro)
        self.mutant_conv2 = GCN(num_features_pro, num_features_pro)
        self.mutant_fc1 = nn.Linear(num_features_pro, output_dim)
        self.mutant_fc2 = nn.Linear(num_features_pro, num_features_pro)
        self.mutant_ba1 = nn.BatchNorm1d(output_dim)

        self.relu = nn.ReLU() #activation function
        self.dropout = nn.Dropout(dropout)

        # Combined layers
        self.wild_MLP = MLP(2*output_dim, n_output, inner_dim, num_layers)

    def forward(self, wild_data, mutant_data):
        # Get graph input for wild_type
        wild_x, wild_edge_index, wild_batch = wild_data.x, wild_data.edge_index, wild_data.batch
        # Get graph input for mutant
        mutant_x, mutant_edge_index, mutant_batch = mutant_data.x, mutant_data.edge_index, mutant_data.batch

        h = wild_x
        x = self.wild_conv1(wild_x, wild_edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        h = h + x #skip connection

        x = self.wild_conv2(h, wild_edge_index)

        # Global pooling
        x = gmp(x, wild_batch)

        # Flatten
        x = self.relu(self.wild_fc1(x))
        x = self.dropout(x)
        #x = self.wild_ba1(x)

        # Get graph input for mutant
        ht = mutant_x
        xt = self.mutant_conv1(mutant_x, mutant_edge_index)
        xt = self.relu(xt)
        xt = self.dropout(xt)
        ht = ht + xt

        xt = self.mutant_conv2(ht, mutant_edge_index)

        # Global pooling
        xt = gmp(xt, mutant_batch)

        # Flatten
        xt = self.relu(self.mutant_fc1(xt))
        xt = self.dropout(xt)

        # Concatenation  
        xc = torch.cat((xt, x),1)

        # Dense layers
        out = self.wild_MLP(xc)

        return out
    
net = staGCNN()
print(net)
