import torch
import torch.nn as nn
from Layer_optuna import GCN, MLP
from torch_geometric.nn import global_max_pool as gmp, global_add_pool as gap, global_mean_pool as gep, AttentionalAggregation
from torch_geometric.nn.norm import GraphNorm
from torch.nn import Linear



class staGCNN(nn.Module):
    def __init__(self, trial):

        super(staGCNN, self).__init__()

        n_output=1
        num_features_pro = 1024

        self.p = 0.3
        self.num_dense_layers = 1
        self.output_dim = 1024
        self.inner_dim = 512
        

        # For wild_type
        self.n_output = n_output
        self.wild_conv1 = GCN(num_features_pro, num_features_pro)
        self.wild_conv2 = GCN(num_features_pro, num_features_pro)
        self.wild_fc1 = nn.Linear(num_features_pro, self.output_dim)
        self.wild_fc2 = nn.Linear(num_features_pro, num_features_pro)
        self.wild_ba1 = nn.BatchNorm1d(self.output_dim)
        self.norm = GraphNorm(num_features_pro)
        self.att = AttentionalAggregation(Linear(num_features_pro, 1))

        # For mutant
        self.mutant_conv1 = GCN(num_features_pro, num_features_pro)
        self.mutant_conv2 = GCN(num_features_pro, num_features_pro)
        self.mutant_fc1 = nn.Linear(num_features_pro, self.output_dim)
        self.mutant_fc2 = nn.Linear(num_features_pro, num_features_pro)
        self.mutant_ba1 = nn.BatchNorm1d(self.output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.p)

        # Combined layers
        self.wild_MLP = MLP(2*self.output_dim, n_output, self.inner_dim, self.num_dense_layers, self.p)

    def forward(self, wild_data, mutant_data):
        # Get graph input for wild_type
        wild_x, wild_edge_index, wild_batch = wild_data.x, wild_data.edge_index, wild_data.batch
        #print(wild_data.ptr)
        # Get graph input for mutant
        mutant_x, mutant_edge_index, mutant_batch = mutant_data.x, mutant_data.edge_index, mutant_data.batch

        h = wild_x
        x = self.wild_conv1(wild_x, wild_edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        h = h + x
        #gcnn_concat.append(x)
        x = self.wild_conv2(h, wild_edge_index)
        #x = self.dropout(x)
        #gcnn_concat.append(x)
        #x = self.wild_conv1(x, wild_edge_index)
        #gcnn_concat.append(x)
        #x = torch.cat(gcnn_concat, dim=1)

        # Global pooling
        #x = self.pool(x, wild_batch)
        x = gmp(x, wild_batch)
        #x = self.att(x, wild_batch)

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
        #gcnn_mut_concat.append(xt)
        xt = self.mutant_conv2(ht, mutant_edge_index)
        #xt = self.dropout(xt)
        #gcnn_mut_concat.append(xt)
        #xt = self.mutant_conv1(xt, mutant_edge_index)
        #gcnn_mut_concat.append(xt)
        #xt = torch.cat(gcnn_mut_concat, dim=1)

        # Global poolingi
        #xt = self.pool(x, mutant_batch)
        xt = gmp(xt, mutant_batch)
        #xt = self.att(xt, mutant_batch)

        # Flatten
        xt = self.relu(self.mutant_fc1(xt))
        xt = self.dropout(xt)
        #xt = self.mutant_ba1(xt)

        # Concatenation  
        xc = torch.cat((xt, x),1)

        # Dense layers
        out = self.wild_MLP(xc)

        return out
    
