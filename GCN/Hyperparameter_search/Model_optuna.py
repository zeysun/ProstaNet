import torch
import torch.nn as nn
from Layer_optuna import GCN, MLP
from torch_geometric.nn import global_max_pool as gmp, global_add_pool as gap, global_mean_pool as gep, AttentionalAggregation
from torch_geometric.nn.norm import GraphNorm
from torch.nn import Linear


'''
Architecture of GCN-based method for hyperparameter search
'''
class staGCNN(nn.Module):
    def __init__(self, trial):


        super(staGCNN, self).__init__()

        n_output=1 #dimension of output
        num_features_pro = 71 #dimension of input features

        '''
        The hyperparameters fine-tuned in GCN-based method

        p: dropout rate
        num_dense_layers: number of last dense layers
        output_dim: output dimension of dense layer after pooling layer
        inner_dim: dimension of inner layers in the final dense layers
        '''
        self.p = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.num_dense_layers = trial.suggest_categorical("num_dense_layers", [1, 2, 3])
        self.output_dim = trial.suggest_categorical("output_dim", [35, 71, 142])
        self.inner_dim = trial.suggest_categorical("inner_dim", [71, 35, 17])
        

        # For wild_type
        self.n_output = n_output
        self.wild_conv1 = GCN(num_features_pro, num_features_pro) #GCN layers processing wild-type protein graphs
        self.wild_conv2 = GCN(num_features_pro, num_features_pro)
        self.wild_fc1 = nn.Linear(num_features_pro, self.output_dim) #dense layer processing result from pooling layer
        self.wild_fc2 = nn.Linear(num_features_pro, num_features_pro)
        self.wild_ba1 = nn.BatchNorm1d(self.output_dim)

        # For mutant
        self.mutant_conv1 = GCN(num_features_pro, num_features_pro)
        self.mutant_conv2 = GCN(num_features_pro, num_features_pro)
        self.mutant_fc1 = nn.Linear(num_features_pro, self.output_dim)
        self.mutant_fc2 = nn.Linear(num_features_pro, num_features_pro)
        self.mutant_ba1 = nn.BatchNorm1d(self.output_dim)

        self.relu = nn.ReLU() #activation function
        self.dropout = nn.Dropout(self.p)

        # Combined layers
        self.wild_MLP = MLP(2*self.output_dim, n_output, self.inner_dim, self.num_dense_layers, self.p)

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
    
