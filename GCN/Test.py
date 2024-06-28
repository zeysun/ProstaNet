import torch
import numpy as np
import pandas as pd
from Metrics import *
from Test_Data_prepare import dir_testset, in_testset
from Model_mulGCN import staGCNN
from torch.utils.data import ConcatDataset
from torch_geometric.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

'''
Evaluate the accuracy of GCN-based method models from cross-validation

Return testing accuracy and loss for Ssym dataset
'''
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")
model = staGCNN()

i = 0
total_loss = 0
total_R = 0

# Anti-symmetry augmented Ssym dataset used as the testing set for cross validation
testset = ConcatDataset([dir_testset, in_testset])
testloader = DataLoader(dataset=testset, batch_size=8, num_workers=0, shuffle=True)

test_label = []
test_pred = []

for i in range(1, 6):
    # Calculate accuracy for the prediction of five models output from cross-validation
    model.load_state_dict(torch.load(f"../LTJ_features/GCN{i}.pth"))
    model.to(device)
    model.eval()
    predictions = torch.Tensor()
    labels = torch.Tensor()
    loss_func = nn.BCELoss()
    with torch.no_grad():
        for count, (wild, mutant, label) in enumerate(testloader):
            wild = wild.to(device)
            mutant = mutant.to(device)
            output = model(wild, mutant)
            predictions = torch.cat((predictions, output.cpu()), 0)
            labels = torch.cat((labels, label.view(-1,1).cpu()), 0)
    
    loss = loss_func(predictions.float(), labels.float())
    loss_r = loss.detach().numpy()
    labels = labels.detach().numpy()
    predictions = predictions.detach().numpy()
    test_A = get_accuracy(labels, predictions, 0.5)
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    print(test_A)

    total_loss += loss_r
    total_R += test_A

avg_loss = np.round(total_loss/5, decimals=3)
avg_A = np.round(total_R/5, decimals=3)

print(f'test_loss : {avg_loss} - test_R : {avg_A}')
