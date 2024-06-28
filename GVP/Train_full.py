import torch
import math
from Metrics import *
import numpy as np

import torch.nn as nn
from Data_prepare import dir_dataset, in_dataset
from Model_GVP import StaGVP
from torch_geometric.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")

print("Datalength")
print(len(dir_dataset))

total_samples = len(dir_dataset)
n_iterations = math.ceil(total_samples/5)

#utilities
def train(model, device, trainloader, optimizer, epoch):
    
    print(f'Training on {len(trainloader)} samples.....')
    model.train()
    loss_func = nn.BCELoss()
    predictions_tr = torch.Tensor()
    labels_tr = torch.Tensor()

    for count,(wild, mutant, label) in enumerate(trainloader):
        wild = wild.to(device)
        mutant = mutant.to(device)
        optimizer.zero_grad()
        output = model(wild, mutant)
        predictions_tr = torch.cat((predictions_tr, output.cpu()), 0)
        labels_tr = torch.cat((labels_tr, label.view(-1,1).cpu()), 0)
        loss = loss_func(output, label.view(-1,1).float().to(device))
        loss.backward()
        optimizer.step()

    loss = loss_func(predictions_tr.float(), labels_tr.float())
    loss_r = loss.detach().numpy()
    loss_r = np.round(loss_r, decimals=3)
    labels_tr = labels_tr.detach().numpy()
    predictions_tr = predictions_tr.detach().numpy()
    train_A = get_accuracy(labels_tr, predictions_tr, 0.5)
    train_A_r = np.round(train_A, decimals=3)
    print(f'Epoch {epoch-1} / {num_epochs} [==============================] - train_loss : {loss_r} - train_A : {train_A_r}')
    return loss_r

seed = 42
torch.manual_seed(seed)

n_epochs_stop = 10
epochs_no_improve = 0
early_stop = False


trainset = ConcatDataset([dir_dataset, in_dataset])
   
trainloader = DataLoader(dataset=trainset, batch_size=16, num_workers=0, shuffle=True)


min_loss = 100
best_loss = 100

model = StaGVP((91, 3), (182, 6), (32, 1), (32, 1))
model.to(device)
num_epochs = 1000
loss_func = nn.BCELoss()
optimizer =  torch.optim.Adam(model.parameters(), lr= 0.00001, weight_decay=0.001)
  
for epoch in range(num_epochs):
  train_loss = train(model, device, trainloader, optimizer, epoch+1)
     
    # Checkpoint
  if(train_loss < best_loss):
    best_loss = train_loss
    best_loss_epoch = epoch
    torch.save(model.state_dict(), f"../../LTJ_features/GVP.pth") #path to save the model
    print("Model")
    
  if(train_loss < min_loss):
    epochs_no_improve = 0
    min_loss = train_loss
    min_loss_epoch = epoch
  elif train_loss> min_loss :
    epochs_no_improve += 1
  if epoch == 190:
    print('Early stopping!' )
    early_stop = True
    break
