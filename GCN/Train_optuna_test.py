import torch
import math
from Metrics import *
import numpy as np
import optuna
from optuna.trial import TrialState

import torch.nn as nn
from Data_prepare import dir_dataset, trainset, valset
from Test_Data_prepare import dir_testset, in_testset
from Model_optuna_test import staGCNN
from torch_geometric.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset

trainsets = []
validatesets = []

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")

print("Datalength")
print(len(dir_dataset))

total_samples = len(dir_dataset)
n_iterations = math.ceil(total_samples/5)

#utilities
def train(model, device, trainloader, optimizer):
    
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

    labels_tr = labels_tr.detach().numpy()
    predictions_tr = predictions_tr.detach().numpy()
    train_A = get_accuracy(labels_tr, predictions_tr, 0.5)
    train_A_r = np.round(train_A, decimals=3)

def validate(model, device, validateloader):
 
   model.eval()
   predictions = torch.Tensor()
   labels = torch.Tensor()
   loss_func = nn.BCELoss()
   with torch.no_grad():
       for count, (wild, mutant, label) in enumerate(validateloader):
           wild = wild.to(device)
           mutant = mutant.to(device)
           output = model(wild, mutant)
           predictions = torch.cat((predictions, output.cpu()), 0)
           labels = torch.cat((labels, label.view(-1,1).cpu()), 0)
   
   loss = loss_func(predictions.float(), labels.float())
   loss_r = loss.detach().numpy()
   loss_r = np.round(loss_r, decimals=3)
   return loss_r

def test(model, device, testloader, tn):

  model.load_state_dict(torch.load(f"../LTJ_features/model/GCN144.pth"))
  model.eval()
  predictions = torch.Tensor()
  labels = torch.Tensor()
  with torch.no_grad():
      for count, (wild, mutant, label) in enumerate(testloader):
          wild = wild.to(device)
          mutant = mutant.to(device)
          output = model(wild, mutant)
          predictions = torch.cat((predictions, output.cpu()), 0)
          labels = torch.cat((labels, label.view(-1,1).cpu()), 0)
  
  labels = labels.detach().numpy()
  predictions = predictions.detach().numpy()
  test_A = get_accuracy(labels, predictions, 0.5)
  se = sensitivity(labels, predictions, 0.5)
  sp = specificity(labels, predictions, 0.5)
  roc = auroc(labels, predictions)
  f1 = f_score(labels, predictions, 0.5)
 
  avg_R = np.round(test_A, decimals=3)
  roc = np.round(roc, decimals=3)
  f1 = np.round(f1, decimals=3)
  print(f'se: {se}, sp: {sp}, roc: {roc}, f1: {f1}')
 
  return avg_R

def test_dir(model, device, testloader_dir, tn):

  model.load_state_dict(torch.load("../LTJ_features/model/GCN144.pth"))
  model.eval()
  predictions = torch.Tensor()
  labels = torch.Tensor()
  with torch.no_grad():
      for count, (wild, mutant, label) in enumerate(testloader_dir):
          wild = wild.to(device)
          mutant = mutant.to(device)
          output = model(wild, mutant)
          predictions = torch.cat((predictions, output.cpu()), 0)
          labels = torch.cat((labels, label.view(-1,1).cpu()), 0)

  labels = labels.detach().numpy()
  predictions = predictions.detach().numpy()
  test_A = get_accuracy(labels, predictions, 0.5)

  avg_R = np.round(test_A, decimals=3)
  return avg_R

def test_in(model, device, testloader_in, tn):

  model.load_state_dict(torch.load(f"../LTJ_features/model/GCN144.pth"))
  model.eval()
  predictions = torch.Tensor()
  labels = torch.Tensor()
  with torch.no_grad():
      for count, (wild, mutant, label) in enumerate(testloader_in):
          wild = wild.to(device)
          mutant = mutant.to(device)
          output = model(wild, mutant)
          predictions = torch.cat((predictions, output.cpu()), 0)
          labels = torch.cat((labels, label.view(-1,1).cpu()), 0)

  labels = labels.detach().numpy()
  predictions = predictions.detach().numpy()
  test_A = get_accuracy(labels, predictions, 0.5)

  avg_R = np.round(test_A, decimals=3)
  return avg_R

def objective(trial):

  batch_size = trial.suggest_categorical("batch", [8, 16, 32, 64])
  lr = trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3, 1e-2])
  wd = trial.suggest_categorical("wd", [1e-5, 1e-4, 1e-3])

  
  seed = 42
  torch.manual_seed(seed)
  
  n_epochs_stop = 10
  epochs_no_improve = 0
  early_stop = False
     
  trainloader = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
  validateloader = DataLoader(dataset=valset, batch_size=batch_size, num_workers=0, shuffle=True)

  testset = ConcatDataset([dir_testset, in_testset])
  testloader = DataLoader(dataset=testset, batch_size=64, num_workers=0, shuffle=True, drop_last=True)

  testloader_dir = DataLoader(dataset=dir_testset, batch_size=64, num_workers=0, shuffle=True, drop_last=True)
  testloader_in = DataLoader(dataset=in_testset, batch_size=64, num_workers=0, shuffle=True, drop_last=True)
  
  min_loss = 100
  best_loss = 100

  model = staGCNN(trial)
  model.to(device)
  num_epochs = 1000
  optimizer =  torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
  print(f"Trial {trial.number}: batch={batch_size}, lr={lr}, wd={wd}, dropout={model.p}, dense_layers={model.num_dense_layers}, output_dim={model.output_dim}, inner_dim={model.inner_dim}")
  
  #for epoch in range(num_epochs):
  #  train(model, device, trainloader, optimizer)
  #  val_loss = validate(model, device, validateloader)
     
    # Checkpoint
  #  if(val_loss < best_loss):
  #    best_loss = val_loss
  #    torch.save(model.state_dict(), f"../LTJ_features/model/GCN{trial.number}.pth") #path to save the model
      
  #  if(val_loss < min_loss):
  #    epochs_no_improve = 0
  #    min_loss = val_loss
  #    min_loss_epoch = epoch
  #  elif val_loss> min_loss :
  #    epochs_no_improve += 1
  #  if epoch > 20 and epochs_no_improve >= n_epochs_stop:
  #    early_stop = True
  #    break

  tn = trial.number
  test_R = test(model, device, testloader, tn)
  test_D = test_dir(model, device, testloader_dir, tn)
  test_I = test_in(model, device, testloader_in, tn)
  print(f'test_r {test_R}, test_d {test_D}, test_I {test_I}')

  trial.report(test_R, epoch)

  if trial.should_prune():
     raise optuna.exceptions.TrialPruned()
  
  return test_R

if __name__=="__main__":
   study = optuna.create_study(direction="maximize")

   study.optimize(objective, n_trials=150)

   pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
   complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

   print("Study statistics: ")
   print("  Number of finished trials: ", len(study.trials))
   print("  Number of pruned trials: ", len(pruned_trials))
   print("  Number of complete trials: ", len(complete_trials))
 
   print("Best trial:")
   trial = study.best_trial
 
   print("  Value: ", trial.value)
 
   print("  Params: ")
   for key, value in trial.params.items():
       print("    {}: {}".format(key, value))
