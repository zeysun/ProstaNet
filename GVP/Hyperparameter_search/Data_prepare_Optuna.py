import os
import torch
import glob
import numpy as np
import math
from torch.utils.data import Dataset as Dataset_n
from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset


processed_dir = "../../LTJ_features/processed_GVP/" #the folder stores structure graphs
npy_file = "../../LTJ_features/Q2278_direct.npy" #dataset file


'''
Creating training datasets for hyperparameters search of ProstaNet
'''
class dir_Dataset(Dataset_n):
    def __init__(self, npy_file, processed_dir):
      self.npy_ar = np.loadtxt(npy_file, dtype=str)
      self.processed_dir = processed_dir
      self.wild = self.npy_ar[:,0]
      self.mutant = self.npy_ar[:,5]
      self.label = self.npy_ar[:,7].astype(float)
      self.n_samples = self.npy_ar.shape[0]

    def __len__(self):
       return(self.n_samples)
    
    def __getitem__(self, index):
       wild = os.path.join(self.processed_dir, self.wild[index]+".pt")
       mutant = os.path.join(self.processed_dir, self.mutant[index]+".pt")
       wild = torch.load(glob.glob(wild)[0])
       mutant = torch.load(glob.glob(mutant)[0])
       return(wild, mutant, torch.tensor(self.label[index])) #for each mutation return structure graphs of its wild-type and mutant, and its label
    
class in_Dataset(Dataset_n):
    def __init__(self, npy_file, processed_dir):
      self.npy_ar = np.loadtxt(npy_file, dtype=str)
      self.processed_dir = processed_dir
      self.wild = self.npy_ar[:,5]
      self.mutant = self.npy_ar[:,0]
      self.label = self.npy_ar[:,6].astype(float)
      self.n_samples = self.npy_ar.shape[0]

    def __len__(self):
       return(self.n_samples)
    
    def __getitem__(self, index):
       wild = os.path.join(self.processed_dir, self.wild[index]+".pt")
       mutant = os.path.join(self.processed_dir, self.mutant[index]+".pt")
       wild = torch.load(glob.glob(wild)[0])
       mutant = torch.load(glob.glob(mutant)[0])
       return(wild, mutant, torch.tensor(self.label[index]))

final_pairs = np.loadtxt(npy_file, dtype=str)
size = final_pairs.shape[0]
size = 2 * size
    
dir_dataset = dir_Dataset(npy_file = npy_file ,processed_dir= processed_dir)
in_dataset = in_Dataset(npy_file = npy_file ,processed_dir= processed_dir)

dataset = ConcatDataset([dir_dataset, in_dataset]) #training set contains direct and their corresponding reverse mutations
trainset, valset = torch.utils.data.random_split(dataset, [math.floor(0.8 * size), size - math.floor(0.8 * size) ]) #split the dataset into 8:2 for training and validation
