import os
import torch
import glob
import numpy as np
import math
from torch.utils.data import Dataset as Dataset_n
from torch_geometric.data import DataLoader


processed_dir = "../../LTJ_Test_features/processed_GVP/" #the folder stores structure graphs of testing data
npy_file = "../../LTJ_Test_features/Ssym_direct.npy" #dataset file of testing data

'''
Creating testing datasets of direct and reverse mutations for cross-validation
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

dir_testset = dir_Dataset(npy_file = npy_file ,processed_dir= processed_dir)
in_testset = in_Dataset(npy_file = npy_file ,processed_dir= processed_dir)
