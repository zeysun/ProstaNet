import numpy as np
import os
from tqdm import tqdm
import pathlib
from pathlib import Path

import json
import torch, math
import torch.nn.functional as F

import biographs as bg
from Bio.PDB.PDBParser import PDBParser

import torch
import networkx as nx
from torch_geometric.data import Data

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")

# Dictionary for getting Residue symbols
ressymbl = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU':'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN':'Q', 'ARG':'R', 'SER': 'S','THR': 'T', 'VAL': 'V', 'TRP':'W', 'TYR': 'Y'}

# List for getting single letter amino-acid codes
pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Both Micheletti and Acthely are Amino acids parameter sets
Micheletti = {'A': [0.1461, -0.2511, 0.3323, -0.1348, -0.0751, 0.5029, -0.2376, -0.3111, -0.2432, -0.2119, 0.0864, 0.1754, -0.1496, 0.5126, -0.5081, -0.1515, -0.0218, -0.9737, -0.0724, 0.3642],
              'R': [-0.2511, 0.9875, -0.6728, 0.1974, -0.6062, -0.121, -0.4586, 0.2466, 0.9985, 0.1034, -0.1302, 0.7273, -0.4676, 0.4855, -0.0067, -0.118, 0.3967, -1.4845, 0.4237, -0.5168],
              'N': [0.3323, -0.6728, -0.1962, 0.7855, 0.6139, 0.4502, -0.3154, -0.1649, 0.8099, 0.2317, -0.0605, 0.6158, 1.8413, 0.3461, 0.3707, 0.6249, -0.5914, -0.3028, -0.6968, -0.104],
              'D': [-0.1348, 0.1974, 0.7855, -0.0531, 0.2278, -0.1466, 0.2194, 0.1528, -0.2501, 0.2659, 0.0585, -0.0642, 0.1491, 0.4899, 0.0755, -0.1609, 0.2193, -0.7832, 0.0182, 0.0092],
              'C': [-0.0751, -0.6062, 0.6139, 0.2278, -0.2544, 0.1387, 0.2791, 0.1847, 5.4553, 0.2965, -0.0196, -0.604, 1.4331, -1.3925, -0.172, 0.1837, 0.262, -3.5239, 0.2585, 0.0296],
              'Q': [0.5029, -0.121, 0.4502, -0.1466, 0.1387, 0.8438, -0.5234, -0.0425, 0.5803, -0.1875, -0.4168, 0.2349, -0.2908, 0.379, 0.0525, -0.9002, 0.1006, 1.2075, -0.5137, 0.0029],
              'E': [-0.2376, -0.4586, -0.3154, 0.2194, 0.2791, -0.5234, 0.6456, -0.0113, -0.7232, 0.7647, -0.0453, -0.9604, 0.3231, -0.1143, 0.5402, 0.2888, 0.0948, -0.9357, 0.3261, 0.1387],
              'G': [-0.3111, 0.2466, -0.1649, 0.1528, 0.1847, -0.0425, -0.0113, 0.099, -0.0951, 0.0446, -0.1538, -0.1308, 0.2339, 0.0189, 0.9071, -0.3528, 0.1084, -1.2366, -0.0737, 0.1995],
              'H': [-0.2432, 0.9985, 0.8099, -0.2501, 5.4553, 0.5803, -0.7232, -0.0951, 0.1314, -0.0476, -0.4529, 0.2934, 3.1785, -0.019, -0.2032, 0.9858, -0.5871, -0.6739, 0.7276, -0.6893],
              'I': [-0.2119, 0.1034, 0.2317, 0.2659, 0.2965, -0.1875, 0.7647, 0.0446, -0.0476, 0.6801, -0.0782, 0.0855, -0.9283, -0.9792, 0.4353, 0.1538, -0.4179, 0.2734, -0.4792, 0.2618],
              'L': [0.0864, -0.1302, -0.0605, 0.0585, -0.0196, -0.4168, -0.0453, -0.1538, -0.4529, -0.0782, -0.0748, 0.2119, -0.2531, -0.2127, -0.5026, 0.1004, 0.377, 1.0659, 0.354, -0.194],
              'K': [0.1754, 0.7273, 0.6158, -0.0642, -0.604, 0.2349, -0.9604, -0.1308, 0.2934, 0.0855, 0.2119, 0.5109, -0.4667, -0.4479, 0.9888, 0.5015, -0.5895, -0.1668, 0.7956, -0.6987],
              'M': [-0.1496, -0.4676, 1.8413, 0.1491, 1.4331, -0.2908, 0.3231, 0.2339, 3.1785, -0.9283, -0.2531, -0.4667, 3.1655, 0.101, -0.8698, 0.2007, -0.219, 98.4886, -0.3258, -0.5331],
              'F': [0.5126, 0.4855, 0.3461, 0.4899, -1.3925, 0.379, -0.1143, 0.0189, -0.019, -0.9792, -0.2127, -0.4479, 0.101, -1.3128, -0.6986, -0.1223, 0.4102, 0.6057, -0.3256, 0.0008],
              'P': [-0.5081, -0.0067, 0.3707, 0.0755, -0.172, 0.0525, 0.5402, 0.9071, -0.2032, 0.4353, -0.5026, 0.9888, -0.8698, -0.6986, -0.3621, -0.3125, 0.5402, 1.3914, 0.0996, 0.1362],
              'S': [-0.1515, -0.118, 0.6249, -0.1609, 0.1837, -0.9002, 0.2888, -0.3528, 0.9858, 0.1538, 0.1004, 0.5015, 0.2007, -0.1223, -0.3125, -0.0802, -0.2393, -0.233, -0.1895, -0.0443],
              'T': [-0.0218, 0.3967, -0.5914, 0.2193, 0.262, 0.1006, 0.0948, 0.1084, -0.5871, -0.4179, 0.377, -0.5895, -0.219, 0.4102, 0.5402, -0.2393, 0.3269, 0.3848, -0.1235, 0.4075],
              'W': [-0.9737, -1.4845, -0.3028, -0.7832, -3.5239, 1.2075, -0.9357, -1.2366, -0.6739, 0.2734, 1.0659, -0.1668, 98.4886, 0.6057, 1.3914, -0.233, 0.3848, 13.1813, 0.3708, -0.1516],
              'Y': [-0.0724, 0.4237, -0.6968, 0.0182, 0.2585, -0.5137, 0.3261, -0.0737, 0.7276, -0.4792, 0.354, 0.7956, -0.3258, -0.3256, 0.0996, -0.1895, -0.1235, 0.3708, -0.7699, -0.2175],
              'V': [0.3642, -0.5168, -0.104, 0.0092, 0.0296, 0.0029, 0.1387, 0.1995, -0.6893, 0.2618, -0.194, -0.6987, -0.5331, 0.0008, 0.1362, -0.0443, 0.4075, -0.1516, -0.2175, 0.1445]}

Acthely = {'A':[-0.591, -1.302, -0.733, 1.57, -0.146],
           'C':[-1.343, 0.465, -0.862, -1.02, -0.255],
           'D':[1.05, 0.302, -3.656, -0.259, -3.242],
           'E':[1.357, -1.453, 1.477, 0.113, -0.837],
           'F':[-1.006, -0.59, 1.891, -0.397, 0.412],
           'G':[-0.384, 1.652, 1.33, 1.045, 2.064],
           'H':[0.336, -0.417, -1.673, -1.474, -0.078],
           'I':[-1.239, -0.547, 2.131, 0.393, 0.816],
           'K':[1.831, -0.561, 0.533, -0.277, 1.648],
           'L':[-1.019, -0.987, -1.505, 1.266, -0.912],
           'M':[-0.663, -1.524, 2.219, -1.005, 1.212],
           'N':[0.945, 0.828, 1.299, -0.169, 0.933],
           'P':[0.189, 2.081, -1.628, 0.421, -1.392],
           'Q':[0.931, -0.179, -3.005, -0.503, -1.853],
           'R':[1.538, -0.055, 1.502, 0.44, 2.897],
           'S':[-0.228, 1.399, -4.76, 0.67, -2.647],
           'T':[-0.032, 0.326, 2.213, 0.908, 1.313],
           'V':[-1.337, -0.279, -0.544, 1.242, -1.262],
           'W':[-0.595, 0.009, 0.672, -2.128, -0.184],
           'Y':[0.26, 0.83, 3.097, -0.838, 1.512]}

def _normalize(tensor, dim=-1):
     return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


class prot2graph():
    def __init__(self, root):
       self.root = root
       self.raw_paths = os.path.join(self.root, 'raw_com') # PDB files of proteins in datasets
       self.processed_paths = os.path.join(self.root, 'GCN_fea') # Output folder of structure graphs

    def process(self, datalist):
        data_list =[]
        count = 0
        with open(datalist) as f:
            data = json.load(f)
            for i in data:
                name = i['name']
                pdb = os.path.join('../LTJ_features/raw_com', f"{name}.pdb")
                # Folder of PSSM results of proteins in dataset
                pssm_file = os.path.join('/home/til60/Desktop/Blast/blast_result', f"{name}.pssm")
                with torch.no_grad():
                    coords = torch.as_tensor(i['coord'],device="cpu", dtype=torch.float32)

                    # Node features extraction
                    node_feats = self._get_seq_emb(i['seq'], pdb, pssm_file, coords)

                    # Adjacency matrix extraction
                    mat = self._get_adjacency(pdb)

                    # Edge index extraction
                    edge_index = self._get_edgeindex(pdb, mat)
            
                    data = Data(x = node_feats, edge_index = edge_index )
                    count += 1
                    data_list.append(data)
                    torch.save(data, self.processed_paths + "/"+ name +'.pt')

        self.data_prot = data_list
        print(data_list)
        print(count)

    # Constructs a modecular network and returns its adjacency matrix as a dense matrix
    def _get_adjacency(self, file):
        molecule = bg.Pmolecule(file)
        network = molecule.network()
        mat = nx.adjacency_matrix(network)
        m = mat.todense()
        return m
    
    # Finds non-zero elements in the adjacency matrix, representing edges
    def _get_edgeindex(self, file, adjacency_mat):
        edge_ind = []
        m = self._get_adjacency(file)

        a = np.nonzero(m > 0)[0]
        b = np.nonzero(m > 0)[1]
        edge_ind.append(a)
        edge_ind.append(b)
        return torch.tensor(np.array(edge_ind), dtype= torch.long)
    
    # Use biopython to get structure from a pdb file
    def _get_structure(self, file):
        parser = PDBParser()
        structure = parser.get_structure(id, file)
        return structure
    
    # Generate sequence from the protein 3D structure
    def _get_sequence(self, structure):
        sequence =""
        for model in structure:
          for chain in model:
            for residue in chain:
              if residue.get_resname() in ressymbl.keys():
                  sequence = sequence+ ressymbl[residue.get_resname()]
        return sequence
    
    # Use advanced amino acids encoding methods to extract features from residues
    def _get_seq_emb(self, sequence, file_path, pssm_file, coords):
       
       one_hot = self._get_one_hot_symbftrs(sequence)

       res_fea = self._get_res_ftrs(sequence)

       res_energy = self._get_res_energy(sequence)

       res_score = self._get_res_score(file_path)

       res_pssm = self._get_pssm(pssm_file)

       mask = torch.isfinite(coords.sum(dim=(1,2)))
       coords[~mask] = np.inf

       dihedrals = self._dihedrals(coords)
       
       # Return features representation concatenated by results of several encoding methods
       return torch.tensor(np.hstack((res_pssm, res_score, res_energy, res_fea, dihedrals)), dtype = torch.float)
    
    # One_hot encoding method
    def _get_one_hot_symbftrs(self, sequence):
        one_hot_symb = np.zeros((len(sequence),len(pro_res_table)))
        row= 0
        for res in sequence:
            col = pro_res_table.index(res)
            one_hot_symb[row][col]=1
            row +=1
        return torch.tensor(one_hot_symb, dtype= torch.float)
    
    # Physicochemical properties method by using Acthely factors
    def _get_res_ftrs(self, sequence):
        res_ftrs_out = []
        for res in sequence:
            res_ftrs_out.append(Acthely[res])
        res_ftrs_out= np.array(res_ftrs_out)
        return torch.tensor(res_ftrs_out, dtype = torch.float)
    
    # Structure-based encoding method by using Micheletti potentials
    def _get_res_energy(self, sequence):
        res_energy_out = []
        for res in sequence:
            res_energy_out.append(Micheletti[res])
        res_energy_out = np.array(res_energy_out)
        return torch.tensor(res_energy_out, dtype = torch.float)
    
    # Residue scoring encoding method by using Rosetta scoring function
    # each amino acid in sequence has 20 Rosetta scoring metrics
    def _get_res_score(self, pdb):
        scoring = False
        profile = []
        for line in open(pdb):
            if line.startswith("VRT"):
                scoring = False
            if scoring:
                data = [float(v) for v in line.split()[1:-1]]
                sele = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                sele_data = [data[i] for i in sele]
                profile.append(sele_data)
            if line.startswith("pose"):
               scoring = True
        return torch.tensor(profile, dtype = torch.float)
    
    # Evolution-based encoding method by using PSSM
    # the method used to generate pssm file for proteins is in the other file
    def _get_pssm(self, pssm_file):
        with open(pssm_file, 'r') as f:
            pssm = f.readlines()

        data = []
        for line in pssm[3:]:
            if len(line.split()) == 44:
                profile = []
                for v in line.split()[2:22]:
                    f = 1 / (1 + math.exp(-int(v)))
                    f = round(f, 4)
                    profile.append(f)
                data.append(profile)
        return torch.tensor(data, dtype = torch.float)
    
    def _dihedrals(self, X, eps=1e-7):
        # First 3 coordinates are N, CA, C
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        
        # Shifted slices of unit vectors
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [-1, 3])

        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features
    

if __name__ == '__main__':
   
   datalist = '../LTJ_features/json_path/structures_list.json'
   prot_graphs = prot2graph('../LTJ_features')
   prot_graphs.process(datalist)
