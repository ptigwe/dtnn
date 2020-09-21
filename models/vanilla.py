import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle as pkl
import pytorch_lightning as pl
import utils
import data


class InteractionBlock(nn.Module):
    def __init__(self, basis, hidden):
        super().__init__()
        self.cf = nn.Linear(basis, hidden)
        self.fc = nn.Linear(hidden, basis, False)
    
    def forward(self, C, D_hat, sizes):
        X = self.cf(C)
        X = X.unsqueeze(-2) * D_hat
        X = torch.tanh(self.fc(X))
        
        num_batch = C.shape[0] if len(C.shape) > 2 else 1
        mask = utils.mask_2d(sizes, data.MAX_ATOMS)
        mask = mask.to(X.device)
        return (mask.unsqueeze(-1) * X).sum(-3)

class MLP(nn.Module):
    def __init__(self, basis, hidden, target):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(basis, hidden),
                                 nn.Tanh(),
                                 nn.Linear(hidden, target))

    def forward(self, C):
        return self.mlp(C)

class TargetLayer(nn.Module):
    def __init__(self, basis, hidden, target_sz, target_type):
        super().__init__()
        self.target_type = target_type
        if target_type == 'single':
            self.mlp = MLP(basis, hidden, target_sz)
        else:
            self.mlps = nn.ModuleList([MLP(basis, hidden, 1) for _ in range(target_sz)])

    def forward(self, C):
        if self.target_type == 'single':
            return self.mlp(C)
        else:
            return torch.cat([mlp(C) for mlp in self.mlps], 2)

class MDTNN(nn.Module):
    def __init__(self, basis, num_atoms, num_gauss, hidden, T=3, target_sz=1,
                 target_type='single'):
        super().__init__()
        self.basis = basis
        self.T = T
        
        self.C_embed = nn.Embedding(num_atoms + 1, basis)
        self.df = nn.Linear(num_gauss, basis)
        self.interaction = InteractionBlock(basis, basis)
        self.target = TargetLayer(basis, hidden, target_sz, target_type)
    
    def forward(self, Z, D, sizes):
        C = self.C_embed(Z)
        d_hat = self.df(D)
        
        for _ in range(self.T):
            C = C + self.interaction(C, d_hat, sizes)
            
        E = self.target(C)#.squeeze()
        mask = utils.mask_1d(sizes, data.MAX_ATOMS).unsqueeze(-1)
        mask = mask.to(E.device)
        return (mask * E).sum(1)#.squeeze()
