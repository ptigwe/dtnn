import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle as pkl
import pytorch_lightning as pl
import utils
import data

max_atoms = 30

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
        mask = utils.mask_2d(sizes, max_atoms)
        mask = mask.to(X.device)
        return (mask.unsqueeze(-1) * X).sum(-3)

class MDTNN(nn.Module):
    def __init__(self, basis, num_atoms, num_gauss, hidden, T=3):
        super().__init__()
        self.basis = basis
        self.T = T
        
        self.C_embed = nn.Embedding(num_atoms + 1, basis)
        self.df = nn.Linear(num_gauss, basis)
        self.interaction = InteractionBlock(basis, basis)
        self.mlp = nn.Sequential(nn.Linear(basis, hidden),
                                 nn.Tanh(),
                                 nn.Linear(hidden, 1))
    
    def forward(self, Z, D, sizes):
        C = self.C_embed(Z)
        d_hat = self.df(D)
        
        for _ in range(self.T):
            C = C + self.interaction(C, d_hat, sizes)
            
        E = self.mlp(C).squeeze()
        mask = utils.mask_1d(sizes, max_atoms)
        mask = mask.to(E.device)
        return (mask * E).sum(-1)#.squeeze()


def init_weights(m):
    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight)
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: 
            m.bias.data.fill_(0)

