import os
import utils
import data
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from torch_geometric.nn import global_add_pool, MessagePassing
from torch_geometric.data import Data, DataLoader, InMemoryDataset


class InteractionBlockMLP(nn.Module):
    def __init__(self, basis, num_gauss, hidden):
        super().__init__()
        self.cf = nn.Linear(basis, hidden)
        self.df = nn.Linear(num_gauss, hidden)
        self.fc = nn.Linear(hidden, basis, False)
    
    def forward(self, c, d):
        return torch.tanh(self.fc(self.cf(c) * self.df(d)))


class InteractionBlock(MessagePassing):
    def __init__(self, basis, num_gauss, hidden, **kwargs):
        super().__init__(**kwargs)
        self.inter_blk = InteractionBlockMLP(basis, num_gauss, hidden)
    
    def forward(self, x, edge_index, edge_attr):
        return x + self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        return self.inter_blk(x_j, edge_attr)


class MLP(nn.Module):
    def __init__(self, in_size, h_size, out_size):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_size, h_size),
                                 nn.Tanh(),
                                 nn.Linear(h_size, out_size))
    
    def forward(self, X):
        return self.mlp(X)


class DTNN(nn.Module):
    def __init__(self, basis, num_atoms, num_gauss, hidden, T=3, target_sz=1,
                 target_type='single', **kwargs):
        super().__init__(**kwargs)
        self.embed = nn.Embedding(num_atoms + 1, basis)
        self.inter_blk = InteractionBlock(basis, num_gauss, basis)
        self.readout_mlp = MLP(basis, hidden, 4)
        self.T = T
        
    def forward(self, data):
        C = self.embed(data.Z) 
        
        for _ in range(self.T):
            C = self.inter_blk(C, data.edge_index, data.edge_attr)
        
        return global_add_pool(self.readout_mlp(C), data.batch)
    
    def message(self, x_i, x_j, edge_attr):
        return self.inter_blk(x_j, edge_attr)
