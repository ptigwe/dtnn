import itertools

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
    def __init__(self, basis, hidden):
        super().__init__()
        self.cf = nn.Linear(basis, hidden)
        self.df = nn.Linear(basis, hidden)
        self.fc = nn.Linear(hidden, basis, False)
    
    def forward(self, c, d):
        return torch.tanh(self.fc(self.cf(c) * self.df(d)))


class InteractionBlock(MessagePassing):
    def __init__(self, basis, hidden, **kwargs):
        super().__init__(**kwargs)
        self.inter_blk = InteractionBlockMLP(basis, hidden)
    
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
    def __init__(self, basis, hidden, T=3, **kwargs):
        super().__init__(**kwargs)
        self.embed = nn.Embedding(10, basis)
        self.inter_blk = InteractionBlock(basis, basis)
        self.readout_mlp = MLP(basis, hidden, 4)
        self.T = T
        
    def forward(self, data):
        C = self.embed(data.Z) 
        
        for _ in range(self.T):
            C = self.inter_blk(C, data.edge_index, data.edge_attr)
        
        return global_add_pool(self.readout_mlp(C), data.batch)
    
    def message(self, x_i, x_j, edge_attr):
        return self.inter_blk(x_j, edge_attr)


class DTNNModule(pl.LightningModule):
    def __init__(self, basis, hidden, T):
        super().__init__()
        self.dtnn = DTNN(basis, hidden, T)
        
    def forward(self, data):
        return self.dtnn(data)
    
    def prepare_data(self):
        self.dataset = data.GraphQM8('')
        size = len(self.dataset)

        if os.path.isfile('data/split.pkl'):
            with open('data/split.pkl', 'rb') as f:
                split_dict = pkl.load(f)
        else:
            split_dict = utils.create_random_split(size)

        self.train_dataset = self.dataset[list(split_dict['train'])]
        self.valid_dataset = self.dataset[list(split_dict['val'])]
        self.test_dataset = self.dataset[list(split_dict['test'])]
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-4)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 32)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, 32)

    def test_dataloader(self):
        return DataLoader(self.test_dataloader, 32)

    def training_step(self, batch, batch_idx):
        y_pred = self.forward(batch)
        loss = F.l1_loss(batch.y, y_pred)
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        result.log_dict({'train_loss': loss})
        return result

    def validation_step(self, batch, batch_idx):
        y_pred = self.forward(batch)
        loss = F.l1_loss(batch.y, y_pred)
        result =  pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log_dict({'val_loss': loss})
        return result

    def test_step(self, batch, batch_idx):
        result = self.validation_step(batch, batch_idx)
        result.rename_keys({'val_loss': 'test_loss'})
        return result


def main():
    trainer = pl.Trainer()
    model = DTNNModule(11, 5, 3)
    trainer.fit(model)

if __name__ == '__main__':
    main()
