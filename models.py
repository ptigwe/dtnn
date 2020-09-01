import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle as pkl
import pytorch_lightning as pl
import utils
import data

min_atoms = 5
max_atoms = 30
num_atoms = 110
mu_min = -1
mu_max = 10
delta_mu = 0.2
basis = 30
num_gauss = int((mu_max - mu_min) / delta_mu) + 1
hidden = 15

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


class DTNNModule(pl.LightningModule):
    def __init__(self, basis, num_atoms, num_gauss, hidden, target):
        super().__init__()
        self.dtnn = MDTNN(basis, num_atoms, num_gauss, hidden)
        self.target = target
    
    def forward(self, Z, D, sizes):
        return self.dtnn(Z, D, sizes)
    
    def prepare_data(self):
        self.dataset = data.QM8Dataset('data/sdf.json', self.target, max_atoms, mu_min, delta_mu, mu_max, nrows=100, dist_method='graph')
        size = len(self.dataset)
        test_size = int(size * 0.2)
        sizes = [size - 2*test_size, test_size, test_size]
        self.train_dataset, self.test_dataset, self.valid_dataset = random_split(self.dataset, sizes)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 15)
    
    def step(self, batch, batch_idx, loss_fn):
        Z, D, sizes, target = batch
        predict = self.forward(Z, D, sizes)
        loss = loss_fn(predict, target)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, F.mse_loss)
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        result.log_dict({'train_loss': loss})
        return result
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, 50)
    
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, F.l1_loss)
        
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({'val_loss': loss})
        return result
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 50)
    
    def test_step(self, batch, batch_idx):
        result = self.validation_step(batch, batch_idx)
        result.rename_keys({'val_loss': 'test_loss'})
        return result
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-4)

def init_weights(m):
    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight)
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: 
            m.bias.data.fill_(0)
