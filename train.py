import models
import utils
import data
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

min_atoms = 5
max_atoms = 30
num_atoms = 110
mu_min = -1
mu_max = 10
delta_mu = 0.2
basis = 30
num_gauss = int((mu_max - mu_min) / delta_mu) + 1
hidden = 15


class DTNNModule(pl.LightningModule):
    def __init__(self, basis, num_atoms, num_gauss, hidden, target):
        super().__init__()
        self.dtnn = models.MDTNN(basis, num_atoms, num_gauss, hidden)
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


def main():
    trainer = pl.Trainer()
    model = DTNNModule(basis, num_atoms, num_gauss, hidden, 'E1-CC2')
    model.apply(models.init_weights)
    trainer.fit(model)

if __name__ == '__main__':
    main()
