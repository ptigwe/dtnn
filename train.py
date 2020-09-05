from argparse import ArgumentParser
import models
import utils
import data
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


class DTNNModule(pl.LightningModule):
    def __init__(self, basis=30, hidden=15, target='E1-CC2', dist_method='euclid',
                 mu_max=10, mu_min=-1, delta_mu=0.2, sigma=0.2,
                 num_workers=8, learning_rate=1e-4,
                 fname='data/rdkit_bound.json',
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        num_gauss = 1 + int((self.hparams.mu_max - self.hparams.mu_min)
                             / self.hparams.delta_mu)

        self.dtnn = models.MDTNN(self.hparams.basis, data.NUM_ATOMS,
                                 num_gauss, self.hparams.hidden, 3,
                                 len(self.hparams.target), 'multiple')

    
    def forward(self, Z, D, sizes):
        return self.dtnn(Z, D, sizes)
    
    def prepare_data(self):
        self.dataset = data.QM8Dataset(self.hparams.fname, self.hparams.target,
                                       data.MAX_ATOMS, self.hparams.mu_min,
                                       self.hparams.delta_mu, self.hparams.mu_max,
                                       nrows=1000, dist_method=self.hparams.dist_method)
        size = len(self.dataset)
        test_size = int(size * 0.2)
        sizes = [size - 2*test_size, test_size, test_size]
        self.train_dataset, self.test_dataset, self.valid_dataset = random_split(self.dataset, sizes)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)
    
    def step(self, batch, batch_idx, loss_fn):
        Z, D, sizes, target = batch
        predict = self.forward(Z, D, sizes)
        loss = F.smooth_l1_loss(predict, target, reduction='none')
        losses = loss.mean(0)
        losses = {target: losses[i] for i, target in enumerate(self.hparams.target)}
        return loss.mean(), losses
    
    def training_step(self, batch, batch_idx):
        loss, losses = self.step(batch, batch_idx, F.mse_loss)
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)
        result.log_dict({'train_loss': loss})
        result.log_dict({f'train_{target}': val for target, val in losses.items()})
        return result
    
    def validation_step(self, batch, batch_idx):
        loss, losses = self.step(batch, batch_idx, F.l1_loss)
        
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({'val_loss': loss})
        result.log_dict({f'val_{target}': val for target, val in losses.items()})
        return result
    
    def test_step(self, batch, batch_idx):
        result = self.validation_step(batch, batch_idx)
        result.rename_keys({'val_loss': 'test_loss'})
        result.rename_keys({f'val_{target}': f'test_{target}' for target in self.hparams.target})
        return result
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--basis', type=int, default=30)
        parser.add_argument('--hidden', type=int, default=15)
        parser.add_argument('--target', type=str, default=['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2'])
        parser.add_argument('--dist_method', type=str, default='euclid')
        parser.add_argument('--mu_max', type=float, default=1)
        parser.add_argument('--mu_min', type=float, default=-1)
        parser.add_argument('--delta_mu', type=float, default=0.2)
        parser.add_argument('--sigma', type=float, default=0.2)
        parser.add_argument('--batch_size', type=int, default=50)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--fname', type=str, default='data/rdkit_euclid.json')
        return parser


def main():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DTNNModule.add_model_specific_args(parser)
    args = parser.parse_args()

    model = DTNNModule(**vars(args))
    model.apply(models.init_weights)

    wandb_logger = pl.loggers.WandbLogger(name='TestRun', project='DTNN')
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)
    trainer.fit(model)
    trainer.test(model)

if __name__ == '__main__':
    main()
