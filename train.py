from argparse import ArgumentParser
import os
import models
import utils
import data
import pickle as pkl
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import wandb
import pytorch_lightning as pl


class DTNNModule(pl.LightningModule):
    def __init__(self, basis=30, hidden=15, target='E1-CC2', dist_method='euclid',
                 target_type='single', mu_max=10, mu_min=-1, delta_mu=0.2,
                 sigma=0.2, num_workers=8, learning_rate=1e-4,
                 fname='data/rdkit_bound.json',
                 split_file='data/split.pkl',
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        num_gauss = 1 + int((self.hparams.mu_max - self.hparams.mu_min)
                             / self.hparams.delta_mu)

        self.dtnn = models.MDTNN(self.hparams.basis, data.NUM_ATOMS,
                                 num_gauss, self.hparams.hidden, 3,
                                 len(self.hparams.target), self.hparams.target_type)

    
    def forward(self, Z, D, sizes):
        return self.dtnn(Z, D, sizes)
    
    def prepare_data(self):
        self.dataset = data.QM8Dataset(self.hparams.fname, self.hparams.target,
                                       data.MAX_ATOMS, self.hparams.mu_min,
                                       self.hparams.delta_mu, self.hparams.mu_max,
                                       nrows=None, dist_method=self.hparams.dist_method)
        size = len(self.dataset)

        if os.path.isfile(self.hparams.split_file):
            with open(self.hparams.split_file, 'rb') as f:
                split_dict = pkl.load(f)
        else:
            split_dict = utils.create_random_split(size)

        self.train_dataset = Subset(self.dataset, split_dict['train'])
        self.valid_dataset = Subset(self.dataset, split_dict['val'])
        self.test_dataset = Subset(self.dataset, split_dict['test'])
    
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
        
        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
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
        parser.add_argument('--target', type=list, default=['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2'])
        parser.add_argument('--target_type', type=str, default='single')
        parser.add_argument('--dist_method', type=str, default='euclid')
        parser.add_argument('--mu_max', type=float, default=10)
        parser.add_argument('--mu_min', type=float, default=-1)
        parser.add_argument('--delta_mu', type=float, default=0.2)
        parser.add_argument('--sigma', type=float, default=0.2)
        parser.add_argument('--batch_size', type=int, default=50)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--fname', type=str, default='data/sdf.json')
        parser.add_argument('--split_file', type=str, default='data/split.pkl')
        return parser


def main():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DTNNModule.add_model_specific_args(parser)
    args = parser.parse_args()

    model = DTNNModule(**vars(args))
    model.apply(models.init_weights)
    print(model)

    checkpoint_cbk = pl.callbacks.ModelCheckpoint('checkpoints/{epoch}_{val_loss:.2f}',
                                                     save_top_k=1,
                                                     verbose=True,
                                                     monitor='val_loss',
                                                     mode='min')
    early_stop_cbk = pl.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=True)
    wandb_logger = pl.loggers.WandbLogger(name=f'{args.target_type}_{args.dist_method}',
                                          project='DTNN')
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger,
                                            checkpoint_callback=checkpoint_cbk,
                                            early_stop_callback=early_stop_cbk)

    trainer.fit(model)
    trainer.test(model)
    wandb.save(checkpoint_cbk.best_model_path)

if __name__ == '__main__':
    main()
