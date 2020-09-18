import itertools

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


class QM8(InMemoryDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform, None)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ["data/sdf.json"]
    
    @property
    def processed_file_names(self):
        return ['processed.pt']
    
    def process(self):
        data_df = pd.read_json(self.raw_file_names[0],
                               lines=True, nrows=None)
        data_list = []
        for i, row in data_df.iterrows():
            edges = np.array(list(itertools.permutations(range(len(row.Z)), 2)))
            Z = torch.LongTensor(row.Z)
            
            D = np.array(data.get_distance_matrix(row, 'is_3D'))
            D_hat = data.gaussian_expansion(D, -1, 0.2, 1, 0.2)
            D = np.array([D_hat[x, y, :] for x, y in edges])
            
            d = Data(Z=Z,
                     edge_index=torch.LongTensor(edges.T),
                     num_nodes=len(Z),
                     edge_attr=torch.FloatTensor(D),
                     y=torch.FloatTensor([row[['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2']]]))
            print(d)
            data_list.append(d)
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), f'processed/{self.processed_file_names[0]}')


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
        self.dataset = QM8('')
        size = len(self.dataset)

        if os.path.isfile('data/split.pkl'):
            with open('data/split.pkl', 'rb') as f:
                split_dict = pkl.load(f)
        else:
            split_dict = utils.create_random_split(size)

        self.train_dataset = self.dataset[split_dict['train']]
        self.valid_dataset = self.dataset[split_dict['val']]
        self.test_dataset = self.dataset[split_dict['test']]
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-4)
        
    def train_dataloader(self):
        return DataLoader(self.train_data, 32)

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
