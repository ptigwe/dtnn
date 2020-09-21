import torch
import utils
import numpy as np
import pandas as pd

from enum import Enum
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from torch.utils.data import DataLoader, Dataset

NUM_ATOMS=9
MAX_ATOMS=27
TARGET=['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2']

def gaussian_expansion(D, mu_min=-1, delta_mu=0.2, mu_max=10, sigma=0.2):
    mu = np.arange(mu_min, mu_max + delta_mu, delta_mu)
    diff = D[:,:,np.newaxis] - mu[np.newaxis, np.newaxis, :]
    return np.exp(-diff ** 2 / (2 * sigma))

def get_distance_matrix(X, dist_method):
    assert(dist_method in ('is_3D', 'euclid', 'graph'))

    if dist_method == 'is_3D':
        return X.euclid_D if X.is_3D else X.graph_D

    return X[f'{dist_method}_D']

class QM8Dataset(Dataset):
    def __init__(self, fname, target=TARGET, max_atoms=MAX_ATOMS, mu_min=-1, delta_mu=0.2,
                 mu_max=10, sigma=0.2, nrows=None, dist_method='euclid'):
        df = pd.read_json(fname, lines=True, orient='records', nrows=nrows)
        self.target = torch.FloatTensor(df[target].values)
        Zs, Ds, sizes = [], [], []

        for i, x in df.iterrows():
            Zs.append(utils.pad_(torch.LongTensor(x.Z), max_atoms))
            D = np.array(get_distance_matrix(x, dist_method))
            D_hat = gaussian_expansion(D, mu_min, delta_mu, mu_max, sigma)
            Ds.append(utils.pad_Dhat(torch.FloatTensor(D_hat), max_atoms))
            sizes.append(len(x.Z))

        self.Zs = torch.stack(Zs)
        self.Ds = torch.stack(Ds)
        self.sizes = torch.LongTensor(sizes)
        
    def __getitem__(self, idx):
        return self.Zs[idx], self.Ds[idx], self.sizes[idx], self.target[idx]
    
    def __len__(self):
        return len(self.Zs)

class GraphQM8(InMemoryDataset):
    def __init__(self, fname, target=TARGET, max_atoms=MAX_ATOMS, mu_min=-1, delta_mu=0.2,
                 mu_max=10, sigma=0.2, nrows=None, dist_method='euclid'):
        super().__init__('', None, None)

        self.fname = fname
        self.target = target
        self.max_atoms = max_atoms
        self.mu_min = mu_min
        self.delta_mu = delta_mu
        self.mu_max = mu_max
        self.sigma = sigma
        self.nrows = nrows
        self.dist_method = dist_method

        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return [self.fname]
    
    @property
    def processed_file_names(self):
        return ['processed.pt']
    
    def process(self):
        data_df = pd.read_json(self.raw_file_names[0],
                               lines=True, nrows=self.nrows)
        data_list = []
        for i, row in data_df.iterrows():
            edges = np.array(list(itertools.permutations(range(len(row.Z)), 2)))
            Z = torch.LongTensor(row.Z)
            
            D = np.array(data.get_distance_matrix(row, 'is_3D'))
            D_hat = gaussian_expansion(D, self.mu_min, self.delta_mu,
                                       self.mu_max, self.sigma)
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


