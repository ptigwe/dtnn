from torch.utils.data import Dataset, DataLoader
import torch
import utils
import pandas as pd
import numpy as np
from enum import Enum

NUM_ATOMS=110
MAX_ATOMS=27

def gaussian_expansion(D, mu_min=-1, delta_mu=0.2, mu_max=1, sigma=0.2):
    mu = np.arange(mu_min, mu_max + delta_mu, delta_mu)
    diff = D[:,:,np.newaxis] - mu[np.newaxis, np.newaxis, :]
    return np.exp(-diff ** 2 / (2 * sigma))

def get_distance_matrix(X, dist_method):
    assert(dist_method in ('is_3D', 'euclid', 'graph'))

    if dist_method == 'is_3D':
        return X.euclid_D if X.is_3D else X.graph_D

    return X[f'{dist_method}_D']

class QM8Dataset(Dataset):
    def __init__(self, fname, target, max_atoms, mu_min=-1, delta_mu=0.2,
                 mu_max=1, sigma=0.2, nrows=None, dist_method='euclid'):
        df = pd.read_json(fname, lines=True, orient='records', nrows=nrows)
        self.target = torch.FloatTensor(df[target])
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
