from torch.utils.data import Dataset, DataLoader
import torch
import utils
import pandas as pd
import numpy as np

class QM8Dataset(Dataset):
    def __init__(self, target, max_atoms):
        df = pd.read_pickle('data/preprocessed_df.pkl')
        self.target = torch.FloatTensor(df[target])
        Zs, Ds, sizes = [], [], []
        for i, x in df.iterrows():
            Zs.append(utils.pad_(torch.LongTensor(x.Z), max_atoms))
            Ds.append(utils.pad_(torch.FloatTensor(x.D), max_atoms, 2))
            sizes.append(len(x.Z))
        self.Zs = torch.stack(Zs)
        self.Ds = torch.stack(Ds)
        self.sizes = torch.LongTensor(sizes)
        
    def __getitem__(self, idx):
        return self.Zs[idx], self.Ds[idx], self.sizes[idx], self.target[idx]
    
    def __len__(self):
        return len(self.Zs)
