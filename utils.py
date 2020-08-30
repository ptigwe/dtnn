import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def pad_(X, max_atoms, dim=1):
    extra = max_atoms - X.shape[0]
    return F.pad(X, (0, extra) * dim).clone()

def create_dummy(num_atoms, total_atoms):
    Z = torch.LongTensor(num_atoms).random_(total_atoms)
    D = torch.rand((num_atoms, num_atoms))
    D.masked_fill_(torch.eye(num_atoms).bool(), 0)
    return Z + 1, (D + D.T) / 2

def create_dummy_batch(min_atoms, max_atoms, total_atoms, bs):
    Zs, Ds, sizes = [], [], []
    for num_atoms in torch.randint(min_atoms, max_atoms, (bs,)):
        Z, D = create_dummy(num_atoms.item(), total_atoms)
        Zs.append(pad_(Z, max_atoms))
        Ds.append(pad_(D, max_atoms, 2))
        sizes.append(num_atoms)
    Zs = torch.stack(Zs)
    Ds = torch.stack(Ds)
    return Zs, Ds, torch.LongTensor(sizes)

def transform_D(D, sz):
    shape = list(D.shape) + [sz]
    return D.unsqueeze(-1).expand(shape)

def create_mask(method):
    def fn(sizes, full_size):
        masks = []

        for size in sizes:
            masks.append(method(size.item(), full_size))

        return torch.stack(masks) if len(sizes) > 1 else mask
    return fn

@create_mask
def mask_2d(size, full_size):
    mask = torch.zeros((full_size, full_size))
    mask[np.diag_indices(size)] = 1
    mask[:size, :size] -= 1
    mask.abs()
    return mask

@create_mask
def mask_1d(size, full_size):
    mask = torch.zeros((full_size,))
    mask[:size] = 1
    return mask

def read_raw_data():
    with open('data/preprocessed.pkl', 'rb') as f:
        data = pkl.load(f)
    return data

def process_data(data, max_atoms):
    res = {}
    
    for smile, (Z, D) in data.items():
        res[smile] = (pad_(torch.LongTensor(Z), max_atoms), pad_(torch.FloatTensor(D), max_atoms, 2))
    
    return res
