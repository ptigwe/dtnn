import torch.nn as nn

from . import torch_geom, vanilla


def init_weights(m):
    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight)
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: 
            m.bias.data.fill_(0)
