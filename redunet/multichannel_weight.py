import time
import torch
import torch.nn as nn
from opt_einsum import contract



class MultichannelWeight(nn.Module):
    def __init__(self, channels, *dimension, dtype=torch.complex64):
        super(MultichannelWeight, self).__init__()
        self.weight = nn.Parameter(torch.randn(channels, channels, *dimension, dtype=dtype))
        self.shape = self.weight.shape
        self.dtype = dtype

    def __getitem__(self, item):
    	return self.weight[item]

    def forward(self, V):
        start = time.time()
        out =  contract("bi...,ih...->bh...", V.type(self.dtype), self.weight.conj())
        end = time.time()
        #print('multichannel weight forward', end - start)
        return out
        # return torch.einsum("bi...,ih...->bh...", V.type(self.dtype), self.weight.conj())
