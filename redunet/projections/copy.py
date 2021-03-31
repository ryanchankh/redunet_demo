import torch.nn as nn


class Copy(nn.Module):
    def __init__(self, ncopies):
        super(Copy, self).__init__()
        self.ncopies = ncopies
    
    def forward(self, X):
        return X.repeat(1, self.ncopies, 1, 1)