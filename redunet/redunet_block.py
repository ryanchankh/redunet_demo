import torch.nn as nn
from redunet import (
    Vector,
    Fourier1D,
    Fourier2D,
    ReduNet
)


def ReduNetVector(num_classes: int, 
                  dimension: int, 
                  num_layers: int = 5,
                  lift_dim: int = 0,
                  eta: float = 0.5,
                  eps: float = 0.1,
                  lmbda: float = 500
                  ) -> ReduNet:
    layers = []
    if lift_dim > 0:
      layers = [nn.Linear(dimension, lift_dim, bias=False)]
      dimension = lift_dim
    layers += [Vector(eta, eps, lmbda, num_classes, dimension) for _ in range(num_layers)]
    return ReduNet(*layers)

def ReduNet1D(num_classes: int, 
              channels: int,
              time: int,
              num_layers: int = 5,
              eta: float = 0.5,
              eps: float = 0.1,
              lmbda: float = 500
              ) -> ReduNet:
    layers = [Fourier1D(eta, eps, lmbda, num_classes, (channels, time)) for _ in range(num_layers)]
    return ReduNet(*layers)

def ReduNet2D(num_classes: int, 
              channels: int,
              height: int,
              width: int,
              num_layers: int = 5,
              eta: float = 0.5,
              eps: float = 0.1,
              lmbda: float = 500
              ) -> ReduNet:
    layers = [Fourier2D(eta, eps, lmbda, num_classes, (channels, height, width)) for _ in range(num_layers)]
    return ReduNet(*layers)