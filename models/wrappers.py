import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class Wrapper(nn.Module):
    """
    Wrapper to perform the rearrangment of MPOSE 2021 :'(
    """
    def __init__(self, num_kpoint = 13):
        super().__init__()
        self.rearrange = nn.Sequential(
            Rearrange('b t (x y) -> b x y t',x = num_kpoint),
            Rearrange('b p d t -> b d t p 1')
        )
        
    def forward(self, x):
        x = self.rearrange(x)

        return x