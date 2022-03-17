import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


import torch
import torch.nn as nn 
import numpy as np




from models.msg3d.msg3d import MSG3D

# Load Data
data = np.load('data/data.npy')
labels = np.load('data/annotations.npy')
x = torch.from_numpy(data[0]).float()
x = torch.reshape(x, (1,30, 26))

# GNN model

GNN = MSG3D(graph="graph.mpose.Graph")


num_kpoint = 13
model = nn.Sequential(
    Rearrange('b t (x y) -> b x y t',x = num_kpoint),
    Rearrange('b p d t -> b d t p 1'),
    GNN
)

print(model(x).shape)



