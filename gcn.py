import torch
import torch.nn as nn 
import numpy as np
from models.agcn import ACGN
from models.aagcn import AACGN
from models.wrappers import Wrapper

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on: {device}")

# Load Data

X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_test  = np.load('data/X_test.npy')
y_test  = np.load('data/y_test.npy') 

# GNN model
GNN = ACGN(graph="graph.mpose.Graph")
#GNN = AACGN(graph="graph.mpose.Graph")
#GNN = DSTANet()
#GNN = MSG3D(graph="graph.mpose.Graph")

save_checkpoint(GNN, "checkpoints/out.pt")


x = torch.from_numpy(X_train[0]).float()
x = torch.reshape(x, (1,30, 26)).to(device)

model = nn.Sequential(
    Wrapper(num_kpoint=13),
    GNN
).to(device)
out = model(x)

print(out.shape)