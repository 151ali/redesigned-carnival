import torch
import torch.nn as nn 
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
import numpy as np
import random
import wandb
from utils import *

from models.agcn import ACGN
from models.aagcn import AACGN
from models.wrappers import Wrapper
from models.msg3d.msg3d import MSG3D


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on: {device}")

# Hyperparameters
learning_rate  = 0.1
batch_size     = 4
num_epochs     = 70


# Load Data
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_test  = np.load('data/X_test.npy')
y_test  = np.load('data/y_test.npy') 

# Dataset
train_x = torch.Tensor(X_train) # transform to torch tensor
train_y = torch.Tensor(y_train)

test_x = torch.Tensor(X_test)
test_y = torch.Tensor(y_test)

train_dataset = TensorDataset(train_x, train_y) # create your datset
test_dataset = TensorDataset(test_x, test_y)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# GNN model
GNN = ACGN(graph="graph.mpose.Graph")
#GNN = AACGN(graph="graph.mpose.Graph")
#GNN = DSTANet()
#GNN = MSG3D(graph="graph.mpose.Graph")

#x = torch.from_numpy(X_train[0]).float()
#x = torch.reshape(x, (1,30, 26)).to(device)

# Initialize the model
model = nn.Sequential(Wrapper(num_kpoint=13),GNN).to(device)

# Deterministic behavior
seed_all()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'max')

experiment = f"gnn-based"

wandb.init(
    project  = "master-thesis-2022",
    entity   = "151ali",
    group    = experiment,
)

# Training loop

for epoch in range(num_epochs):
    num_correct = 0
    num_samples = 0
    losses = []

    model.train()

    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device).squeeze(1)
        targets = targets.type(torch.LongTensor).to(device=device) #  PyTorch won't accept a `FloatTensor` as categorical target

        # forward
        scores = model(data)
        _, predictions = scores.max(1)

        #print(predictions.get_device())
        #print(targets.get_device())

        num_correct += (predictions == targets).sum()
        num_samples += predictions.size(0)

        running_acc = num_correct / num_samples
        wandb.log({"train_accuracy": running_acc})

        loss = criterion(scores, targets)
        wandb.log({"train_loss": loss.item()})

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent update step/adam step
        optimizer.step()

    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)

    # evaluation per epoth
    model.eval()
    vnum_correct = 0
    vnum_samples = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device=device).squeeze(1)
            y = y.type(torch.LongTensor).to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            vnum_correct += (predictions == y).sum()
            vnum_samples += predictions.size(0)
        
        val_acc = vnum_correct / vnum_samples

    wandb.log({"val_acc": val_acc.item()})


# print(f"Accuracy on test set: {check_accuracy(test_loader, model, device)*100:.2f}")
