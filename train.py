import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

import wandb

from utils import *

from tqdm import tqdm
import numpy as np
import random
import argparse


# from torchsummary import summary

par = argparse.ArgumentParser(description='Training.')
par.add_argument('--model', type=str, default='lstm', help='Train LSTM model.')
par.add_argument('--lr', type=float, default=0.005, help='Learning rate.')
par.add_argument('--epoch', type=int, default=150, help='Number of epochs')
par.add_argument('--batch', type=int, default=8, help='Batch size.')
par.add_argument('--depth', type=int, default=1, help='Depth of the architectre.')


# for deterministic behavior ( REPRODUCIBILITY )
seed = 7
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

torch.manual_seed(seed) # seed the RNG for all devices (both CPU and CUDA)
np.random.seed(seed)
random.seed(seed)

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" running on: {device}")


# parameters
args = par.parse_args()
model_name     = args.model
learning_rate  = args.lr
batch_size     = args.batch
num_epochs     = args.epoch
depth          = args.depth


in_features   = 26
num_poses     = 30
num_classes   = 20



# model
model = generate_model(
    model_name,
    depth,
    num_poses     =  num_poses,
    in_features   = in_features, 
    num_classes   = num_classes,
    device = device
)
# model.to(device)
# get_num_params(model)
# print(model)

# Distributed GPU training if CUDA can detect more than 1 GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model, device_ids=cuda_devices)

print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))


# Loss 
# Using Croos EntropyLoss for classification
criterion = nn.CrossEntropyLoss()
# Using Negative Log Likelihood Loss function for predicting the masked_token
# criterion = nn.NLLLoss(ignore_index=0)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# x = torch.randn(1, 30, 26)
# print(model(x).shape)

# Save samples as numpy arrays
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_test  = np.load('data/X_test.npy')
y_test  = np.load('data/y_test.npy') 
#print(X_train.shape, X_test.shape)


train_x = torch.Tensor(X_train) # transform to torch tensor
train_y = torch.Tensor(y_train)

test_x = torch.Tensor(X_test)
test_y = torch.Tensor(y_test)

train_dataset = TensorDataset(train_x, train_y) # create your datset
test_dataset = TensorDataset(test_x, test_y)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# PS:
#  One hot encoding is a good trick to be aware of in PyTorch,
#  but it is important to know that you do not actually need this 
#  if you are building a classifier with cross entropy loss. 
#  In that case, just pass the class index targets into the loss function
#  and PyTorch will take care of the rest.

experiment = f"experiment_depth_{depth}"

wandb.init(
    project  = "master-thesis-2022",
    entity   = "151ali",
    group    = experiment,
    job_type = "eval"
)

wandb.config.update(args)
# TRAIN


for epoch in range(num_epochs):
    num_correct = 0
    num_samples = 0

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


        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent update step/adam step
        optimizer.step()

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
