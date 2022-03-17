import torch
import numpy as np

from  models.lstm import *
from  models.mlp import *
from  models.vit import *


def generate_model(name, depth, **kwargs):
    if name =="lstm":
    
        model = LSTM(
            **kwargs,

            hidden_size   = hidden_size,
            num_layers    = depth,
            bidirectional = False,
        )
    elif name =="vit":
        model = ViT(
            **kwargs,
            emb_size      = 768,
            depth         = depth,
            
        )
    elif name =="mlp":
        model = MLP(
            **kwargs,
            hidden_features = depth,
        )
    else:
        return None

    return model

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            print(x)
            print(y)
            exit(0)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    # Toggle model back to train 
    model.train()
    return num_correct / num_samples


def get_num_params(net):
    """
    TODO : review this function
    """
    modules = [module for module in net.modules()]
    params = [param.shape for param in net.parameters()]

    # Print Model Summary
    print(modules[0])
    total_params=0
    for i in range(0,len(modules)-1):
        j = 2*i
        param = (params[j-2][1]*params[j-2][0])+params[j-1][0]
        total_params += param
        #print("Layer",i,"->\t",end="")
        #print("Weights:", params[j-2][0],"x",params[j-2][1],"\tBias: ",params[j-1][0], "\tParameters: ", param)
    print("\nTotal Params: ", total_params)


def save_checkpoint(model, out_path):
    torch.save(model.state_dict(), out_path)


def load_checkpoint(model, chekpoint_path):
    return model.load_state_dict(torch.load(chekpoint_path))