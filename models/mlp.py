import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class MLP(nn.Module):
    def __init__(
        self,
        num_poses    : int = 30, 
        in_features    : int  = 26,
        hidden_features: int = 768,
        num_classes    : int = 20,

        device="cpu"
    ):
        super(MLP, self).__init__()
        
        self.projection = nn.Sequential(
            Rearrange('b s f -> b (s f)'),
        )
        self.fc1 = nn.Linear(num_poses * in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes)
    def forward(self, x):
        
        x = self.projection(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    x = torch.randn(1, 30, 26)
    model = MLP()
    print(model(x).shape)