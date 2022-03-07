import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    LSTM many to one
    """
    def __init__(
        self,
        in_features     : int = 26,
        num_poses: int = 30,
        num_classes    : int = 20,

        hidden_size    : int = 128,
        num_layers     : int = 8,
        bidirectional  : bool=False,
        device="cpu"
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        self.lstm = nn.LSTM(in_features, hidden_size, num_layers, batch_first=True,bidirectional=bidirectional)
        self.fc   = nn.Linear(hidden_size * num_poses, num_classes)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x,(h0, c0))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    x = torch.randn(1, 30, 26)
    model = LSTM()
    print(model(x).shape)