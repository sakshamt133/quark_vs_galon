import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
             nn.Linear(2, 64),
             nn.BatchNorm1d(64),
             nn.ReLU(),
             nn.Linear(64, 128),
             nn.BatchNorm1d(128),
             nn.ReLU(),
             nn.Linear(128, 64),
             nn.ReLU(),
             nn.Linear(64, 32),
             nn.ReLU(),
             nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.model(x)