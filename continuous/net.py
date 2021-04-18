import torch
from torch import nn
from tianshou.data import to_torch
import numpy as np

class Net(nn.Module):
    def __init__(self, num_layers=2, state_shape=728, action_shape=0, concat=False, hidden_layer_size=512, device="cpu"):
        super().__init__()
        self.device = device
        state_shape = 728
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=3),
            nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(kernel_size = 3)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(), nn.BatchNorm1d(64), 
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(), nn.BatchNorm1d(64), 
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64)
        )

        feature_shape = 967 + np.prod(action_shape)
        layers = [feature_shape] + [hidden_layer_size]*num_layers
        self.last_linear = []
        for i, o in zip(layers[:-1], layers[1:]):
            print(i, o)
            self.last_linear.append(nn.Linear(i, o))
            self.last_linear.append(nn.ReLU(inplace=True))
        self.last_linear = nn.Sequential(*self.last_linear)

    def forward(self, obs, state=None, info={}):
        obs = to_torch(obs, device=self.device, dtype=torch.float32)
        obs = obs.reshape(obs.size(0), -1)

        batch_size = obs.shape[0]
        laser = obs.view(batch_size, 1, -1)[:,:,:721]
        params = obs.view(batch_size, -1)[:,721:]

        embedding1 = self.block1(laser)
        embedding2 = self.block2(embedding1)
        
        y = torch.cat((embedding1, embedding2), dim = 2)
        y = nn.ReLU()(y)
        # embedding3 = self.block3(y)

        # y = torch.cat((embedding2, embedding3), dim = 2)
        y = nn.ReLU()(y)
        y = nn.AvgPool1d(10)(y)
        y = y.view(batch_size, -1)

        feature = torch.cat((y, params), dim = 1)
        feature = self.last_linear(feature)
        
        return feature, state




