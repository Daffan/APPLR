import os
import json
import pickle
from os.path import join, dirname, abspath, exists
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
from os.path import join, dirname, abspath

from continuous.net import Net as Net1
from tianshou.utils.net.common import Net as Net2
from tianshou.utils.net.continuous import Actor, Critic
import torch

a = torch.rand(1,728)
net1 = Net1()
net2 = Net2(2, 728, device = "cpu", hidden_layer_size = 128)

b1, _ = net1(a)
b2, _ = net2(a)

print(b1.shape)
print(b2.shape)
device = "cpu"
actor = Actor(
        net1, 7,
            1, device, hidden_layer_size=512
        ).to(device)
action, _ = actor(a)
print(action.shape)
