from os.path import join, dirname, abspath, exists
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

import numpy as np
import random
from continuous.utils import load_appli_data, load_appld_data, preprocessing, load_config

from tianshou.utils.net.common import Net
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.continuous import Actor, ActorProb, Critic
from tianshou.data import Batch
from tianshou.data import to_torch
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_path = "continuous/configs/td3_condor.json"
config = load_config(config_path)
env_config = config['env_config']
training_config = config['training_config']

state_shape = np.array([721])
action_shape = np.array([8])

range_dict = {
    'max_vel_x': [0.2, 2],
    'max_vel_theta': [0.314, 3.14],
    'vx_samples': [4, 20],
    'vtheta_samples': [4, 60],
    'occdist_scale': [0.05, 1],
    'path_distance_bias': [0.1, 1.5],
    'goal_distance_bias': [0.1, 2],
    'inflation_radius': [0.01, 0.5]
}
action_space_low = np.array([range_dict[pn][0] for pn in env_config['param_list']])
action_space_high = np.array([range_dict[pn][1] for pn in env_config['param_list']])
action_bias = torch.tensor((action_space_low + action_space_high) / 2.0, device=device)
action_scale = torch.tensor((action_space_high - action_space_low) / 2.0, device=device)

def output_to_action(outputs):
    actions = to_torch(outputs, device=device, dtype=torch.float32)
    actions *= action_scale
    actions += action_bias
    return actions

def action_to_output(actions):
    outputs = to_torch(actions, device=device, dtype=torch.float32)
    outputs -= action_bias
    outputs /= action_scale
    return outputs

seed = 13
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

ix, iy = load_appli_data()
dx, dy = load_appld_data()
x = np.concatenate([ix, dx])
y = np.concatenate([iy, dy])
xx, yy = preprocessing(x, y, 4)
yy = action_to_output(yy)

net = Net(training_config['num_layers'], state_shape, device=device, hidden_layer_size=training_config['hidden_size'])

if config['section'] == 'SAC':
    actor = ActorProb(
        net, action_shape,
        1, device, hidden_layer_size=training_config['hidden_size']
    ).to(device)
else:
    actor = Actor(
        net, action_shape,
        1, device, hidden_layer_size=training_config['hidden_size']
    ).to(device)

optimizer = torch.optim.Adam(actor.parameters(), lr=0.000001)
loss_func = torch.nn.MSELoss()

xx = to_torch(xx, device=device, dtype=torch.float32)
yy = to_torch(yy, device=device, dtype=torch.float32)
idx = list(range(xx.shape[0]))
idx_test = idx[::5]
idx_train = [i for i in idx if i not in idx_test]

xx_train = xx[idx_train]
yy_train = yy[idx_train]

xx_test = xx[idx_test]
yy_test = yy[idx_test]

for e in range(40):
    test_loss = np.inf
    for t in range(2000):
        idx = list(range(xx_train.shape[0]))
        random.shuffle(idx)
        xxx = xx_train[idx]
        yyy = yy_train[idx]
        outputs, _ = actor(xxx)
        loss = loss_func(outputs, yyy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Train: epoch: %d, step: %d, loss: %f" %(e+1, t+1, loss), end="\r")

    with torch.no_grad():
        outputs, _ = actor(xx_test)
        loss = loss_func(outputs, yy_test)
        print("\nTest: epoch: %d, loss: %f" %(e+1, loss))
        if loss < test_loss:
            torch.save(actor.state_dict(), "actor.pth")
            test_loss = loss
