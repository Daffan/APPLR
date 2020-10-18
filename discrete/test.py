from os.path import join, dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import jackal_navi_envs
from jackal_navi_envs.jackal_env_wrapper import *
from policy import DuelingDQN

import gym
import numpy
import torch
from torch import nn

import argparse
from datetime import datetime
import time
import os
from os.path import join
import json

parser = argparse.ArgumentParser(description = 'Jackal navigation simulation')
parser.add_argument('--model', dest = 'model', type = str, default = 'results/DQN_testbed_2020_08_30_10_58', help = 'path to the saved model and configuration')
parser.add_argument('--policy', dest = 'policy', type = str, default = 'policy_26.pth')
parser.add_argument('--record', dest='record', action='store_true')
parser.add_argument('--gui', dest='gui', action='store_true')
parser.add_argument('--seed', dest='seed', type = int, default = 43)
parser.add_argument('--avg', dest='avg', type = int, default = 2)
parser.add_argument('--default', dest='default', action='store_true')
parser.add_argument('--world', dest = 'world', type = str, default = 'Benchmarking/train/world_1.world')



args = parser.parse_args()
model_path = args.model
record = args.record
gui = 'true' if args.gui else 'false'
seed = args.seed
avg = args.avg
policy = args.policy
world = args.world
default = args.default

config_path = join(model_path, 'config.json')
model_path = join(model_path, policy)

with open(config_path, 'rb') as f:
    config = json.load(f)

env_config = config['env_config']
env_config['world_name'] = world
env_config['gui'] = gui
wrapper_config = config['wrapper_config']
training_config = config['training_config']

if record:
    env_config['world_name'] = env_config['world_name'].split('.')[0] + '_camera' + '.world'
'''
class DuelingDQN(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_layer = [64, 64], cnn = True, feature_layer = [256]):
        super().__init__()
        if cnn:
            self.feature = nn.Sequential(
                nn.Linear(720, feature_layer[0]), nn.ReLU(inplace=True)
                )
            feature_shape = feature_layer[0] + 7
        else:
            self.feature = lambda x: x.view(x.shape[0], -1)
            feature_shape = state_shape

        layers = [np.prod(feature_shape)] + hidden_layer
        self.value = []
        self.advantage = []
        for i, o in zip(layers[:-1], layers[1:]):
            self.value.append(nn.Linear(i, o))
            self.value.append(nn.ReLU(inplace=True))
            self.advantage.append(nn.Linear(i, o))
            self.advantage.append(nn.ReLU(inplace=True))
        self.advantage.append(nn.Linear(o, np.prod(action_shape)))
        self.value.append(nn.Linear(o, 1))

        self.value = nn.Sequential(*self.value)
        self.advantage = nn.Sequential(*self.advantage)

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        laser = obs.view(batch, 1, -1)[:,:,:720]
        params = obs.view(batch, -1)[:, 720:]

        embedding = self.feature(laser).view(batch, -1)
        feature = torch.cat((embedding, params), dim = 1)

        advantage = self.advantage(feature)
        value = self.value(feature)
        logits = value + advantage - advantage.mean(1, keepdim=True)
        return logits, state
'''
state_dict = {}
state_dict_raw = torch.load(model_path)
for key in state_dict_raw.keys():
    if key.split('.')[0] == 'model':
        state_dict[key[6:]] = state_dict_raw[key]

env = wrapper_dict[wrapper_config['wrapper']](gym.make('jackal_discrete-v0', **env_config), **wrapper_config['wrapper_args'])
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
model = DuelingDQN(state_shape, action_shape, hidden_layer = training_config['hidden_layer'], cnn = training_config['cnn'])
model.load_state_dict(state_dict)
model = model.float()

range_dict = {
    'max_vel_x': [0.1, 2],
    'max_vel_theta': [0.314, 3.14],
    'vx_samples': [1, 12],
    'vtheta_samples': [1, 40],
    'path_distance_bias': [0.1, 1.5],
    'goal_distance_bias': [0.1, 2]
}

rs = []
cs = []
pms = np.array(env_config['param_init'])
pms = np.expand_dims(pms, -1)
succeed = 0
for i in range(avg):
    print(">>>>>>>>>>>>>>>>>>>>>> Running: %d/%d <<<<<<<<<<<<<<<<<<<<<<<<<<" %(i+1, avg))
    r = 0
    count = 0
    obs = env.reset()
    done = False
    while not done:
        obs = torch.tensor([obs]).float()
        actions = model(obs)[0].detach().numpy()[0]
        action = np.argmax(actions.reshape(-1))
        if default:
            action = list(range(len(actions)))[-1]
        obs, reward, done, info = env.step(action)
        print('current step: %d, X position: %f, Y position: %f, rew: %f' %(count, info['X'], info['Y'] , reward))
        print(info['params'])
        params = np.array(info['params'])
        pms = np.append(pms, np.expand_dims(params, -1), -1)
        r += reward
        count += 1
    if count != env_config['max_step'] and reward != -1000:
        succeed += 1
        rs.append(r)
        cs.append(count)
print("succeed: %d/%d \t episode reward: %.2f \t steps: %d" %(succeed, avg, sum(rs)/float((len(rs))), sum(cs)/float((len(cs)))))

env.close()

# from matplotlib import pyplot as plt
# fig, axe = plt.subplots(6, 1, figsize = (6,18))

# for i in range(6):
#     axe[i].plot(pms[i, :])
#     axe[i].set_ylabel(env_config['param_list'][i])
# plt.show()


######## About recording ###########
# Add the camera model to the world you used to train
# Check onetwofive_meter_split_camera.world
# The frames will be save to folder /tmp/camera_save_tutorial
# run
# ffmpeg -r 30 -pattern_type glob -i '/tmp/camera_save_tutorial/default_camera_link_my_camera*.jpg' -c:v libx264 my_camera.mp4
# to generate the video
