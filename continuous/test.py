from os.path import join, dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import jackal_navi_envs
from jackal_navi_envs.jackal_env_wrapper import *

import gym
import numpy
import torch
from torch import nn

import argparse
from datetime import datetime
import time
import os
import json

from policy import TD3Policy
from tianshou.utils.net.common import Net
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.data import Batch

parser = argparse.ArgumentParser(description = 'Jackal navigation simulation')
parser.add_argument('--model', dest = 'model', type = str, default = 'results/DQN_testbed_2020_08_30_10_58', help = 'path to the saved model and configuration')
parser.add_argument('--policy', dest = 'policy', type = str, default = 'policy_26.pth')
parser.add_argument('--record', dest='record', action='store_true')
parser.add_argument('--default', dest='default', action='store_true')
parser.add_argument('--gui', dest='gui', action='store_true')
parser.add_argument('--seed', dest='seed', type = int, default = 43)
parser.add_argument('--avg', dest='avg', type = int, default = 2)
parser.add_argument('--world', dest = 'world', type = str, default = 'Benchmarking/train/world_1.world')
parser.add_argument('--noise', dest='noise', action='store_true')

args = parser.parse_args()
model_path = args.model
record = args.record
gui = 'true' if args.gui else 'false'
seed = args.seed
avg = args.avg
default = args.default
policy = args.policy
world = args.world
noise = args.noise

config_path = model_path + '/config.json'
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

env = wrapper_dict[wrapper_config['wrapper']](gym.make('jackal_continuous-v0', **env_config), **wrapper_config['wrapper_args'])
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net(training_config['num_layers'], state_shape, device=device, hidden_layer_size=training_config['hidden_size'])
actor = Actor(
    net, action_shape,
    1, device, hidden_layer_size=training_config['hidden_size']
).to(device)
actor_optim = torch.optim.Adam(actor.parameters(), lr=training_config['actor_lr'])
net = Net(training_config['num_layers'], state_shape,
          action_shape, concat=True, device=device, hidden_layer_size=training_config['hidden_size'])
critic1 = Critic(net, device, hidden_layer_size=training_config['hidden_size']).to(device)
critic1_optim = torch.optim.Adam(critic1.parameters(), lr=training_config['critic_lr'])
critic2 = Critic(net, device, hidden_layer_size=training_config['hidden_size']).to(device)
critic2_optim = torch.optim.Adam(critic2.parameters(), lr=training_config['critic_lr'])
policy = TD3Policy(
    actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
    action_range=[env.action_space.low, env.action_space.high],
    tau=training_config['tau'], gamma=training_config['gamma'],
    exploration_noise=GaussianNoise(sigma=training_config['exploration_noise']),
    policy_noise=training_config['policy_noise'],
    update_actor_freq=training_config['update_actor_freq'],
    noise_clip=training_config['noise_clip'],
    reward_normalization=training_config['rew_norm'],
    ignore_done=training_config['ignore_done'],
    estimation_step=training_config['n_step'])
print(training_config['hidden_size'])
state_dict = torch.load(model_path)
policy.load_state_dict(state_dict)

if not noise:
    policy._noise = None
print(env.action_space.low, env.action_space.high)

range_dict = {
    'max_vel_x': [0.1, 2],
    'max_vel_theta': [0.314, 3.14],
    'vx_samples': [1, 12],
    'vtheta_samples': [1, 40],
    'path_distance_bias': [0.1, 1.5],
    'goal_distance_bias': [0.1, 2],
    'inflation_radius': [0.1, 0.6]
}

# from matplotlib import pyplot as plt

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
        obs_batch = Batch(obs=[obs], info={})
        # obs = torch.tensor([obs]).float()
        if not default:
            actions = policy(obs_batch).act.cpu().detach().numpy().reshape(-1)
        else:
            actions = np.array([1, 1.57, 6, 20, 0.75, 1])
        obs_new, rew, done, info = env.step(actions)
        obs = obs_new
        # plt.plot(obs)
        # plt.show()

        print('current step: %d, X position: %f, Y position: %f, rew: %f' %(count, info['X'], info['Y'] , rew))
        print(info['params'])
        params = np.array(info['params'])
        pms = np.append(pms, np.expand_dims(params, -1), -1)
        r += rew
        count += 1
    if count != env_config['max_step'] and rew != -500:
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
