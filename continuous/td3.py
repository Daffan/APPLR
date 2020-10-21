from os.path import join, dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import jackal_navi_envs

import gym
import numpy as np
try:
    sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
except:
    pass
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from policy import TD3Policy
from tianshou.utils.net.common import Net
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.data import Collector, ReplayBuffer, PrioritizedReplayBuffer
from collector import Collector as Fake_Collector
from offpolicy import offpolicy_trainer

sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')

import pickle
import argparse
import json
from datetime import datetime
import os

parser = argparse.ArgumentParser(description = 'Jackal navigation simulation')
parser.add_argument('--config', dest = 'config_path', type = str, default = 'configs/dqn.json', help = 'path to the configuration file')
parser.add_argument('--save', dest = 'save_path', type = str, default = 'results/', help = 'path to the saving folder')

args = parser.parse_args()
config_path = args.config_path
save_path = args.save_path

# Load the config files
with open(config_path, 'rb') as f:
    config = json.load(f)

env_config = config['env_config']
wrapper_config = config['wrapper_config']
training_config = config['training_config']

# Config logging
now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M")
save_path = os.path.join(save_path, config['section'] + "_" + dt_string)
if not os.path.exists(save_path):
    os.mkdir(save_path)
writer = SummaryWriter(save_path)
with open(os.path.join(save_path, 'config.json'), 'w') as fp:
    json.dump(config, fp)

# initialize the env --> num_env can only be one right now
wrapper_dict = jackal_navi_envs.jackal_env_wrapper.wrapper_dict
if not config['use_container']:
    if config['env'] == 'jackal':
        env = wrapper_dict[wrapper_config['wrapper']](gym.make('jackal_continuous-v0', **env_config), **wrapper_config['wrapper_args'])
    else:
        env = gym.make('Pendulum-v0')
    train_envs = DummyVectorEnv([lambda: env for _ in range(1)])
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
else:
    train_envs = config
    Collector = Fake_Collector

    state_shape = np.array((728,)) if config['env'] == 'jackal' else np.array((3,))
    action_shape = np.array((7,)) if config['env'] == 'jackal' else np.array((1,))

# config random seed
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])
if not config['use_container']:
    train_envs.seed(config['seed'])
'''
net = Net(training_config['layer_num'], state_shape, action_shape, config['device']).to(config['device'])
optim = torch.optim.Adam(net.parameters(), lr=training_config['learning_rate'])
'''
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

action_space_low = np.array([0.1, 0.314, 4., 8., 0.1, 0.1, 0.1]) if config['env'] == 'jackal' else np.array([-2])
action_space_high = np.array([2., 3.14, 12., 40., 1.5, 2., 0.6]) if config['env'] == 'jackal' else np.array([2])
policy = TD3Policy(
    actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
    action_range=[action_space_low, action_space_high],
    tau=training_config['tau'], gamma=training_config['gamma'],
    exploration_noise=GaussianNoise(sigma=training_config['exploration_noise']),
    policy_noise=training_config['policy_noise'],
    update_actor_freq=training_config['update_actor_freq'],
    noise_clip=training_config['noise_clip'],
    reward_normalization=training_config['rew_norm'],
    ignore_done=training_config['ignore_done'],
    estimation_step=training_config['n_step'])

if training_config['prioritized_replay']:
    buf = PrioritizedReplayBuffer(
            training_config['buffer_size'],
            alpha=training_config['alpha'], beta=training_config['beta'])
else:
    buf = ReplayBuffer(training_config['buffer_size'])

train_collector = Collector(policy, train_envs, buf)
train_collector.collect(n_step=training_config['pre_collect'])

train_fn = lambda e: [policy.set_exp_noise(GaussianNoise(sigma=(max(0.02, training_config['exploration_noise']*(1-(e-1)/training_config['epoch']/training_config['exploration_ratio']))))), \
                      torch.save(policy.state_dict(), os.path.join(save_path, 'policy_%d.pth' %(e)))]

result = offpolicy_trainer(
        policy, train_collector, training_config['epoch'],
        training_config['step_per_epoch'], training_config['collect_per_step'],
        training_config['batch_size'], update_per_step=training_config['update_per_step'],
        train_fn=train_fn, writer=writer)

train_envs.close()
