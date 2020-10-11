import os
import json
import pickle
from os.path import join, dirname, abspath, exists
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import jackal_navi_envs
from .dqn import DuelingDQN
from torch import nn
import torch
import gym
import numpy as np
import random
import time
random.seed(43)
benchmarking_test = [0, 8, 17, 19, 27, 32, 41, 47, 48, 57, 64, 69, 76, 78, 88, 93, 100, 104, 112, 118, 123, 129, 133, 138, 144, 150, 159, 163, 168, 175, 184, 189, 193, 201, 208, 214, 218, 226, 229, 237, 240, 246, 256, 258, 265, 270, 277, 284, 290, 294]
random.shuffle(benchmarking_test)

BASE_PATH = '/u/zifan/APPLR/buffer_test'

def init_actor(id):
    assert os.path.exists(BASE_PATH)
    actor_path = join(BASE_PATH, 'actor_%s' %(str(id)))
    if not exists(actor_path):
        os.mkdir(actor_path) # path to store all the trajectories

    with open(join(BASE_PATH, 'config.json'), 'rb') as f:
        config = json.load(f)
    return config
'''
class DuelingDQN(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_layer = [64, 64], cnn = True):
        super().__init__()
        if cnn:
            self.feature = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2),
                nn.ReLU(), nn.MaxPool1d(kernel_size = 5),
                nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
                nn.ReLU(), nn.MaxPool1d(kernel_size = 5),
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
                nn.ReLU(), nn.AvgPool1d(6)
                )
            feature_shape = 70
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
        laser = obs.view(batch, 1, -1)[:,:,:721]
        params = obs.view(batch, -1)[:, 721:]

        embedding = self.feature(laser).view(batch, -1)
        feature = torch.cat((embedding, params), dim = 1)

        advantage = self.advantage(feature)
        value = self.value(feature)
        logits = value + advantage - advantage.mean(1, keepdim=True)
        return logits, state
'''
def load_model(model):
    model_path = join(BASE_PATH, 'policy.pth')
    state_dict = {}
    state_dict_raw = None
    while state_dict_raw is None:
        try:
            state_dict_raw = torch.load(model_path)
        except:
            time.sleep(0.1)
            pass
    for key in state_dict_raw.keys():
        if key.split('.')[0] == 'model':
            state_dict[key[6:]] = state_dict_raw[key]

    model.load_state_dict(state_dict)
    model = model.float()

    with open(join(BASE_PATH, 'eps.txt'), 'r') as f:
        eps = float(f.readlines()[0])

    return model, eps

def write_buffer(traj, ep, id):
    with open(join(BASE_PATH, 'actor_%s' %(str(id)), 'traj_%d.pickle' %(ep)), 'wb') as f:
        pickle.dump(traj, f)

def main(id, avg, default):

    config = init_actor(id)
    env_config = config['env_config']
    if env_config['world_name'] != "sequential_applr_testbed.world":
        env_config['world_name'] = 'Benchmarking/test/world_%d.world' %(benchmarking_test[id])
        assert os.path.exists('/jackal_ws/src/jackal_helper/worlds/Benchmarking/test/world_%d.world' %(benchmarking_test[id]))
    wrapper_config = config['wrapper_config']
    training_config = config['training_config']
    wrapper_dict = jackal_navi_envs.jackal_env_wrapper.wrapper_dict
    if config['env'] == 'jackal':
        env = wrapper_dict[wrapper_config['wrapper']](gym.make('jackal_discrete-v0', **env_config), **wrapper_config['wrapper_args'])
    else:
        env = gym.make('CartPole-v1')
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    model = DuelingDQN(state_shape, action_shape, hidden_layer = training_config['hidden_layer'], cnn = training_config['cnn'])

    ep = 0
    for _ in range(avg):
        obs = env.reset()
        ep += 1
        traj = []
        done = False
        count = 0
        model, eps = load_model(model)
        while not done:
            p = random.random()
            obs = torch.tensor([obs]).float()
            actions = model(obs)[0].detach().numpy()[0]
            if p>eps:
                action = np.argmax(actions.reshape(-1))
            else:
                action = random.choice(list(range(len(actions))))
            if default:
                action = list(range(len(actions)))[-1] # Keep the default parameters unchanged
            obs_new, rew, done, info = env.step(action)
            count += 1
            # print('current step: %d, X position: %f, Y position: %f, rew: %f, succeed: %d' %(count, info['X'], info['Y'], rew, info['succeed']), end = '\r')
            traj.append([obs, action, rew, done, info])
            obs = obs_new
        # print('count: %d, rew: %f' %(count, rew))
        write_buffer(traj, ep, id)
    env.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'start an actor')
    parser.add_argument('--id', dest='actor_id', type = int, default = 1)
    parser.add_argument('--avg', dest='avg', type = int, default = 3)
    parser.add_argument('--default', dest='default', type = bool, default = False)

    id = parser.parse_args().actor_id
    default = parser.parse_args().default
    avg = parser.parse_args().avg
    main(id, avg, default)
