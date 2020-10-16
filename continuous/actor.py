import os
import json
import pickle
from os.path import join, dirname, abspath, exists
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import jackal_navi_envs
from torch import nn
import torch
import gym
import numpy as np
import random
import time
random.seed(43)
benchmarking_train = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 101, 102, 103, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 119, 120, 121, 122, 124, 125, 126, 127, 128, 130, 131, 132, 134, 135, 136, 137, 139, 140, 141, 142, 143, 145, 146, 147, 148, 149, 151, 152, 153, 154, 155, 156, 157, 158, 160, 161, 162, 164, 165, 166, 167, 169, 170, 171, 172, 173, 174, 176, 177, 178, 179, 180, 181, 182, 183, 185, 186, 187, 188, 190, 191, 192, 194, 195, 196, 197, 198, 199, 200, 202, 203, 204, 205, 206, 207, 209, 210, 211, 212, 213, 215, 216, 217, 219, 220, 221, 222, 223, 224, 225, 227, 228, 230, 231, 232, 233, 234, 235, 236, 238, 239, 241, 242, 243, 244, 245, 247, 248, 249, 250, 251, 252, 253, 254, 255, 257, 259, 260, 261, 262, 263, 264, 266, 267, 268, 269, 271, 272, 273, 274, 275, 276, 278, 279, 280, 281, 282, 283, 285, 286, 287, 288, 289, 291, 292, 293, 295, 296, 297, 298, 299]
random.shuffle(benchmarking_train)

#BASE_PATH = '/u/zifan/buffer'
BASE_PATH = '/home/gauraang/buffer'

def init_actor(id):
    assert os.path.exists(BASE_PATH)
    actor_path = join(BASE_PATH, 'actor_%s' %(str(id)))
    if not exists(actor_path):
        os.mkdir(actor_path) # path to store all the trajectories

    with open(join(BASE_PATH, 'config.json'), 'rb') as f:
        config = json.load(f)
    return config

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

def main(id):

    config = init_actor(id)
    env_config = config['env_config']
    if env_config['world_name'] != "sequential_applr_testbed.world":
        env_config['world_name'] = 'Benchmarking/train/world_%d.world' %(benchmarking_train[id])
        assert os.path.exists('/jackal_ws/src/jackal_helper/worlds/Benchmarking/train/world_%d.world' %(benchmarking_train[id]))
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
    while True:
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
            obs_new, rew, done, info = env.step(action)
            count += 1
            print('current step: %d, X position: %f, Y position: %f, rew: %f, succeed: %d' %(count, info['X'], info['Y'], rew, info['succeed']), end = '\r')
            traj.append([obs, action, rew, done, info])
            obs = obs_new
        # print('count: %d, rew: %f' %(count, rew))
        write_buffer(traj, ep, id)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'start an actor')
    parser.add_argument('--id', dest='actor_id', type = int, default = 1)
    id = parser.parse_args().actor_id
    main(id)
