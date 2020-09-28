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

BASE_PATH = '/tmp/buffer'

def init_actor(id):
    assert os.path.exists(BASE_PATH)
    actor_path = join(BASE_PATH, 'actor_%s' %(str(id)))
    if not exists(actor_path):
        os.mkdir(actor_path) # path to store all the trajectories

    with open(join(BASE_PATH, 'config.json'), 'rb') as f:
        config = json.load(f)
    return config

class DuelingDQN(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_layer = [128, 128]):
        super().__init__()
        layers = [np.prod(state_shape)] + hidden_layer
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
        advantage = self.advantage(obs.view(batch, -1))
        value = self.value(obs.view(batch, -1))
        logits = value + advantage - advantage.mean(1, keepdim=True)
        return logits, state

def load_model(model):
    model_path = join(BASE_PATH, 'policy.pth')
    state_dict = {}
    state_dict_raw = torch.load(model_path)
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
    wrapper_config = config['wrapper_config']
    training_config = config['training_config']
    wrapper_dict = jackal_navi_envs.jackal_env_wrapper.wrapper_dict
    env = wrapper_dict[wrapper_config['wrapper']](gym.make('jackal_discrete-v0', **env_config), **wrapper_config['wrapper_args'])

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    model = DuelingDQN(state_shape, action_shape, hidden_layer = training_config['hidden_layer'])

    ep = 0
    while True:
        obs = env.reset()
        ep += 1
        traj = []
        model, eps = load_model(model)
        done = False
        while not done:
            p = random.random()
            actions = np.array(model(torch.tensor([obs]).float())[0].detach().cpu())
            action = np.argmax(actions.reshape(-1)) if p>eps else random.choice(list(range(len(actions))))
            obs, rew, done, info = env.step(action)
            traj.append([obs, action, rew, done, info])
        write_buffer(traj, ep, id)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'start an actor')
    parser.add_argument('--id', dest='actor_id', type = int, default = 1)
    id = parser.parse_args().actor_id
    main(id)
