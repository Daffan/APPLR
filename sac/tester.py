import os
import json
import pickle
from os.path import join, dirname, abspath, exists
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import jackal_navi_envs

from policy import TD3Policy
from tianshou.utils.net.common import Net
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.data import Batch

from torch import nn
import torch
import gym
import numpy as np
import random
import time
import os

random.seed(43)
SET = os.getenv("TEST_SET")
benchmarking_test = [0, 8, 17, 19, 27, 32, 41, 47, 48, 57, 64, 69, 76, 78, 88, 93, 100, 104, 112, 118, 123, 129, 133, 138, 144, 150, 159, 163, 168, 175, 184, 189, 193, 201, 208, 214, 218, 226, 229, 237, 240, 246, 256, 258, 265, 270, 277, 284, 290, 294]
if SET != 'test':
    benchmarking_test = [i for i in list(range(300)) if i not in benchmarking_test] # all the training world
    assert len(benchmarking_test)==250

BASE_PATH = join(os.getenv('HOME'), 'buffer_test')

def init_actor(id):
    assert os.path.exists(BASE_PATH)
    actor_path = join(BASE_PATH, 'actor_%s' %(str(id)))
    if not exists(actor_path):
        os.mkdir(actor_path) # path to store all the trajectories

    with open(join(BASE_PATH, 'config.json'), 'rb') as f:
        config = json.load(f)
    return config

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

    model.load_state_dict(state_dict_raw)
    model = model.float()

    return model

def write_buffer(traj, ep, id):
    with open(join(BASE_PATH, 'actor_%s' %(str(id)), 'traj_%d.pickle' %(ep)), 'wb') as f:
        pickle.dump(traj, f)

def main(id, avg, default):

    config = init_actor(id)
    env_config = config['env_config']
    if env_config['world_name'] != "sequential_applr_testbed.world":
        env_config['world_name'] = 'Benchmarking/%s/world_%d.world' %(SET, benchmarking_test[id])
        assert os.path.exists('/jackal_ws/src/jackal_helper/worlds/Benchmarking/%s/world_%d.world' %(SET, benchmarking_test[id]))
    wrapper_config = config['wrapper_config']
    training_config = config['training_config']
    wrapper_dict = jackal_navi_envs.jackal_env_wrapper.wrapper_dict
    if config['env'] == 'jackal':
        env = wrapper_dict[wrapper_config['wrapper']](gym.make('jackal_continuous-v0', **env_config), **wrapper_config['wrapper_args'])
    else:
        env = gym.make('CartPole-v1')
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
        exploration_noise=None,
        policy_noise=training_config['policy_noise'],
        update_actor_freq=training_config['update_actor_freq'],
        noise_clip=training_config['noise_clip'],
        reward_normalization=training_config['rew_norm'],
        ignore_done=training_config['ignore_done'],
        estimation_step=training_config['n_step'])
    print(env.action_space.low, env.action_space.high)
    ep = 0
    for _ in range(avg):
        obs = env.reset()
        obs_batch = Batch(obs=[obs], info={})
        ep += 1
        traj = []
        done = False
        count = 0
        policy = load_model(policy)
        while not done:
            if not default:
                actions = policy(obs_batch).act.cpu().detach().numpy().reshape(-1)
            else:
                actions = np.array([0.5, 1.57, 6, 20, 0.75, 1, 0.3])
            obs_new, rew, done, info = env.step(actions)
            count += 1
            info["world"] = benchmarking_test[id]
            traj.append([obs, actions, rew, done, info])
            obs_batch = Batch(obs=[obs_new], info={})
            obs = obs_new
        # print('count: %d, rew: %f' %(count, rew))
        write_buffer(traj, ep, id)
    env.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'start an actor')
    parser.add_argument('--id', dest='actor_id', type = int, default = 1)
    parser.add_argument('--avg', dest='avg', type = int, default = 20)
    parser.add_argument('--default', dest='default', type = bool, default = False)

    id = parser.parse_args().actor_id
    default = parser.parse_args().default
    avg = parser.parse_args().avg
    main(id, avg, default)
