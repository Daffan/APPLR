import os
import json
import pickle
from os.path import join, dirname, abspath, exists
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import jackal_navi_envs

from policy import TD3Policy, SACPolicy
from tianshou.utils.net.common import Net
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.continuous import Actor, Critic, ActorProb
from tianshou.data import Batch

from torch import nn
import torch
import gym
import numpy as np
import random
import time
import os
from utils import Benchmarking_train, Benchmarking_test, path_to_world

random.seed(43)
SET = os.getenv("TEST_SET")
benchmarking_test = Benchmarking_test
if SET != 'test':
    benchmarking_test = Benchmarking_train
    assert len(benchmarking_test)==250

BASE_PATH = join(os.getenv('HOME'), 'buffer_test')

def init_actor(id):
    assert os.path.exists(BASE_PATH)
    import rospy
    rospy.init_node('gym', anonymous=True, log_level=rospy.FATAL)
    rospy.set_param('/use_sim_time', True)
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
        assert os.path.exists(path_to_world(benchmarking_train[id]))
    wrapper_config = config['wrapper_config']
    training_config = config['training_config']
    wrapper_dict = jackal_navi_envs.jackal_env_wrapper.wrapper_dict
    env = wrapper_dict[wrapper_config['wrapper']](gym.make(config["env"], **env_config), **wrapper_config['wrapper_args'])
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    actor_optim = torch.optim.Adam(actor.parameters(), lr=training_config['actor_lr'])
    net = Net(training_config['num_layers'], state_shape,
              action_shape, concat=True, device=device, hidden_layer_size=training_config['hidden_size'])
    critic1 = Critic(net, device, hidden_layer_size=training_config['hidden_size']).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=training_config['critic_lr'])
    critic2 = Critic(net, device, hidden_layer_size=training_config['hidden_size']).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=training_config['critic_lr'])

    if config['section'] == 'SAC':
        policy = SACPolicy(
            actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
            action_range=[env.action_space.low, env.action_space.high],
            tau=training_config['tau'], gamma=training_config['gamma'],
            reward_normalization=training_config['rew_norm'],
            ignore_done=training_config['ignore_done'],
            alpha=training_config['sac_alpha'],
            exploration_noise=None,
            estimation_step=training_config['n_step'])
    else:
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
