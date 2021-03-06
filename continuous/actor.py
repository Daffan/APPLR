import os
import json
import pickle
from os.path import join, dirname, abspath, exists
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import jackal_navi_envs
from jackal_navi_envs import range_dict
from jackal_navi_envs.APPLX import APPLD_policy, APPLE_policy, APPLI_policy
from torch import nn
import torch
import gym
import numpy as np
import random
import time
from policy import TD3Policy, SACPolicy
from tianshou.utils.net.common import Net as MLP
from net import Net as CNN
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.continuous import Actor, Critic, ActorProb
from tianshou.data import Batch
from utils import train_worlds, Benchmarking_train, Benchmarking_test, path_to_world

random.seed(43)
train_worlds = train_worlds*100
# random.shuffle(train_worlds)

BASE_PATH = join(os.getenv('HOME'), 'buffer')
APPLD_policy, APPLE_policy, APPLI_policy = APPLD_policy(), APPLE_policy(), APPLI_policy() 
def init_actor(id):
    import rospy
    rospy.logwarn(">>>>>>>>>>>>>>>>>> actor id: %s <<<<<<<<<<<<<<<<<<" %(str(id)))
    # rospy.init_node('gym', anonymous=True, log_level=rospy.FATAL)
    # rospy.set_param('/use_sim_time', True)
    assert os.path.exists(BASE_PATH)
    actor_path = join(BASE_PATH, 'actor_%s' %(str(id)))
    if not exists(actor_path):
        os.mkdir(actor_path) # path to store all the trajectories

    f = None
    while f is None:
        try:
            f = open(join(BASE_PATH, 'config.json'), 'rb')
        except:
            rospy.logwarn("wait for critor to be initialized")
            time.sleep(2)

    config = json.load(f)

    num_env = config['training_config']['num_actor']
    def count_actor():
        files = os.listdir(BASE_PATH)
        num_actor = sum([f.startswith("actor_") for f in files])
        return num_actor
    # wait until most of actors successfully initialized, tolerance 50 envs
    #while num_env-count_actor() > 50:
    #    time.sleep(10)

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
    # exploration noise std
    with open(join(BASE_PATH, 'eps.txt'), 'r') as f:
        eps = None
        while eps is not None:
            try:
                eps = float(f.readlines()[0])
            except IndexError:
                pass

    return model, eps

def write_buffer(traj, ep, id):
    with open(join(BASE_PATH, 'actor_%s' %(str(id)), 'traj_%d.pickle' %(ep)), 'wb') as f:
        pickle.dump(traj, f)

def get_random_action():
    actions = []
    for k in range_dict.keys():
        maxlimit = range_dict[k][1]
        minlimit = range_dict[k][0]
        p = random.random()
        actions.append(minlimit+p*(maxlimit-minlimit))
    return actions

def main(id):

    config = init_actor(id)
    env_config = config['env_config']
    if env_config['world_name'] != "sequential_applr_testbed.world":
        assert os.path.exists(join("/jackal_ws/src/jackal_helper/worlds", path_to_world(train_worlds[id])))
        env_config['world_name'] = path_to_world(train_worlds[id])
    wrapper_config = config['wrapper_config']
    training_config = config['training_config']
    wrapper_dict = jackal_navi_envs.jackal_env_wrapper.wrapper_dict
    env = wrapper_dict[wrapper_config['wrapper']](gym.make(config["env"], **env_config), **wrapper_config['wrapper_args'])
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Net = CNN if training_config["cnn"] == True else MLP
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
    print(">>>>>>>>>>>>>> Running on world_%d <<<<<<<<<<<<<<<<" %(train_worlds[id]))
    ep = 0
    while True:
        obs = env.reset()
        gp = env.gp
        scan = env.scan
        obs_batch = Batch(obs=[obs], info={})
        ep += 1
        traj = []
        ctcs = []
        done = False
        count = 0
        policy, eps = load_model(policy)
        try:
            policy.set_exp_noise(GaussianNoise(sigma=eps))
        except:
            pass
        while not done:
            time.sleep(0.01)
            p = random.random()
            obs = torch.tensor([obs]).float()
            # actions = np.array([0.5, 1.57, 6, 20, 0.8, 1, 0.3])
            #else:
            obs_x = [scan, gp]
            """
            if p < eps/3.:
                actions = APPLD_policy.forward(obs_x)
                print("APPLD", actions)
            elif p < 2*eps/3.:
                actions = APPLI_policy.forward(obs_x)
                print("APPLI", actions)
            elif p < eps:
                actions = APPLE_policy.forward(obs_x)
                print("APPLE", actions)
            else:
                actions = policy(obs_batch).act.cpu().detach().numpy().reshape(-1)
            if p < eps:
                if train_worlds[id] in [74, 271, 213, 283, 265, 273, 137, 209, 194]:
                    actions = APPLI_policy.forward(obs_x)
                elif train_worlds[id] in [293, 105, 153, 292, 254, 221, 245]:
                    actions = APPLD_policy.forward(obs_x) 
            """
            if p<eps:
                actions = get_random_action()
                actions = np.array(actions)
            else:
                actions = policy(obs_batch).act.cpu().detach().numpy().reshape(-1)
            ctc = critic1(obs, torch.tensor([actions]).float()).cpu().detach().numpy().reshape(-1)[0]
            ctcs.append(ctc)
            obs_new, rew, done, info = env.step(actions)
            count += 1
            gp = info.pop("gp")
            scan = info.pop("scan")
            info["world"] = train_worlds[id]
            traj.append([obs, actions, rew, done, info])
            obs_batch = Batch(obs=[obs_new], info={})
            obs = obs_new
            #print(rew, done, info)

        """
        # filter the traj that has lower discounted reward as it predicted by the critic
        if p < eps:
            def compute_discouted_rew(rew, gamma):
                return sum([r*(gamma**i) for i, r in enumerate(rew)])
            rews = [t[2] for t in traj]
            discounted_rew = [compute_discouted_rew(rews[i:], training_config["gamma"]) for i in range(len(rews))]
            assert len(ctcs) == len(discounted_rew)
            use = [r > c for r, c in zip(discounted_rew, ctcs)]
            traj_new = [t for u, t in zip(use, traj) if u]
        else:
            traj_new = traj
        """
        traj_new = traj
        if len(traj_new) > 0:
            write_buffer(traj_new, ep, id)
        # write_buffer(traj, ep, id)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'start an actor')
    parser.add_argument('--id', dest='actor_id', type = int, default = 1)
    id = parser.parse_args().actor_id
    main(id)
