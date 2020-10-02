from os.path import exists, join
import json
import os
import torch
import time
import pickle

BASE_PATH = '/tmp/buffer'
class Collector(object):

    def __init__(self, policy, env, replaybuffer):
        '''
        it's a fake tianshou Collector object with the same api
        '''
        super().__init__()
        self.policy = policy
        self.env = env
        self.num_actor = env['training_config']['num_actor']
        self.ids = list(range(self.num_actor))
        self.ep_count = [0]*self.num_actor
        self.buffer = replaybuffer

        if not exists(BASE_PATH):
            os.mkdir(BASE_PATH)
        # save the current path
        self.update_policy()
        # save the env config the actor should read from
        with open(join(BASE_PATH, 'config.json'), 'w') as fp:
            json.dump(self.env, fp)

    def update_policy(self):
        torch.save(self.policy.state_dict(), join(BASE_PATH, 'policy.pth'))
        with open(join(BASE_PATH, 'eps.txt'), 'w') as f:
            f.write(str(self.policy.eps))

    def buffer_expand(self, traj):
        for i in range(len(traj)):
            obs_next = traj[i+1][0] if i < len(traj)-1 else traj[i][0]
            self.buffer.add(traj[i][0], traj[i][1], \
                            traj[i][2], traj[i][3], \
                            obs_next, traj[i][4])

    def collect(self, n_step):
        # collect happens after policy is updated
        self.update_policy()
        steps = 0
        ep_rew = []
        ep_len = []
        while steps < n_step:
            time.sleep(1)
            for id in self.ids:
                c = self.ep_count[id]
                base = join(BASE_PATH, 'actor_%d' %(id))
                try:
                    trajs = sorted(os.listdir(base))
                except:
                    trajs = []
                    # print('waiting actor %d to be initialized' %(id))
                ct = len(trajs)
                self.ep_count[id] = ct
                time.sleep(0.1) # to prevent ran out of input error
                for i in range(c, ct):
                    t = 'traj_%d.pickle' %(i+1)
                    print('read actor_%d %s' %(id, t))
                    with open(join(base, t), 'rb') as f:
                        traj = pickle.load(f)
                        ep_rew.append(sum([t[2] for t in traj]))
                        ep_len.append(len(traj))
                        self.buffer_expand(traj)
                        steps += len(traj)
        return {'n/st': steps, 'ep_rew': sum(ep_rew)/len(ep_rew), 'ep_len': sum(ep_len)/len(ep_len)}


