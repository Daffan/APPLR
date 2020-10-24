import gym
import numpy as np
# import matplotlib.pyplot as plt
gym.logger.set_level(40)

class SequentialWorldWrapper(gym.Wrapper):

    def __init__(self, env, goal_distance_reward = 2, stuck_punishment = 0.5, punishment_reward = -1000, reward_scale = 1):
        '''A wrapper that will shape the reward by the length of the globle path. The robot flip over or stuck at the same
        place for 100 step will terminate and return a large negative reward.
        args:
            env -- GazeboJackalNavigationEnv
            arg['goal_distance_reward'] -- [float] coefficient of change of distance when added into reward
            arg['stuck_punishment'] -- [float] coefficient of stuck_punishment reward
            arg['punishment_reward'] -- [int] the large negative reward when flip over or stuck
        '''
        super(SequentialWorldWrapper, self).__init__(env)
        self.goal_distance_reward = goal_distance_reward
        self.stuck_punishment = stuck_punishment
        self.punishment_reward = punishment_reward
        self.reward_scale = reward_scale
        self.global_path = self.env.navi_stack.robot_config.global_path
        self.gp_len = sum([self.distance(self.global_path[i+1], self.global_path[i]) for i in range(len(self.global_path)-1)])

    def distance(self, p1, p2):
        return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

    def reset(self):
        obs = self.env.reset()
        self.env._set_param('/X', self.env.gazebo_sim.get_model_state().pose.position.x)
        self.rp = [] # sequence of robot X position
        return obs

    # def visual_path(self):
    #     plt.scatter(self.global_path[:,0], self.global_path[:,1])
    #     plt.scatter(self.env.navi_stack.robot_config.X, self.env.navi_stack.robot_config.Y)
    #     plt.show()

    def step(self, action):
        # take one step
        obs, rew, done, info = self.env.step(action)
        # reward is the decrease of the distance
        position = self.env.gazebo_sim.get_model_state().pose.position
        rew += (position.x - self.env._get_param('/X')) * self.goal_distance_reward
        self.env._set_param('/X', position.x)
        rew += self.env.navi_stack.punish_rewrad()*self.stuck_punishment
        rp = np.array(position.x)
        try:
            self.rp.append(rp)
            # When robot don't move forward for 100 steps, terminate and return large negative rew
            if len(self.rp) > 100:
                if self.rp[-1] < self.rp[-100]:
                    done = True
                    rew = self.punishment_reward
        except:
            pass
        if position.z > 0.1: # or
            done = True
            rew = self.punishment_reward
        if position.x > 42: # or
            done = True
        info['X'] = position.x
        info['Y'] = position.y
        rew = rew/self.reward_scale

        return obs, rew, done, info

class BenchMarkingWrapper(gym.Wrapper):

    def __init__(self, env, goal_distance_reward = 2, stuck_punishment = 0.5, punishment_reward = -100, reward_scale = 1):
        '''A wrapper that will shape the reward by the length of the globle path. The robot flip over or stuck at the same
        place for 100 step will terminate and return a large negative reward.
        args:
            env -- GazeboJackalNavigationEnv
            arg['goal_distance_reward'] -- [float] coefficient of change of distance when added into reward
            arg['stuck_punishment'] -- [float] coefficient of stuck_punishment reward
            arg['punishment_reward'] -- [int] the large negative reward when flip over or stuck
        '''
        super(BenchMarkingWrapper, self).__init__(env)
        self.goal_distance_reward = goal_distance_reward
        self.stuck_punishment = stuck_punishment
        self.punishment_reward = punishment_reward
        self.reward_scale = reward_scale

    def reset(self):
        obs = self.env.reset()
        self.Y = self.env.gazebo_sim.get_model_state().pose.position.y
        return obs

    def step(self, action):
        # take one step
        obs, rew, done, info = self.env.step(action)
        # reward is the decrease of the distance
        position = self.env.gazebo_sim.get_model_state().pose.position
        rew += (position.y - self.Y) * self.goal_distance_reward
        self.Y = position.y
        #rew += self.env.navi_stack.punish_rewrad()*self.stuck_punishment

        if position.z > 0.1 or not info['succeed']: # or
            done = True
            rew = self.punishment_reward
            info['succeed'] = False

        if position.y > 10: # 10 for benchmarking
            done = True
            info['succeed'] = True

        info['X'] = position.x
        info['Y'] = position.y
        rew = rew/self.reward_scale

        return obs, rew, done, info

from .jackal_env_discrete import JackalEnvDiscrete
import random
benchmarking_train = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 101, 102, 103, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 119, 120, 121, 122, 124, 125, 126, 127, 128, 130, 131, 132, 134, 135, 136, 137, 139, 140, 141, 142, 143, 145, 146, 147, 148, 149, 151, 152, 153, 154, 155, 156, 157, 158, 160, 161, 162, 164, 165, 166, 167, 169, 170, 171, 172, 173, 174, 176, 177, 178, 179, 180, 181, 182, 183, 185, 186, 187, 188, 190, 191, 192, 194, 195, 196, 197, 198, 199, 200, 202, 203, 204, 205, 206, 207, 209, 210, 211, 212, 213, 215, 216, 217, 219, 220, 221, 222, 223, 224, 225, 227, 228, 230, 231, 232, 233, 234, 235, 236, 238, 239, 241, 242, 243, 244, 245, 247, 248, 249, 250, 251, 252, 253, 254, 255, 257, 259, 260, 261, 262, 263, 264, 266, 267, 268, 269, 271, 272, 273, 274, 275, 276, 278, 279, 280, 281, 282, 283, 285, 286, 287, 288, 289, 291, 292, 293, 295, 296, 297, 298, 299]

class BenchMarkingResartWrapper(gym.Wrapper):

    def __init__(self, env, env_config, goal_distance_reward = 2, stuck_punishment = 0.5, punishment_reward = -100, reward_scale = 1, switch_interval = 2):
        '''A wrapper that will shape the reward by the length of the globle path. The robot flip over or stuck at the same
        place for 100 step will terminate and return a large negative reward.
        args:
            env -- GazeboJackalNavigationEnv
            arg['goal_distance_reward'] -- [float] coefficient of change of distance when added into reward
            arg['stuck_punishment'] -- [float] coefficient of stuck_punishment reward
            arg['punishment_reward'] -- [int] the large negative reward when flip over or stuck
        '''
        super(BenchMarkingResartWrapper, self).__init__(env)
        self.env_config = env_config
        self.goal_distance_reward = goal_distance_reward
        self.stuck_punishment = stuck_punishment
        self.punishment_reward = punishment_reward
        self.reward_scale = reward_scale
        self.ep_count = 1
        self.switch_interval = switch_interval

    def reset(self):
        if self.ep_count%(self.switch_interval+1):
            self.env.close()
            self.env_config['world_name'] = 'Benchmarking/train/world_%d.world' %(random.choice(benchmarking_train))
            self.env = JackalEnvDiscrete(**self.env_config)
            self.ep_count = 0
        self.ep_count += 1
        obs = self.env.reset()
        self.env._set_param('/Y', self.env.gazebo_sim.get_model_state().pose.position.y)
        self.rp = [] # sequence of robot Y position
        return obs

    def step(self, action):
        # take one step
        obs, rew, done, info = self.env.step(action)
        # reward is the decrease of the distance
        position = self.env.gazebo_sim.get_model_state().pose.position
        rew += (position.y - self.env._get_param('/Y')) * self.goal_distance_reward
        self.env._set_param('/Y', position.y)
        rew += self.env.navi_stack.punish_rewrad()*self.stuck_punishment
        rp = np.array(position.y)

        if position.z > 0.1: # or
            done = True
            rew = self.punishment_reward

        if position.y > 10: # 10 for benchmarking
            done = True

        info['X'] = position.x
        info['Y'] = position.y
        rew = rew/self.reward_scale

        return obs, rew, done, info


wrapper_dict = {
    'sequential_world_wrapper': SequentialWorldWrapper,
    'bench_marking_wrapper': BenchMarkingWrapper,
    'bench_marking_restart_wrapper': BenchMarkingResartWrapper,
    'default': lambda env: env
}
