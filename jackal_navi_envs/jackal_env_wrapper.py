import gym
import numpy as np
# import matplotlib.pyplot as plt
gym.logger.set_level(40)

class SequentialWorldWrapper(gym.Wrapper):

    def __init__(self, env, goal_distance_reward = 2, stuck_punishment = 0.5, punishment_reward = -1000):
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
        rew = rew/100

        return obs, rew, done, info


wrapper_dict = {
    'sequential_world_wrapper': SequentialWorldWrapper,
    'default': lambda env: env
}
