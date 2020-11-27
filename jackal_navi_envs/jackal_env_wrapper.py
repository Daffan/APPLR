import gym
import numpy as np
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

    def __init__(self, env, goal_distance_reward = 2, stuck_punishment = 0.5,\
                punishment_reward = -100, collision = 0, reward_scale = 1):
        '''A wrapper that will shape the reward by the length of the globle path. The robot flip over
        will terminate and return a large negative reward.
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
        self.collision = collision

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

        if self.collision:
            laser = obs[:721]
            d = np.mean(sorted(laser)[:10])+0.5 #(oringinal range was [-0.5, 0.5], move to [0, 1])
            if d < 0.1:
                rew -= self.collision/(d+0.01)

        if position.z > 0.1 or not info['succeed']:
            done = True
            rew += self.punishment_reward
            info['succeed'] = False

        if position.y > 10: # 10 for benchmarking
            done = True
            info['succeed'] = True

        info['X'] = position.x
        info['Y'] = position.y
        rew = rew/self.reward_scale

        return obs, rew, done, info

class BenchMarkingWrapperReward(gym.Wrapper):

    def __init__(self, env, goal_distance_reward = 2, success_reward = 10, \
                smoothness = 0.1, prevent_extreme = 0.1, stuck_punishment = 0.1, \
                punishment_reward = -100, collision = 0.1, reward_scale = 1):
        '''A wrapper that will shape the reward by the length of the globle path. The robot flip over
        will terminate and return a large negative reward. Some more reward functions are added
        args:
            env -- GazeboJackalNavigationEnv
            arg['goal_distance_reward'] -- [float] coefficient of change of distance when added into reward
            arg['stuck_punishment'] -- [float] coefficient of stuck_punishment reward
            arg['punishment_reward'] -- [int] the large negative reward when flip over or stuck
        '''
        super(BenchMarkingWrapperReward, self).__init__(env)
        self.goal_distance_reward = goal_distance_reward
        self.stuck_punishment = stuck_punishment
        self.punishment_reward = punishment_reward
        self.reward_scale = reward_scale
        self.success_reward = success_reward
        self.smoothness = smoothness
        self.prevent_extreme = prevent_extreme
        self.collision = collision

    def reset(self):
        obs = self.env.reset()
        self.Y = self.env.gazebo_sim.get_model_state().pose.position.y
        self.params = self.env.param_init
        return obs

    def step(self, action):
        # take one step
        obs, rew, done, info = self.env.step(action)
        # reward is the decrease of the distance
        position = self.env.gazebo_sim.get_model_state().pose.position
        rew = 0 # get rid of the time step
        rew += (0.99*position.y - self.Y) * self.goal_distance_reward
        self.Y = position.y
        rew += self.env.navi_stack.punish_rewrad()*self.stuck_punishment

        # smoothness
        if self.smoothness:
            scale = self.env.action_space.high - self.env.action_space.low
            rew += -np.sum(np.abs(np.array(info['params']) - self.params)*self.smoothness/scale)
            self.params = info['params']

        # prevent_extreme
        if self.prevent_extreme:
            high = self.env.action_space.high - np.array(self.env.param_init)
            low = self.env.action_space.low - np.array(self.env.param_init)
            r = 0
            for i in range(len(action)):
                t = action[i] - self.env.param_init[i]
                if t > 0:
                    r += -self.prevent_extreme * t / high[i]
                else:
                    r += -self.prevent_extreme * t / low[i]
            rew += r

        # collision
        if self.collision:
            laser = obs[:721]
            d = np.mean(sorted(laser)[:10])+0.5 #(oringinal range was [-0.5, 0.5], move to [0, 1])
            if d < 0.1:
                rew -= self.collision/(d+0.01)

        if position.z > 0.1: # or not info['succeed']:
            done = True
            rew += self.punishment_reward
            info['succeed'] = False

        if position.y > 10:
            done = True
            info['succeed'] = True
            rew += self.success_reward

        info['X'] = position.x
        info['Y'] = position.y
        rew = rew/self.reward_scale

        return obs, rew, done, info

wrapper_dict = {
    'sequential_world_wrapper': SequentialWorldWrapper,
    'bench_marking_wrapper': BenchMarkingWrapper,
    'bench_marking_restart_wrapper': BenchMarkingResartWrapper,
    'bench_marking_wrapper_reward': BenchMarkingWrapperReward,
    'default': lambda env: env
}
