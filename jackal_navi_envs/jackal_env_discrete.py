import gym
import rospy
import rospkg
import roslaunch
import time
import numpy as np
import os
from os.path import dirname, join
import subprocess

from gym import utils, spaces
from std_srvs.srv import Empty
import actionlib
from gym.utils import seeding

from .gazebo_simulation import GazeboSimulation
from .navigation_stack import  NavigationStack

gym.logger.set_level(40)

range_dict = {
    'max_vel_x': [0.1, 2],
    'max_vel_theta': [0.314, 3.14],
    'vx_samples': [4, 12],
    'vtheta_samples': [8, 40],
    'path_distance_bias': [0.1, 1.5],
    'goal_distance_bias': [0.1, 2]
}

class JackalEnvDiscrete(gym.Env):

    def __init__(self, world_name = 'sequential_applr_testbed.world', VLP16 = 'false', gui = 'false', camera = 'false',
                init_position = [-8, 0, 0], goal_position = [54, 0, 0], max_step = 600, time_step = 1, laser_clip = 4,
                param_delta = [0.2, 0.3, 1, 2, 0.2, 0.2], param_init = [0.5, 1.57, 6, 20, 0.75, 1],
                param_list = ['max_vel_x', 'max_vel_theta', 'vx_samples', 'vtheta_samples', 'path_distance_bias', 'goal_distance_bias'],
                init_world = True):
        gym.Env.__init__(self)

        self.world_name = world_name
        self.VLP16 = True if VLP16=='true' else False
        self.gui = True if gui=='true' else False
        self.max_step = max_step
        self.time_step = time_step
        self.laser_clip = laser_clip
        if not world_name.startswith('Benchmarking'):
            self.goal_position = goal_position
            self.init_position = init_position
        else:
            base = dirname(abspath(__file__))
            path = np.load(join(base, 'path_files', 'path_%d.npy' % world_id))
            init_x, init_y = self.path_coord_to_gazebo_coord(*path[0])
            goal_x, goal_y = self.path_coord_to_gazebo_coord(*path[-1])
            goal_x -= init_x
            goal_y -= (init_y-1)
            self.init_position = [init_x, init_y, np.pi/2]
            self.goal_position = [goal_x, goal_y, np.pi/2]

        self.param_delta = param_delta
        self.param_init = param_init
        self.param_list = param_list
        assert len(param_delta) == len(param_init) and \
                len(param_delta) == len(param_list), 'length of params should match'

        # Launch gazebo and navigation demo
        # Should have the system enviroment source to jackal_helper
        if init_world:
            rospack = rospkg.RosPack()
            BASE_PATH = rospack.get_path('jackal_helper')
            self.gazebo_process = subprocess.Popen(['roslaunch', \
                                                    os.path.join(BASE_PATH, 'launch', 'jackal_world_navigation.launch'),
                                                    'world_name:=' + world_name,
                                                    'gui:=' + gui,
                                                    'VLP16:=' + VLP16,
                                                    'camera:=' + camera
                                                    ])
            time.sleep(10)


        rospy.set_param('/use_sim_time', True)
        rospy.init_node('gym', anonymous=True)

        self.gazebo_sim = GazeboSimulation(init_position = self.init_position)
        self.navi_stack = NavigationStack(goal_position = self.goal_position)

        self.action_space = spaces.Discrete(2**len(param_list)+1)
        self.reward_range = (-np.inf, np.inf)
        if VLP16 == 'true':
            self.observation_space = spaces.Box(low=np.array([-1]*(2095)), # a hard coding here
                                                high=np.array([1]*(2095)),
                                                dtype=np.float32)
        elif VLP16 == 'false':
            self.observation_space = spaces.Box(low=np.array([-1]*(721+len(self.param_list))), # a hard coding here
                                                high=np.array([1]*(721+len(self.param_list))),
                                                dtype=np.float32)
        if init_world:
            self._seed()
            self.navi_stack.set_global_goal()
            self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def path_coord_to_gazebo_coord(self, x, y):
        RADIUS = 0.075
        r_shift = -RADIUS - (30 * RADIUS * 2)
        c_shift = RADIUS + 5

        gazebo_x = x * (RADIUS * 2) + r_shift
        gazebo_y = y * (RADIUS * 2) + c_shift

        return (gazebo_x, gazebo_y)

    def _observation_builder(self, laser_scan, local_goal):
        '''
        Observation is the laser scan plus local goal. Episode ends when the
        between gobal goal and robot positon is less than 0.4m. Reward is set
        to -1 for each step
        '''
        scan_ranges = np.array(laser_scan.ranges)
        scan_ranges[scan_ranges > self.laser_clip] = self.laser_clip
        y = 0.001 if abs(local_goal.position.y) < 0.001 else local_goal.position.y
        local_goal_position = np.array([np.arctan(local_goal.position.x/y)])
        params = []
        params_normal = []
        for pn in self.param_list:
            params.append(self.navi_stack.get_navi_param(pn))
            params_normal.append((params[-1]-float(range_dict[pn][1])/2)/float(range_dict[pn][1])) #normalize to [-0.5, 0.5]
        state = np.concatenate([(scan_ranges-self.laser_clip/2)/self.laser_clip, (local_goal_position)/np.pi, np.array(params_normal)])

        pr = np.array([self.navi_stack.robot_config.X, self.navi_stack.robot_config.Y])
        gpl = np.array(self.goal_position[:2])
        self.gp_len = np.sqrt(np.sum((pr-gpl)**2))

        if self.gp_len < 0.4 or self._get_param('/step_count') >= self.max_step:
            done = True
        else:
            done = False

        return state, -1, done, {'params': params}

    def _set_param(self, param_name, param):
        rospy.set_param(param_name, float(param))

    def _get_param(self, param_name):
        return rospy.get_param(param_name)

    def step(self, action):
        assert action < 2**len(self.param_list) + 1
        if action < 2**len(self.param_list): # real action
            action_bin = [int(s) for s in bin(action).replace('0b', '')]
            action_bin = [0]*(len(self.param_list)-len(action_bin)) + action_bin
            assert len(action_bin) == len(self.param_list)
            i = 0
            for a, d, pn in zip(action_bin, self.param_delta, self.param_list):
                param = self.navi_stack.get_navi_param(pn)
                if a == 0:
                    param = max(range_dict[pn][0], param - d)
                elif a == 1:
                    param = min(range_dict[pn][1], param + d)
                i += 1
                self.navi_stack.set_navi_param(pn, param)
        else: # the last action is pass without doing anything
            pass

        step_count = self._get_param('/step_count')
        self._set_param('/step_count', step_count+1)

        # Unpause the world
        self.gazebo_sim.unpause()

        # Sleep for 5s (a hyperparameter that can be tuned)
        rospy.sleep(self.time_step)

        # Collect the laser scan data
        laser_scan = self.gazebo_sim.get_laser_scan()
        local_goal = self.navi_stack.get_local_goal()

        # Pause the simulation world
        self.gazebo_sim.pause()

        return self._observation_builder(laser_scan, local_goal)

    def reset(self):

        self._set_param('/step_count', 0)
        # reset robot in odom frame clear_costmap
        self.navi_stack.reset_robot_in_odom()
        # Resets the state of the environment and returns an initial observation.
        self.gazebo_sim.reset()
        # reset max_vel_x value
        for init, pn in zip(self.param_init, self.param_list):
            self.navi_stack.set_navi_param(pn, init)

        # Unpause simulation to make observation
        self.gazebo_sim.unpause()

        #read laser data
        self.navi_stack.clear_costmap()
        rospy.sleep(0.1)
        self.navi_stack.clear_costmap()

        laser_scan = self.gazebo_sim.get_laser_scan()
        local_goal = self.navi_stack.get_local_goal()
        self.navi_stack.set_global_goal()

        self.gazebo_sim.pause()

        state, _, _, _ = self._observation_builder(laser_scan, local_goal)

        return state

    def close(self):
        os.system("killall -9 rosmaster")
        os.system("killall -9 gzclient")
        os.system("killall -9 gzserver")
        os.system("killall -9 roscore")

if __name__ == '__main__':
    env = GazeboJackalNavigationEnv()
    env.reset()
    print(env.step(0))
    env.unpause()
    time.sleep(30)
    env.close()
