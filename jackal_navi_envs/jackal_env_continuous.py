import gym
import rospy
import rospkg
import roslaunch
import time
import numpy as np
import os
from os.path import dirname, join, abspath
import subprocess

from gym import utils, spaces
from std_srvs.srv import Empty
import actionlib
from gym.utils import seeding

from .gazebo_simulation import GazeboSimulation
from .navigation_stack import  NavigationStack

gym.logger.set_level(40)
'''
range_dict = {
    'max_vel_x': [0.2, 1.8],
    'max_vel_theta': [0.57, 2.57],
    'vx_samples': [3, 9],
    'vtheta_samples': [8, 32],
    'path_distance_bias': [0.25, 1],
    'goal_distance_bias': [0.5, 1.5],
    'inflation_radius': [0.1, 0.5]
}
'''
# range_dict defines the possible range of the parameters. 
# also the range the policy could set the parameters. 
range_dict = {
    'max_vel_x': [0.2, 2],
    'max_vel_theta': [0.314, 3.14],
    'vx_samples': [4, 12],
    'vtheta_samples': [8, 40],
    'path_distance_bias': [0.1, 1.5],
    'goal_distance_bias': [0.1, 2],
    'inflation_radius': [0.1, 0.6]
}

class JackalEnvContinuous(gym.Env):

    def __init__(self, world_name = 'sequential_applr_testbe.world', VLP16 = 'false', gui = 'false', camera = 'false',
                init_position = [-8, 0, 0], goal_position = [54, 0, 0], max_step = 600, time_step = 1, laser_clip = 4,
                param_delta = [0.2, 0.3, 1, 2, 0.2, 0.2], param_init = [0.5, 1.57, 6, 20, 0.75, 1],
                param_list = ['max_vel_x', 'max_vel_theta', 'vx_samples', 'vtheta_samples', 'path_distance_bias', 'goal_distance_bias'],
                ):
        """
        Core RL environment of Jackal navigation
        :param world_name [str]: path to the world file. The base path is jackal_helper/worlds.
            jackal_helper is a supplementary ROS package, which has all the config and world files.
            see https://github.com/Daffan/jackal_helper.git
        :param VLP16 [str]: whether to use valodyne sensor. 'true' or 'false' in string format
        :param gui [str]: whether to invoke with gui. 'true' or 'false' in string format
        :param init_position [list [float]]: init position in world frame
        :param goal_position [list [float]]: relative goal postion in odom frame
        :param max_step [int]: maximum time step
        :param time_step [int]: number of second for each time step
        :param laser_clip [float]: clip the value of laser scan
        :param camera [str]: the feature is deprecated. whether to attach a camera to the robot.
        :param param_delta [list [float]]: feature is deprecated, used for discrete setting.
        :param param_init [list [float]]: initial values of paramters
        :param param_list [list [str]]: list of paramters name the agent will tune
        """
        gym.Env.__init__(self)

        self.world_name = world_name
        self.VLP16 = True if VLP16=='true' else False
        self.gui = True if gui=='true' else False
        self.max_step = max_step
        self.time_step = time_step
        self.laser_clip = laser_clip
        if not world_name.startswith('Benchmarking'):
            # regular world, use init_position from user's input
            self.goal_position = goal_position
            self.init_position = init_position
        else:
            # benchmarking world, load the pre-defined starting position
            base = dirname(abspath(__file__))
            world_id = int(world_name.split('_')[-1].split('.')[0])
            path = np.load(join(base, 'path_files', 'path_%d.npy' % world_id))
            init_x, init_y = self.path_coord_to_gazebo_coord(*path[0])
            goal_x, goal_y = self.path_coord_to_gazebo_coord(*path[-1])
            init_y -= 1
            goal_x -= init_x
            goal_y -= (init_y-5) # put the goal 5 meters backward
            self.init_position = [init_x, init_y, np.pi/2]
            self.goal_position = [goal_x, goal_y, 0] 

        self.param_delta = param_delta
        self.param_init = param_init
        self.param_list = param_list
        assert len(param_delta) == len(param_init) and \
                len(param_delta) == len(param_list), 'length of params should match'

        # Launch gazebo and navigation demo
        # Should have the system enviroment source to jackal_helper
        rospy.logwarn(">>>>>>>>>>>>>>>>>> Load world: %s <<<<<<<<<<<<<<<<<<" %(world_name))
        rospack = rospkg.RosPack()
        BASE_PATH = rospack.get_path('jackal_helper')
        self.gazebo_process = subprocess.Popen(['roslaunch', \
                                                os.path.join(BASE_PATH, 'launch', 'jackal_world_navigation.launch'),
                                                'world_name:=' + world_name,
                                                'gui:=' + gui,
                                                'VLP16:=' + VLP16,
                                                'camera:=' + camera,
                                                'verbose:=' + 'false' 
                                                ])
        time.sleep(10)


        rospy.init_node('gym', anonymous=True, log_level=rospy.FATAL)
        rospy.set_param('/use_sim_time', True)
        #rospy.set_param("/move_base/global_costmap/rolling_window", True)
        #rospy.set_param("/move_base/global_costmap/static_map", False)

        self.gazebo_sim = GazeboSimulation(init_position = self.init_position)
        self.navi_stack = NavigationStack(goal_position = self.goal_position)

        self.action_space = spaces.Box(low=np.array([range_dict[k][0] for k in self.param_list]),
                                                high=np.array([range_dict[k][1] for k in self.param_list]),
                                                dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)
        if VLP16 == 'true':
            self.observation_space = spaces.Box(low=np.array([-1]*(2095)), # a hard coding here
                                                high=np.array([1]*(2095)),
                                                dtype=np.float32)
        elif VLP16 == 'false':
            self.observation_space = spaces.Box(low=np.array([-1]*(721+len(self.param_list))), # a hard coding here
                                                high=np.array([1]*(721+len(self.param_list))),
                                                dtype=np.float32)
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
        Observation is the laser scan, local goal and all the values of paramters. 
        Episode ends when the between gobal goal and robot positon is less than 0.4m. 
        Reward is set to -1 for each step
        '''
        scan_ranges = np.array(laser_scan.ranges)
        scan_ranges[scan_ranges > self.laser_clip] = self.laser_clip
        local_goal_position = np.array([np.arctan2(local_goal.position.y, local_goal.position.x)])
        params = []
        params_normal = []
        for pn in self.param_list:
            params.append(self.navi_stack.get_navi_param(pn))
            params_normal.append((params[-1]-float(range_dict[pn][1])/2)/float(range_dict[pn][1])) #normalize to [-0.5, 0.5]
        state = np.concatenate([(scan_ranges-self.laser_clip/2)/self.laser_clip, (local_goal_position)/np.pi, np.array(params_normal)])

        # check the robot distance to the goal position
        pr = np.array([self.navi_stack.robot_config.X, self.navi_stack.robot_config.Y])
        gpl = np.array(self.goal_position[:2])
        self.gp_len = np.sqrt(np.sum((pr-gpl)**2))
        # terminate when the ditance is less than 0.4 meter or
        # exceed the maximal time step
        if self.gp_len < 0.4 or self.step_count >= self.max_step:
            done = True
        else:
            done = False

        return state, -self.time_step, done, {'params': params, 'succeed': self.step_count < self.max_step}

    def _set_param(self, param_name, param):
        rospy.set_param(param_name, float(param))

    def _get_param(self, param_name):
        return rospy.get_param(param_name)

    def step(self, action):


        for param_value, param_name in zip(action, self.param_list):

            if 'samples' in param_name:
                param_value = int(np.rint(param_value))
            else:
                param_value = float(param_value)

            if param_value < range_dict[param_name][0]:
                param_value = range_dict[param_name][0]
            elif param_value > range_dict[param_name][1]:
                param_value = range_dict[param_name][1]
            self.navi_stack.set_navi_param(param_name, param_value)

        self.step_count+=1

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

        self.step_count=0
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
