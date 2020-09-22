import rospy
import actionlib
from math import radians
import numpy as np

from std_srvs.srv import Empty
import dynamic_reconfigure.client
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Quaternion, Pose, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction
from robot_localization.srv import SetPose
from pyquaternion import Quaternion as qt

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Path, Odometry
import scipy.signal
import time

def _create_global_goal(x, y, angle):
    """
    Create a MoveBaseGoal with x, y position and yaw rotation (in degrees).
    Returns a MoveBaseGoal
    """
    mb_goal = MoveBaseGoal()
    mb_goal.target_pose.header.frame_id = 'odom' # Note: the frame_id must be map
    mb_goal.target_pose.pose.position.x = x
    mb_goal.target_pose.pose.position.y = y
    mb_goal.target_pose.pose.position.z = 0 # z must be 0.0 (no height in the map)

    e = qt(axis = [0, 0, 1], angle = angle).elements
    mb_goal.target_pose.pose.orientation = Quaternion(e[1], e[2], e[3], e[0])

    return mb_goal

def create_PoseWithCovarianceStamped():
    a = PoseWithCovarianceStamped()
    a.header.frame_id = 'odom'
    a.pose.pose.position.x = 0.0
    a.pose.pose.position.y = 0.0
    a.pose.pose.position.z = 0.0
    a.pose.pose.orientation.x = 0.0
    a.pose.pose.orientation.y = 0.0
    a.pose.pose.orientation.z = 0.0
    a.pose.pose.orientation.w = 0.0
    return a

class Robot_config():
    def __init__(self):
        self.X = 0 # inertia frame
        self.Y = 0
        self.Z = 0
        self.PSI = 0
        self.global_path = []
        self.gx = 0 # body frame
        self.gy = 0
        self.gp = 0
        self.los = 1
        self.bad_vel = 0
        self.vel_counter = 0
        # self.los = 5

    def get_robot_status(self, msg):
        q1 = msg.pose.pose.orientation.x
        q2 = msg.pose.pose.orientation.y
        q3 = msg.pose.pose.orientation.z
        q0 = msg.pose.pose.orientation.w
        self.X = msg.pose.pose.position.x
        self.Y = msg.pose.pose.position.y
        self.Z = msg.pose.pose.position.z
        self.PSI = np.arctan2(2 * (q0*q3 + q1*q2), (1 - 2*(q2**2+q3**2)))
        #print(self.X, self.Y, self.PSI)

    def get_global_path(self, msg):
        #print(msg.poses)
        # self.global_path = []
        gp = []
        for pose in msg.poses:
            gp.append([pose.pose.position.x, pose.pose.position.y])
            # self.global_path.append([pose.pose.position.x, pose.pose.position.y])
        #print(len(self.global_path))
        gp = np.array(gp)
        x = gp[:,0]
        # xhat = x
        try:
            xhat = scipy.signal.savgol_filter(x, 19, 3)
        except:
            xhat = x
        y = gp[:,1]
        # yhat = y
        try:
            yhat = scipy.signal.savgol_filter(y, 19, 3)
        except:
            yhat = y
        # plt.figure()
        # plt.plot(xhat, yhat, 'k', linewidth=1)
        # plt.axis('equal')
        # plt.savefig("/home/xuesu/gp_plt.png")
        # plt.close()
        gphat = np.column_stack((xhat, yhat))
        gphat.tolist()
        self.global_path = gphat
        # print(self.global_path)

    def vel_monitor(self, msg):
        vx = msg.linear.x
        if vx < 0.1:
            self.bad_vel += 1
        self.vel_counter += 1





def transform_lg(wp, X, Y, PSI):
    R_r2i = np.matrix([[np.cos(PSI), -np.sin(PSI), X], [np.sin(PSI), np.cos(PSI), Y], [0, 0, 1]])
    R_i2r = np.linalg.inv(R_r2i)
    #print(R_r2i)
    pi = np.matrix([[wp[0]], [wp[1]], [1]])
    pr = np.matmul(R_i2r, pi)
    #print(pr)
    lg = np.array([pr[0,0], pr[1, 0]])
    #print(lg)
    return lg

class NavigationStack():

    def __init__(self, goal_position = [6, 6, 0]):
        self.client = dynamic_reconfigure.client.Client('move_base/TrajectoryPlannerROS')
        self.nav_as = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        self.global_goal = _create_global_goal(goal_position[0], goal_position[1], goal_position[2])
        self._reset_odom = rospy.ServiceProxy('/set_pose', SetPose)
        self._clear_costmap = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)

        self.robot_config = Robot_config()
        self.sub_robot = rospy.Subscriber("/odometry/filtered", Odometry, self.robot_config.get_robot_status)
        self.sub_gp = rospy.Subscriber("/move_base/TrajectoryPlannerROS/global_plan", Path, self.robot_config.get_global_path)
        self.sub_vel = rospy.Subscriber("/jackal_velocity_controller/cmd_vel", Twist, self.robot_config.vel_monitor)

    def set_navi_param(self, param_name, param):

        self.client.update_configuration({param_name: param})
        rospy.set_param('/move_base/TrajectoryPlannerROS/' + param_name, param)

        if param_name == 'max_vel_theta':
            self.client.update_configuration({'min_vel_theta': -param})
            rospy.set_param('/move_base/TrajectoryPlannerROS/' + 'min_vel_theta', -param)

    def get_navi_param(self, param_name):
        param = rospy.get_param('/move_base/TrajectoryPlannerROS/' + param_name)
        return param

    # def get_local_goal(self):

    #    local_goal = None
    #    while local_goal is None:
    #        local_goal = rospy.wait_for_message('/local_goal', Pose)
    #    return local_goal

    def set_global_goal(self):
        self.nav_as.wait_for_server()
        try:
            self.nav_as.send_goal(self.global_goal)
            # print("Published globe goal position!")
        except (rospy.ServiceException) as e:
            print ("/move_base service call failed")

    def reset_robot_in_odom(self):
        rospy.wait_for_service('/set_pose')
        try:
            self._reset_odom(create_PoseWithCovarianceStamped())
        except rospy.ServiceException:
            print ("/set_pose service call failed")

    def clear_costmap(self):
        rospy.wait_for_service('/move_base/clear_costmaps')
        try:
            self._clear_costmap()
        except rospy.ServiceException:
            print ("/clear_costmaps service call failed")

    def reset_global_goal(self, goal_position = [6, 6, 0]):
        self.global_goal = _create_global_goal(goal_position[0], goal_position[1], goal_position[2])

    def punish_rewrad(self):
        try:
            rew =  self.robot_config.bad_vel / self.robot_config.vel_counter * 10.
        except:
            rew = 0
        self.robot_config.bad_vel = 0
        self.robot_config.vel_counter = 0
        return -rew

    def get_local_goal(self):
        gp = self.robot_config.global_path
        X = self.robot_config.X
        Y = self.robot_config.Y
        PSI = self.robot_config.PSI
        los = self.robot_config.los

        #if len(gp)==0:
        lg_x = 0
        lg_y = 0
        #else:
        if len(gp)>0:
            lg_flag = 0
            for wp in gp:
                dist = (np.array(wp)-np.array([X, Y]))**2
                dist = np.sum(dist, axis=0)
                dist = np.sqrt(dist)
                if dist > los:
                    lg_flag = 1
                    lg = transform_lg(wp, X, Y, PSI)
                    lg_x = lg[0]
                    lg_y = lg[1]
                    break
            if lg_flag == 0:
                lg = transform_lg(gp[-1], X, Y, PSI)
                lg_x = lg[0]
                lg_y = lg[1]

        # print(lg_x, lg_y)
        local_goal = Pose()
        local_goal.position.x = lg_x
        local_goal.position.y = lg_y
        local_goal.orientation.w = 1
        return local_goal
