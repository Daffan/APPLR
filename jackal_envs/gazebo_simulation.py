#!/usr/bin/env python

import rospy

from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import LaserScan
from pyquaternion import Quaternion as qt

# from tf.transformations import quaternion_from_euler

def create_model_state(x, y, z, angle):

    model_state = ModelState()
    model_state.model_name = 'jackal'
    model_state.pose.position.x = x
    model_state.pose.position.y = y
    model_state.pose.position.z = z
    e = qt(axis = [0, 0, 1], angle = angle).elements
    model_state.pose.orientation = Quaternion(e[1], e[2], e[3], e[0])
    model_state.reference_frame = "world";

    return model_state

class GazeboSimulation():

    def __init__(self, init_position = [0, 0, 0]):
        self._pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._reset = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self._model_state_getter = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        self._init_model_state = create_model_state(init_position[0],init_position[1],0,init_position[2])

    def pause(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self._pause()
        except rospy.ServiceException:
            print ("/gazebo/pause_physics service call failed")

    def unpause(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self._unpause()
        except rospy.ServiceException:
            print ("/gazebo/unpause_physics service call failed")

    def reset(self):
        # /gazebo/reset_world or /gazebo/reset_simulation will
        # destroy the world setting
        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            #reset_proxy.call()
            self._reset(self._init_model_state)
        except (rospy.ServiceException):
            print ("/gazebo/set_model_state service call failed")

    def get_laser_scan(self):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('front/scan', LaserScan, timeout=5)
            except:
                pass
        return data

    def get_model_state(self):
        rospy.wait_for_service("/gazebo/get_model_state")
        try:
            #reset_proxy.call()
            return self._model_state_getter('jackal', 'world')
        except (rospy.ServiceException):
            print ("/gazebo/get_model_state service call failed")

    def reset_init_model_state(self, init_position = [0, 0, 0]):
        self._init_model_state = create_model_state(init_position[0],init_position[1],0,init_position[2])
