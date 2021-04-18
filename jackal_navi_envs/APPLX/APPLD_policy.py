#!/usr/bin/env python
import os
import scipy
import numpy as np
from scipy import stats

from .classifier import ScanClassifier, preprocess_scan, preprocess_gp


def get_params():
    curve = {
        'max_vel_x': 0.26,
        'max_vel_theta': 2,
        'vx_samples': 13,
        'vtheta_samples': 44,
        'occdist_scale': 0.57,
        'path_distance_bias': 0.76,
        'goal_distance_bias': 0.94,
    }
    open_space = {
        'max_vel_x': 1.91,
        'max_vel_theta': 1.70,
        'vx_samples': 10,
        'vtheta_samples': 47,
        'occdist_scale': 0.08,
        'path_distance_bias': 0.71,
        'goal_distance_bias': 0.35,
    }
    U_turn = {
        'max_vel_x': 0.45,
        'max_vel_theta': 1.02,
        'vx_samples': 20,
        'vtheta_samples': 30,
        'occdist_scale': 0.82,
        'path_distance_bias': 0.88,
        'goal_distance_bias': 0.43,
    }
    narrow_entrance = {
        'max_vel_x': 0.72,
        'max_vel_theta': 0.73,
        'vx_samples': 19,
        'vtheta_samples': 59,
        'occdist_scale': 0.62,
        'path_distance_bias': 1.00,
        'goal_distance_bias': 0.32,
    }
    narrow_corridor = {
        'max_vel_x': 0.22,
        'max_vel_theta': 0.87,
        'vx_samples': 13,
        'vtheta_samples': 31,
        'occdist_scale': 0.30,
        'path_distance_bias': 0.36,
        'goal_distance_bias': 0.71,
    }
    normal1 = {
        'max_vel_x': 0.37,
        'max_vel_theta': 1.33,
        'vx_samples': 9,
        'vtheta_samples': 6,
        'occdist_scale': 0.95,
        'path_distance_bias': 0.83,
        'goal_distance_bias': 0.93,
    }
    normal2 = {
        'max_vel_x': 0.31,
        'max_vel_theta': 1.05,
        'vx_samples': 17,
        'vtheta_samples': 20,
        'occdist_scale': 0.45,
        'path_distance_bias': 0.61,
        'goal_distance_bias': 0.22,
    }
    default = {
        'max_vel_x': 0.5,
        'max_vel_theta': 1.57,
        'vx_samples': 6,
        'vtheta_samples': 20,
        'occdist_scale': 0.1,
        'path_distance_bias': 0.75,
        'goal_distance_bias': 1.0,
    }
    env_params = [curve, open_space, U_turn, narrow_entrance, narrow_corridor, normal1, normal2, default]
    inflates = [0.02, 0.23, 0.005, 0.24, 0.30, 0.01, 0.23, 0.30]
    return env_params, inflates


class ScanClassifierParams:
    def __init__(self):
        self.Dx = 640

        self.used_context = ["curve", "open_space"]
        self.full_train = False

        self.use_conv1d = True
        self.scan_Dhs = [128]
        self.gp_Dhs = [0]
        self.kernel_sizes = [40]
        self.filter_sizes = [32]
        self.strides = [5]

        self.dropout_rate = 0.2
        self.use_EDL = False

        self.lr = 3e-4
        self.epochs = 500
        self.batch_size = 32

        self.use_weigth = False
        self.clipping = 5.0
        self.centering_on_gp = False
        self.cropping = True
        self.theta_noise_scale = 10  # in degree
        self.noise = True
        self.noise_scale = 0.05
        self.flipping = True
        self.translation = True
        self.translation_scale = 0.10

        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "APPLD.pickle")
        self.rslts_dir = None
        self.Dy = len(self.used_context)


class APPLD_policy(object):
    def __init__(self):
        scan_classifier_params = ScanClassifierParams()
        scan_classifier = ScanClassifier(scan_classifier_params)
        scan_classifier._init_model()
        scan_classifier._load_model(scan_classifier_params.model_path)
        self.scan_classifier = scan_classifier

        self.env_params, self.env_inflates = get_params()

    def forward(self, obs):
        scan, global_path = obs
        scan = np.array(scan)
        if len(global_path) == 0:
            return
        envs = ['curve', 'open_space', 'U_turn', 'narrow_entrance', 'narrow_corridor',
                'normal_1', 'normal_2', 'default']

        scan = preprocess_scan(scan, self.scan_classifier)
        global_path = preprocess_gp(global_path.T)
        context_type = self.scan_classifier.predict(scan, global_path)
        ctx = self.scan_classifier.used_context[context_type]
        context_type = envs.index(ctx)
        print("[INFO] current context: ", envs[context_type])

        action = np.array([self.env_params[context_type]['max_vel_x'],
                           self.env_params[context_type]['max_vel_theta'],
                           self.env_params[context_type]['vx_samples'],
                           self.env_params[context_type]['vtheta_samples'],
                           self.env_params[context_type]['occdist_scale'],
                           self.env_params[context_type]['path_distance_bias'],
                           self.env_params[context_type]['goal_distance_bias'],
                           self.env_inflates[context_type]])

        return action


if __name__ == "__main__":
    import rospy
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Path, Odometry
    import dynamic_reconfigure.client
    from context_classifier import Predictor

    policy = APPLD_policy()

    predictor = Predictor(policy.scan_classifier)

    rospy.init_node('context_classifier', anonymous=True)
    env_params, env_inflates = get_params()

    def test(msg):
        if len(predictor.global_path) == 0:
            return
        scan = msg.ranges
        policy.forward([scan, predictor.global_path])

    sub_robot = rospy.Subscriber("/odometry/filtered", Odometry, predictor.update_status)
    sub_gp = rospy.Subscriber("/move_base/TrajectoryPlannerROS/global_plan",
                              Path, predictor.update_global_path, queue_size=1)
    sub_scan = rospy.Subscriber("/front/scan", LaserScan, test, queue_size=1)

    client = dynamic_reconfigure.client.Client('move_base/TrajectoryPlannerROS')
    client2 = dynamic_reconfigure.client.Client('move_base/local_costmap/inflater_layer')
    while not rospy.is_shutdown():
        try:
            ct = predictor.context_type
            params = env_params[ct]
            infla = env_inflates[ct]
            config = client.update_configuration(params)
            config2 = client2.update_configuration({'inflation_radius': infla})
        except dynamic_reconfigure.DynamicReconfigureCallbackException:
            continue
        except rospy.exceptions.ROSInterruptException:
            break


