import os
import torch
import numpy as np
from torch import nn


param_dict = {
    0: {'max_vel_x': 0.5,
        'max_vel_theta': 1.57,
        'vx_samples': 6,
        'vtheta_samples': 20,
        'occdist_scale': 0.1,
        'path_distance_bias': 0.75,
        'goal_distance_bias': 1.00,
        'inflation_radius': 0.3
        },  # default
    1: {'max_vel_x': 0.26,
        'max_vel_theta': 2,
        'vx_samples': 13,
        'vtheta_samples': 44,
        'occdist_scale': 0.57,
        'path_distance_bias': 0.76,
        'goal_distance_bias': 0.94,
        'inflation_radius': 0.02
        },  # curve
    2: {'max_vel_x': 1.91,
        'max_vel_theta': 1.70,
        'vx_samples': 10,
        'vtheta_samples': 47,
        'occdist_scale': 0.08,
        'path_distance_bias': 0.71,
        'goal_distance_bias': 0.35,
        'inflation_radius': 0.23
        },  # open_space
    3: {'max_vel_x': 0.72,
        'max_vel_theta': 0.73,
        'vx_samples': 19,
        'vtheta_samples': 59,
        'occdist_scale': 0.62,
        'path_distance_bias': 1.00,
        'goal_distance_bias': 0.32,
        'inflation_radius': 0.24
        },  # narrow_entrance
    4: {'max_vel_x': 0.22,
        'max_vel_theta': 0.87,
        'vx_samples': 13,
        'vtheta_samples': 31,
        'occdist_scale': 0.30,
        'path_distance_bias': 0.36,
        'goal_distance_bias': 0.71,
        'inflation_radius': 0.3
        },  # narrow_corridor
}


class Config(object):
    def __init__(self,):
        self.laser_clip = 5
        self.num_layers = 1
        self.hidden_size = 128
        self.lg_cum_len = 0.3
        self.load_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "APPLI.pickle")


class Net(nn.Module):
    """Simple MLP backbone.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    :param bool concat: whether the input shape is concatenated by state_shape
        and action_shape. If it is True, ``action_shape`` is not the output
        shape, but affects the input shape.
    :param bool dueling: whether to use dueling network to calculate Q values
        (for Dueling DQN), defaults to False.
    :param norm_layer: use which normalization before ReLU, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``, defaults to None.
    :param int num_atoms: in order to expand to the net of distributional RL,
         defaults to 1.
    """

    def __init__(
        self,
        state_shape,
        action_shape,
        layer_num,
        hidden_layer_size,
        device,
    ):
        super(Net, self).__init__()
        self.device = device
        self.action_num = np.prod(action_shape)
        input_size = np.prod(state_shape)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_layer_size))
        for i in range(layer_num - 1):
            self.layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
        self.output_layer = nn.Linear(hidden_layer_size, action_shape)

    def forward(self, state):
        h = state
        for layer in self.layers:
            h = torch.relu(layer(h))
        output = torch.sigmoid(self.output_layer(h))
        return output


class APPLE_policy(nn.Module):

    def __init__(self,):
        super(APPLE_policy, self).__init__()
        self.param = Config()
        self.device = torch.device("cpu")
        self.model = Net((721,), 5, self.param.num_layers, self.param.hidden_size, self.device)
        self.eps = 0.0
        # self.load_state_dict(torch.load(self.param.load_model))

    def set_eps(self, eps):
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def train(self, mode=True):
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.train(mode)
        return self

    def preprocess_scan(self, scan):
        scan = np.array(scan)
        scan[scan > self.param.laser_clip] = self.param.laser_clip
        scan = scan / self.param.laser_clip - 0.5
        return scan

    def preprocess_gp(self, gp):
        lg = np.zeros(2)
        prev_x, prev_y = 0, 0
        cum_dist = 0
        if len(gp) > 0:
            for wp in gp:
                dist = np.linalg.norm(np.array(wp) - np.array([prev_x, prev_y]))
                prev_x, prev_y = wp[0], wp[1]
                cum_dist += dist
                wp /= np.linalg.norm(wp)        # normalize each way point
                lg += wp
                if cum_dist >= self.param.lg_cum_len:
                    break
        if np.linalg.norm(lg) != 0:
            lg /= np.linalg.norm(lg)

        return lg

    def forward(self, obs):
        scan, gp = obs
        scan = self.preprocess_scan(scan)
        local_goal = self.preprocess_gp(gp)
        local_goal = np.arctan2(local_goal[1], local_goal[0])
        obs = np.concatenate([scan, [local_goal]])[None, :]
        obs = torch.from_numpy(obs.astype(np.float32)).to(self.device)
        q = self.model(obs)
        action = q.argmax(dim=-1).cpu().detach().numpy()
        if not np.isclose(self.eps, 0.0):
            for i in range(len(q)):
                if np.random.rand() < self.eps:
                    q_ = np.random.rand(*q[i].shape)
                    action[i] = q_.argmax()
        action = action[0]

        param_set = param_dict[action]
        param_set = np.array([param_set["max_vel_x"],
                              param_set["max_vel_theta"],
                              param_set["vx_samples"],
                              param_set["vtheta_samples"],
                              param_set["occdist_scale"],
                              param_set["path_distance_bias"],
                              param_set["goal_distance_bias"],
                              param_set["inflation_radius"]])
        return param_set


if __name__ == "__main__":
    import rospy
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Path, Odometry
    import dynamic_reconfigure.client
    from context_classifier import Predictor

    policy = APPLE_policy()
    policy.used_context = None

    predictor = Predictor(policy)

    rospy.init_node('context_classifier', anonymous=True)

    def test(msg):
        if len(predictor.global_path) == 0:
            return
        scan = msg.ranges
        param = policy.forward([scan, predictor.global_path.T])
        print("param", param)

    sub_robot = rospy.Subscriber("/odometry/filtered", Odometry, predictor.update_status)
    sub_gp = rospy.Subscriber("/move_base/TrajectoryPlannerROS/global_plan",
                              Path, predictor.update_global_path, queue_size=1)
    sub_scan = rospy.Subscriber("/front/scan", LaserScan, test, queue_size=1)

    client = dynamic_reconfigure.client.Client('move_base/TrajectoryPlannerROS')
    client2 = dynamic_reconfigure.client.Client('move_base/local_costmap/inflater_layer')
    while not rospy.is_shutdown():
        pass
