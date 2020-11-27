#############################################
# This script tests the jackal navigation environment with random action
#############################################

import jackal_navi_envs
from jackal_navi_envs.jackal_env_wrapper import SequentialWorldWrapper, BenchMarkingWrapper, BenchMarkingWrapperReward
import gym
import random
import numpy as np

def main():
    env = BenchMarkingWrapperReward(gym.make('jackal_continuous-v0', verbose = 'true',\
                                          world_name = 'Benchmarking/train/world_202.world',\
                                          param_delta = [0.2, 0.3, 1, 2, 0.2, 0.2, 0.05], \
                                          param_init = [0.5, 1.57, 6, 20, 0.75, 1, 0.3], \
                                          param_list = ["max_vel_x", "max_vel_theta", "vx_samples", "vtheta_samples", "path_distance_bias", "goal_distance_bias", "inflation_radius"]),
                                    goal_distance_reward = 1, success_reward = 50, \
                                    smoothness = 0.05, prevent_extreme = 0.05, stuck_punishment = 0.1, \
                                    punishment_reward = -50, reward_scale = 1)

    env.reset()
    done  = False
    count = 0
    ep_rew = 0

    high = env.action_space.high
    low = env.action_space.low
    bias = (high + low) / 2
    scale = (high - low) / 2

    for _ in range(10): # run 10 episode

        actions = 2*(np.random.rand(7) - 0.5)
        actions *= scale
        actions += bias

        count += 1
        obs, rew, done, info = env.step(actions)
        ep_rew += rew
        Y = env.navi_stack.robot_config.Y
        X = env.navi_stack.robot_config.X
        p = env.gazebo_sim.get_model_state().pose.position
        print('current step: %d, X position: %f, %f, Y position: %f, %f, rew: %f' %(count, p.x, X, p.y, Y , rew))
        print(actions)
        if done:
            env.reset()
            print(count, ep_rew)
            count = 0
            ep_rew = 0

    env.close()

if __name__ == '__main__':
    main()


