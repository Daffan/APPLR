import jackal_navi_envs
from jackal_navi_envs.jackal_env_wrapper import SequentialWorldWrapper, BenchMarkingWrapper, BenchMarkingResartWrapper
import gym
import json

def main():
    config_path = 'discrete/configs/dqn.json'
    with open(config_path, 'rb') as f:
        config = json.load(f)
    env_config = config['env_config']
    print('test restart episodes of the environment')

    env = BenchMarkingResartWrapper(gym.make('jackal_discrete-v0', verbose = 'true', gui = 'false',\
                                          world_name = 'Benchmarking/train/world_202.world',\
                                          param_init = [0.5, 1.57, 6, 20, 0.75, 1]), env_config = env_config, switch_interval = 4)

    env.reset()
    done  = False
    count = 0

    while True:
        import random
        action = random.choice(list(range(65)))
        count += 1
        obs, rew, done, info = env.step(64)
        Y = env.navi_stack.robot_config.Y
        X = env.navi_stack.robot_config.X
        p = env.gazebo_sim.get_model_state().pose.position
        print('current step: %d, X position: %f, %f, Y position: %f, %f, rew: %f' %(count, p.x, X, p.y, Y , rew))
        if done:
            env.reset()
            print(count, rew)
            count = 0

    env.close()

if __name__ == '__main__':
    main()


