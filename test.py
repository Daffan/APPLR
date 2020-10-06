import jackal_navi_envs
from jackal_navi_envs.jackal_env_wrapper import SequentialWorldWrapper, BenchMarkingWrapper
import gym

def main():
    print('test a single episode of the environment')

    env = BenchMarkingWrapper(gym.make('jackal_discrete-v0', verbose = 'true',\
                                          world_name = 'Benchmarking/train/world_299.world',\
                                          param_init = [0.2, 1.57, 6, 20, 0.75, 1]))

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


