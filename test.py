import jackal_navi_envs
from jackal_navi_envs.jackal_env_wrapper import SequentialWorldWrapper
import gym

def main():
    print('test a single episode of the environment')

    env = SequentialWorldWrapper(gym.make('jackal_discrete-v0'))

    env.reset()
    done  = False
    count = 0

    while not done:
        count += 1
        obs, rew, done, info = env.step(64)
        print('current step: %d, X position: %f' %(count, info['X']))

    env.close()

if __name__ == '__main__':
    main()


