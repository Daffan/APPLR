from os.path import join, dirname, abspath
import sys

sys.path.append(dirname(abspath(__file__)))

import jackal_envs

try:
    sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
except:
    pass

from stable_baselines import DQN
from stable_baselines.common.callbacks import BaseCallback
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')


parser = argparse.ArgumentParser(description = 'Jackal navigation simulation')
parser.add_argument('--config', dest = 'config_path', type = str, default = '../configs/default.json', help = 'path to the configuration file')
parser.add_argument('--save', dest = 'save_path', type = str, default = 'results/', help = 'path to the saving folder')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--buffer-size', type=int, default=50000)

args = parser.parse_args()
config_path = args.config_path
save_path = args.save_path

with open(config_path, 'rb') as f:
    config = json.load(f)

now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M")
save_path = os.path.join(save_path, config['section'] + "_" + dt_string)
if not os.path.exists(save_path):
    os.mkdir(save_path)

with open(os.path.join(save_path, 'config.json'), 'w') as fp:
    json.dump(config, fp)

env = wrapper_dict[config['wrapper_config']['wrapper']] \
        (gym.make('jackal_navigation-v0', **config['env_config']), **config['wrapper_config']['wrapper_args'])

class SaveEveryStepIntervalsCallback(BaseCallback):

    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super(SaveEveryStepIntervalsCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.model.save(os.path.join(self.save_path, 'model_%d' %(self.n_calls)))

        return True
callback = SaveEveryStepIntervalsCallback(500, save_path)

model = DQN(config['policy_network'], env,
                    learning_rate = config['learning_rate'],
                    buffer_size = config['buffer_size'],
                    target_network_update_freq = 64,
                    gamma = config['gamma'], # policy_kwargs = config['policy_kwargs'],
                    verbose=1, tensorboard_log = save_path)

model.learn(config['total_steps'], callback = callback)
model.save(os.path.join(save_path, 'model'))

env.close()
