from os.path import exists, join
import json
import os
import numpy as np
import torch
import time
import pickle

BASE_PATH = '/u/zifan/APPLR-1/continuous/buffer_test'
benchmarking_test = [0, 8, 17, 19, 27, 32, 41, 47, 48, 57, 64, 69, 76, 78, 88, 93, 100, 104, 112, 118, 123, 129, 133, 138, 144, 150, 159, 163, 168, 175, 184, 189, 193, 201, 208, 214, 218, 226, 229, 237, 240, 246, 256, 258, 265, 270, 277, 284, 290, 294]
#benchmarking_test = [i for i in list(range(300)) if i not in benchmarking_test] # all the training world
#assert len(benchmarking_test)==250

def main():
    def get_world_name(dirname):
        idx = benchmarking_test[int(dirname.split('_')[-1])]
        return 'world_' + str(idx)


    result = {}
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for d in dirnames:
            if d.startswith('actor'):
                idx = benchmarking_test[int(d.split('_')[-1])]
                result['world_' + str(idx)] = {
                    'ep_return': [],
                    'ep_length': [],
                    'succeed': []
                }
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for filename in filenames:
            p = join(dirname, filename)
            if p.endswith('.pickle'):
                with open(p, 'rb') as f:
                    traj = pickle.load(f)
                world = get_world_name(dirname)
                result[world]['ep_return'].append(sum([t[2] for t in traj]))
                result[world]['ep_length'].append(len(traj))
                result[world]['succeed'].append(int(traj[-1][-1]['succeed']))

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>> Report <<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print('supported samples: %f per world' %(np.mean([len(result[k]['ep_return']) for k in result.keys()])))
    for k2 in ['ep_return', 'ep_length', 'succeed']:
        k1 = result.keys()
        avg = np.mean([np.mean(result[k][k2]) for k in k1 if result[k][k2]])
        print('Avg %s: %f' %(k2, avg))

    with open(join(BASE_PATH, 'report.json'), 'w') as fp:
        json.dump(result, fp)

if __name__ == "__main__":
    main()


