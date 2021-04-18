from os.path import exists, join
import json
import os
import numpy as np
import time
import pickle
import sys
import logging
sys.path.append("..")
from continuous.utils import Benchmarking_train, Benchmarking_test

BASE_PATH = join(os.getenv('HOME'), 'buffer_test')
SET = os.getenv("TEST_SET")
WORLDS = Benchmarking_test if SET=="test" else list(range(300)) 
NUM_AVG = 40

def main():
    save_path = "test.txt"
    outf =  open(save_path, "w")
    worlds = []
    bad_worlds = []
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for filename in filenames:
            p = join(dirname, filename)
            if p.endswith('.pickle'):
                try:
                    with open(p, 'rb') as f:
                        traj = pickle.load(f)
                    world = int(traj[-1][-1]['world'])
                    ep_return = sum([t[2] for t in traj])
                    ep_length = len(traj)
                    success = int(traj[-1][-1]['succeed'])
                    
                    if len(filenames) == NUM_AVG and world not in worlds:
                        outf.write("%d %d %f %d\n" %(world, ep_length, ep_return, success))
                    else:
                        break
                except:
                    logging.exception("")
                    pass

        # import pdb; pdb.set_trace()
        if dirname.split("/")[-1].startswith("actor"):
            if len(filenames) == NUM_AVG:
                worlds.append(world)
            elif world not in bad_worlds:
                bad_worlds.append(world)
            else:
                print("world %s fail for all two test!" %(world))

    outf.close()
    print(worlds)
    print(bad_worlds)
    print(len(worlds)+len(bad_worlds))

if __name__ == "__main__": 
    main()


