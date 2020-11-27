from os.path import join, dirname, abspath, exists
import os
import argparse
import subprocess
from shutil import copyfile

parser = argparse.ArgumentParser(description = 'Test a TD3 policy on condor')
parser.add_argument('--model', dest = 'model', type = str, default = 'continuous/results/TD3_testbed_2020_11_15_23_03', help = 'path to the saved policy and config.json')
parser.add_argument('--policy', dest = 'policy', type = str, default = 'policy_20.pth')
parser.add_argument('--test', dest = 'test', action="store_true")

args = parser.parse_args()
BASE_PATH = join(os.getenv("HOME"), "buffer_test")

if not exists(BASE_PATH):
    os.mkdir(BASE_PATH)

config_path_src = join(args.model, "config.json")
config_path_dst = join(BASE_PATH, "config.json")
copyfile(config_path_src, config_path_dst)

policy_path_src = join(args.model, args.policy)
policy_path_dst = join(BASE_PATH, "policy.pth")
copyfile(policy_path_src, policy_path_dst)

if args.test:
    os.environ['TEST_SET'] = 'test'
    num_env = 50
else:
    os.environ['TEST_SET'] = 'train'
    num_env 250

subprocess.Popen(["python3", "gen_sub1.py", "--num_env", str(num_env), "test"])

