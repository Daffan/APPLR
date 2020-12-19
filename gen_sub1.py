################################################################################
# The script generates submission files and submit them to HTCondor.
# Submission files are composed actor.sh and run_central_node.py.
# There is an option to run central node locally (comment line 43, 44)
################################################################################

import subprocess,re
import argparse
import os
import time

parser = argparse.ArgumentParser(description = 'generate the submission file')
parser.add_argument('--num_env', dest = 'num_env', type = int, default = 1, help = 'number of jobs to invoke')
parser.add_argument('--out', dest = 'out_path', type = str, default = 'out', help = 'path to the saving folder')
parser.add_argument('--test', dest = 'test', action = 'store_true', help = 'run testers')
parser.add_argument('--sac', dest = 'sac', action = 'store_true', help = 'run sac')
args = parser.parse_args()

if not os.path.exists('out'):
    os.mkdir('out')

cfile = open('central_node.sub', 'w')
if not args.sac:
    s = 'executable/run_central_node.sh'
else:
    s = 'executable/run_central_nodei_sac.sh'
common_command = \
"requirements = InMastodon \n\
+Group = \"GRAD\" \n\
+Project = \"AI_ROBOTICS\" \n\
+ProjectDescription = \"Adaptive Planner Parameter Learning From Reinforcement\" \n\
Executable = %s \n\
Universe   = vanilla\n\
getenv     = true\n\
transfer_executable = false \n\n" %(s)
cfile.write(common_command)
# Loop over various values of an argument and create different output file for each
# Then put it in the queue
s = args.out_path

run_command = \
    'output     = out/out.txt\n\
    log        = out/log.txt\n\
    error      = out/err.txt\n\
    queue 1\n\n'
cfile.write(run_command)
cfile.close()
#if not args.test:
#    subprocess.run(["condor_submit", "central_node.sub"])

#time.sleep(10)

cfile = open('condor.sub', 'w')
if args.sac:
    undersore = "_sac"
else:
    undersore = ""

s = 'executable/tester%s.sh' %(undersore) if args.test else 'executable/actor%s.sh' %(undersore)
common_command = \
"requirements = InMastodon \n\
+Group = \"GRAD\" \n\
+Project = \"AI_ROBOTICS\" \n\
+ProjectDescription = \"Adaptive Planner Parameter Learning From Reinforcement\" \n\
Executable = %s \n\
Universe   = vanilla\n\
getenv     = true\n\
transfer_executable = false \n\n" %(s)
cfile.write(common_command)
# Loop over various values of an argument and create different output file for each
# Then put it in the queue
s = args.out_path
for a in range(args.num_env):
    run_command = \
        'arguments  = %d\n\
        output     = %s/out.%d.txt\n\
        log        = %s/log.%d.txt\n\
        error      = %s/err.%d.txt\n\
        queue 1\n\n' % (a, s, a, s, a, s, a)
    cfile.write(run_command)
cfile.close()
subprocess.run(["condor_submit", "condor.sub"])
