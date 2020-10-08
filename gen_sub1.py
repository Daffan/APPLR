import subprocess,re
import argparse

parser = argparse.ArgumentParser(description = 'generate the submission file')
parser.add_argument('--num_env', dest = 'num_env', type = int, default = 1, help = 'number of jobs to invoke')
parser.add_argument('--out', dest = 'out_path', type = str, default = 'out', help = 'path to the saving folder')
args = parser.parse_args()

cfile = open('condor.sub', 'w')
common_command = \
    "requirements = InMastodon \n\
+Group = \"GUEST\" \n\
+Project = \"AI_ROBOTICS\" \n\
+ProjectDescription = \"Adaptive Planner Parameter Learning From Reinforcement\" \n\
Executable = discrete/actor.sh \n\
Universe   = vanilla\n\
getenv     = true\n\
transfer_executable = false \n\n"
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
