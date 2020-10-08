#!/bin/bash
export PATH=$PATH:/lusr/opt/singularity-3.2.1/bin
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export ROS_LOG_DIR=/tmp
singularity exec -i -n --network=none -p -B /var/condor:/var/condor -B `pwd`:/APPLR APPLR_melodic.simg /bin/bash /APPLR/entrypoint.sh python3 actor.py --id ${@:1}
