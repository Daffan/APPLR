#!/bin/bash
export PATH=$PATH:/lusr/opt/singularity-3.2.1/bin
export ROS_LOG_DIR=/tmp
singularity exec -i --nv -n --network=none -p -B `pwd`:/APPLR APPLR_melodic.simg /bin/bash /APPLR/entrypoint.sh python3 actor.py --id ${@:1}
