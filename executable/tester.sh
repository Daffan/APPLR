#!/bin/bash
export PATH=$PATH:/lusr/opt/singularity-3.2.1/bin
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export ROS_LOG_DIR=/tmp

IMAGE_PATH=APPLR_melodic_tf.simg
singularity exec -i -n --network=none -p -B /var/condor:/var/condor -B `pwd`:/APPLR ${IMAGE_PATH} /bin/bash /APPLR/entrypoint.sh python3 continuous/tester.py --id ${@:1}
