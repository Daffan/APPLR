#!/bin/bash
export PATH=$PATH:/lusr/opt/singularity-3.2.1/bin
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export IMAGE_PATH=APPLR_melodic_tf.simg
singularity exec -i -n --network=none -p -B `pwd`:/APPLR ${IMAGE_PATH} /bin/bash /APPLR/entrypoint.sh ${@:1}
