#!/bin/bash
export PATH=$PATH:/lusr/opt/singularity-3.2.1/bin
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
singularity exec -i -n --network=none -p -B `pwd`:/APPLR ../APPLR/APPLR_melodic.simg /bin/bash /APPLR/entrypoint.sh ${@:1}
