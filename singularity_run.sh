#!/bin/bash
export PATH=$PATH:/lusr/opt/singularity-3.2.1/bin
<<<<<<< HEAD
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
singularity exec -i -n --network=none -p -B `pwd`:/APPLR ../APPLR/APPLR_melodic.simg /bin/bash /APPLR/entrypoint.sh ${@:1}
=======
rm -r ../.ros/log/*
singularity exec -i -n --network=none -p -B `pwd`:/APPLR APPLR_melodic.simg /bin/bash /APPLR/entrypoint.sh ${@:1}
>>>>>>> 6d70b141e86f4679b4f0f61162189ad35b91aa59
