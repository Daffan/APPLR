#!/bin/bash
export PATH=$PATH:/lusr/opt/singularity-3.2.1/bin
rm -r ../.ros/log/*
singularity exec -i -n --network=none -p -B `pwd`:/APPLR APPLR_melodic.simg /bin/bash /APPLR/entrypoint.sh ${@:1}
