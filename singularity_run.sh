#!/bin/bash
export PATH=$PATH:/lusr/opt/singularity-3.2.1/bin
rm -r /u/zifan/buffer/*
rm -r out/*
rm -r ../.ros
singularity exec -i -n --network=none -p -B /var/condor:/var/condor -B `pwd`:/APPLR APPLR_melodic.simg /bin/bash /APPLR/entrypoint.sh ${@:1}
