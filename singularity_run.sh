#!/bin/bash
HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
name=${USER}_APPLR_${HASH}

echo "Launching container named '${name}'"
# Launches a singularity container using our image, and runs the provided command

singularity exec -i -n -p -B `pwd`:/APPLR APPLR_melodic.simg /bin/bash /APPLR/entrypoint.sh ${@:1}
