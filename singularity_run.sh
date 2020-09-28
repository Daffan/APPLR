#!/bin/bash

singularity exec -i -n -p -B `pwd`:/APPLR APPLR_melodic.simg /bin/bash /APPLR/entrypoint.sh ${@:1}
