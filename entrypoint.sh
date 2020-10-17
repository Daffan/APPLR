#!/bin/bash

source /jackal_ws/devel/setup.bash
cd /APPLR/continuous
#cd /APPLR/discrete
exec ${@:1}
