#!/bin/bash
source /jackal_ws/devel/setup.bash
cd /APPLR
roscore &
exec ${@:1}
