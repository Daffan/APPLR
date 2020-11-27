#!/bin/bash

source /jackal_ws/devel/setup.bash
cd /APPLR
exec ${@:1}
