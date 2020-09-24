# APPLR

## Singularity container
* build the singularity image

`sudo singularity build --notest APPLR_melodic.simg APPLR_melodic.def`

* Run test in container

`singularity exec -B /path/to/APPLR:/APPLR APPLR_melodic.simg /bin/bash -c 'source /jackal_ws/devel/setup.bash; python3 /APPLR/test.py'`
