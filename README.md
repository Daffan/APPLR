# APPLR

## Singularity container
* build the singularity image

`sudo singularity build --notest APPLR_melodic.simg APPLR_melodic.def`

* Run test in container

`./singularity_run.sh python3 ../test.py`
