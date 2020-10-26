# APPLR

## Singularity container
* build the singularity image

`sudo singularity build --notest APPLR_melodic.simg APPLR_melodic.def`

* Run test in container

`./singularity_run.sh python3 ../test.py`

## Run test pipeline on condor
Create a folder `buffer_test` and make sure `BASE_PATH` in `continuous/tester.py` point to the folder. Copy the model `policy.pth` and the corresponding `config.json` to `buffer_test`.

* Change the world to test:
Check `Benchmarking_test` list in `continuous/tester.py` that's the world to test on. Test world under folder `Benchmarking/test/`, training world under folder `Benchmarking/train`

* Run default or policy
Hard coding the --default argument to be True or False to control

* Run the test
Under APPLR-1 folder run: (number for num_env argument should match with the length of `Benchmarking_test` list you defined in `continuous/tester.py`)

`python3 gen_sub1.py --num_env 50 --test`

* Get the report
Run the command and check `report.json` under folder `buffer_test`
`python3 continuous/test_benchmarking.py`



