# APPLR

## Jackal Navigation Environment

Two OpenAI gym environments with continuous and discrete actions spaces can be found under `jackal_navi_envs`. They are registered as `jackal_discrete-v0` and `jackal_continuous-v0`. To use the environment locally, you need a workspace with all the jackal dependencies installed. Assume you already set up the Jackal simulation, one extra package `jackal_helper` need to be installed. 
* Under the parent folder of your workspace:

```
cd src
git clone https://github.com/Daffan/jackal_helper.git
cd ..
source devel/setup.bash
catkin_make
```

* The following python dependency need to be installed:
```
pip3 install gym defusedxml pyyaml pyquaternion rospkg
```

* To test the installation:

```
source <your_workspace>/devel/setup/bash
python3 scripts/test_env.py
```

Above command will run a Jackal_navigation_env 10 episodes. 

## Singularity container
* build the singularity image

`sudo singularity build --notest APPLR_melodic.simg APPLR_melodic.def`

* Run test in container

`./singularity_run.sh python3 scripts/test_env.py`

## Train td3 policy on HTCondor
* Check all the environment and training related configuration:
```
cat continuous/config/td3_condor.json
```

* Run the central learning node locally at host
```
./executable/run_central_node.sh
```
Running the central node locally is currently recommanded. A crowded cluster will usually idle your jobs. When you jobs is idled and recovered, it will initialize a new policy! Run it locally can prevent the issue. 

* Run all the actors nodes:
```
pytho3n gen_sub1.py --num_env <num_env in your config file>
```

## Test your policy on HTCondor
* Run test
```
python3 scripts/test_policy_condor.py --model <path/to/logfolder> --policy <policy_file_name> --test
```

* Get the test report
```
python3 scripts/report_test.py
```
This will print the averaging of some metrics and generate `report.json` with all test results. 


