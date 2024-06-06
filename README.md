## Optimizing Resource Allocation Policies in Real-World Business Processes using Hybrid Process Simulation and Deep Reinforcement Learning

Resource allocation refers to the assignment of resources to activities for their execution within a business process at runtime. While resource allocation approaches are common in industries such as manufacturing, directly applying them to business processes remains a challenge.
Recently, techniques like Deep Reinforcement Learning (DRL) have been used to learn efficient resource allocation strategies to minimize the cycle time. While DRL has been proven to work well for simplified synthetic processes, its usefulness in real-world business processes remains untested, partly due to the challenging nature of realizing accurate simulation environments. To overcome this limitation, we propose DRL<sub>HSM</sub> that combines DRL with Hybrid simulation models (HSM). The HSM can accurately replicate the business process behavior so that we can assess the effectiveness of DRL in optimizing real-world business processes. 

In this repository, you will find the code to replicate the experiments in the paper "Optimizing Resource Allocation Policies in Real-World Business Processes using Hybrid Process Simulation and Deep Reinforcement Learning".

## Installation guide

To execute this code, install the following main packages in Python 3.10:

* sb3-contrib==2.2.1
* tensorflow==2.15.0
* simpy==4.0.1
* pm4py==2.7.5.2
* statsmodels==0.14.0
* prophet==1.1.5
* scikit-learn==1.2.1
* scipy==1.11.2
* pandas==1.5.3

or you can use the configuration file called requirements.txt to install all specified package versions.

```shell
pip install -r requirements.txt
```

Otherwise, with the anaconda system you can create an environment using the environment.yml
specification provided in the repository.

```shell
conda env create -f environment.yml
conda activate yourenvname
```


## Getting Started

The main file for training the models is `train_DRL.py`. The following variables may be used to train a model for a log in the desired configuration:

* `NAME_LOG`: the log for which a policy will be trained
* `N_TRACES`: the number of traces in each episode. Setting this variable to `from_input_data` will use the number of traces as used in this paper
* `CALENDAR`: a boolean to indicate if a calendar should be used
* `threshold`: to reduce the number of resources, a threshold value of 20 can be used. To use all resources, the value should be set to 0
* `postpone`: a boolean to indicate if the agent may use the postpone action. This variable was set to False in all experiments of this paper
* `reward_function`: the reward function. Currently, only the reward function `inverse_CT`, as used in this paper, is supported

It is also possible to simulate and optimize your own processes using DRL<sub>HSM</sub>. For specific information on how to implement your own processes, we refer to the [RIMS_Tool documentation](https://francescameneghello.github.io/RIMS_tool/index.html).


#### link log BPI17 https://drive.google.com/file/d/1e6y4w8Phnf_D7KpAG5jRi1ZiPFDQ8zRz/view?usp=sharing
