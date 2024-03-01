from collections import deque
from subprocess import call
import gymnasium as gym
import os
import numpy as np
from gym_env import gym_env
import datetime
import sys

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableMultiInputActorCriticPolicy
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure

from gymnasium.wrappers import normalize
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from callbacks import SaveOnBestTrainingRewardCallback, EvalPolicyCallback
from callbacks import custom_schedule, linear_schedule


class CustomPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128],
                                                          vf=[128, 128])])
if len(sys.argv) > 1:
    NAME_LOG = sys.argv[1]#'BPI_Challenge_2017_W_Two_TS'
    if not sys.argv[2] == 'from_input_data':    
        N_TRACES = int(sys.argv[2])#2000
    else:
        N_TRACES = sys.argv[2]
    CALENDAR = True if sys.argv[3] == "True" else False
    threshold = int(sys.argv[4])
    postpone = True if sys.argv[5] == "True" else False
    reward_function = sys.argv[6]
else:
    NAME_LOG = 'BPI_Challenge_2012_W_Two_TS'
    N_TRACES = 'from_input_data'
    CALENDAR = False
    threshold = 20
    postpone = False
    reward_function = 'inverse_CT'

#### to use BPI_Challenge_2017_W_Two_TS first download the entire log from 'https://drive.google.com/file/d/1juGeinUqaxkLBEmObIBYiRA3NqAMOcoN/view?usp=drive_link' and place it in the folder of the same name inside example

if __name__ == '__main__':
    #if true, load model for a new round of training
    load_model = False
    postpone_penalty = 0
    time_steps = 10240000
    n_steps = 12800# Number of steps for each network update
    # if NAME_LOG == 'BPI_Challenge_2017_W_Two_TS':
    #     n_steps = 38400
    # Create log dir
    now = datetime.datetime.now()
    #log_dir = f"./tmp/{NAME_LOG}_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}/"  # Logging training results
    log_dir = f"tmp/{NAME_LOG}_{N_TRACES}_C{CALENDAR}_T{threshold}_P{postpone}_{reward_function}/"
    os.makedirs(log_dir, exist_ok=True)

    #print(f'Training agent for {config_type} with {time_steps} timesteps in updates of {n_steps} steps.')
    # Create and wrap the environment
    # Reward functions: 'AUC', 'case_task'
    env_simulator = gym_env(NAME_LOG, N_TRACES, CALENDAR, threshold=threshold, postpone=postpone, reward_function=reward_function)  # Initialize env
    env = Monitor(env_simulator, log_dir)

    # Create the model
    model = MaskablePPO(CustomPolicy, env_simulator, clip_range=0.2, learning_rate=3e-5, n_steps=int(n_steps), batch_size=256, gamma=0.999, verbose=1)

    #Logging to tensorboard. To access tensorboard, open a bash terminal in the projects directory, activate the environment (where tensorflow should be installed) and run the command in the following line
    # tensorboard --logdir ./tmp/
    # then, in a browser page, access localhost:6006 to see the board
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    # checkpoint_callback = CheckpointCallback(
    #     save_freq=100000,
    #     save_path=log_dir,
    #     name_prefix="rl_model",
    #     save_replay_buffer=True,
    #     save_vecnormalize=True,
    # )
    # Train the agent
    eval_env = gym_env(NAME_LOG, N_TRACES, CALENDAR, threshold=threshold, postpone=postpone, reward_function=reward_function)  # Initialize env
    eval_env = Monitor(eval_env, log_dir)
    eval_callback = EvalPolicyCallback(check_freq=5*int(n_steps), nr_evaluations=5, log_dir=log_dir, eval_env=eval_env)

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=int(time_steps), callback=eval_callback)

    model.save(f'{log_dir}/model_final')


    
    #print(type(time_steps), type(callback))
    #model.learn(total_timesteps=int(time_steps), callback=callback)

    #model.save(f'{log_dir}/_{running_time}_final')