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

from callbacks import SaveOnBestTrainingRewardCallback
from callbacks import custom_schedule, linear_schedule


class CustomPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128],
                                                          vf=[128, 128])])

NAME_LOG = 'BPI_Challenge_2017_W_Two_TS'

#### to use BPI_Challenge_2017_W_Two_TS first download the entire log from 'https://drive.google.com/file/d/1juGeinUqaxkLBEmObIBYiRA3NqAMOcoN/view?usp=drive_link' and place it in the folder of the same name inside example

if __name__ == '__main__':
    #if true, load model for a new round of training
    
    running_time = 5000
    N_TRACES = 1000
    num_cpu = 1
    load_model = False
    postpone_penalty = 0
    time_steps = 5e6
    n_steps = 25600# Number of steps for each network update
    # Create log dir
    now = datetime.datetime.now()
    log_dir = f"tmp/{NAME_LOG}_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}/"  # Logging training results

    os.makedirs(log_dir, exist_ok=True)

    #print(f'Training agent for {config_type} with {time_steps} timesteps in updates of {n_steps} steps.')
    # Create and wrap the environment
    # Reward functions: 'AUC', 'case_task'
    env_simulator = gym_env(NAME_LOG, N_TRACES)  # Initialize env

    env = Monitor(env_simulator, log_dir)

    # Create the model
    model = MaskablePPO(CustomPolicy, env_simulator, clip_range=0.2, learning_rate=3e-5, n_steps=int(n_steps), batch_size=256, gamma=0.999, verbose=1)

    #Logging to tensorboard. To access tensorboard, open a bash terminal in the projects directory, activate the environment (where tensorflow should be installed) and run the command in the following line
    # tensorboard --logdir ./tmp/
    # then, in a browser page, access localhost:6006 to see the board
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    # Train the agent
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=log_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    model.learn(total_timesteps=int(time_steps), callback=checkpoint_callback)

    model.save(f'{log_dir}/_{running_time}_final')

    # Train the agent
    #callback = SaveOnBestTrainingRewardCallback(check_freq=int(n_steps), log_dir=log_dir)
    #print(type(time_steps), type(callback))
    #model.learn(total_timesteps=int(time_steps), callback=callback)

    #model.save(f'{log_dir}/_{running_time}_final')