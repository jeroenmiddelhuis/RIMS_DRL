from collections import deque
from subprocess import call
import gymnasium as gym
import os
import numpy as np
from gym_env import gym_env
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
                                           net_arch=[dict(pi=[128],
                                                          vf=[128])])



if __name__ == '__main__':
    #if true, load model for a new round of training
    
    running_time = 5000
    num_cpu = 1
    load_model = False
    postpone_penalty = 0
    time_steps = 5e7 # Total timesteps
    n_steps = 51200 # Number of steps for each network update
    # Create log dir
    log_dir = f"./tmp/{int(time_steps)}_{n_steps}/" # Logging training results

    os.makedirs(log_dir, exist_ok=True)

    #print(f'Training agent for {config_type} with {time_steps} timesteps in updates of {n_steps} steps.')
    # Create and wrap the environment
    # Reward functions: 'AUC', 'case_task'
    env = gym_env()  # Initialize env

    env = Monitor(env, log_dir)

    # Create the model
    model = MaskablePPO(MaskableActorCriticPolicy, env, clip_range=0.1, learning_rate=linear_schedule(3e-4),n_steps=int(n_steps), batch_size=512, gamma=0.999, verbose=1)

    #Logging to tensorboard. To access tensorboard, open a bash terminal in the projects directory, activate the environment (where tensorflow should be installed) and run the command in the following line
    # tensorboard --logdir ./tmp/
    # then, in a browser page, access localhost:6006 to see the board
    # model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    # Train the agent
    callback = SaveOnBestTrainingRewardCallback(check_freq=int(n_steps), log_dir=log_dir)
    print(type(time_steps), type(callback))
    model.learn(total_timesteps=int(time_steps), callback=callback)

    model.save(f'{log_dir}/_{running_time}_final')