import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from typing import Callable

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-5:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

class EvalPolicyCallback(BaseCallback):
    def __init__(self, check_freq: int, nr_evaluations: int, log_dir: str, eval_env, model_name: str = 'best_model', verbose: int = 1):
        super(EvalPolicyCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.nr_evaluations = nr_evaluations
        self.log_dir = log_dir
        self.eval_env = eval_env
        self.save_path = os.path.join(log_dir, model_name)
        self.best_mean_cycle_time = np.inf
        self.prev_steps = 0
        self.episode_lengths = []

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if (self.n_calls - 1) % self.check_freq == 0 and self.n_calls > 1: # After model update
            print('\n-------- Evaluating current policy --------')
            mean_cycle_times = []
            total_rewards = []
            for epoch in range(self.nr_evaluations):
                state, _ = self.eval_env.reset()
                isTerminated = False

                while isTerminated == False:
                    action, _state = self.model.predict(state, action_masks=self.eval_env.action_masks())
                    state, reward, isTerminated, _, _ = self.eval_env.step(action)

                mean_cycle_times.append(self.eval_env.mean_cycle_time)
                total_rewards.append(self.eval_env.mean_cycle_time)

            mean_cycle_time = sum(mean_cycle_times)/len(mean_cycle_times)
            total_reward = sum(total_rewards)/len(total_rewards)

            print(f"Num timesteps: {self.num_timesteps}")
            print(f"Best mean cycle time: {self.best_mean_cycle_time:.2f} - Last mean cycle time: {mean_cycle_time:.2f}")

            if mean_cycle_time < self.best_mean_cycle_time:
                self.best_mean_cycle_time = mean_cycle_time
                # Example for saving best model
                if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                self.model.save(self.save_path)
            print('---- Finished evaluating current policy ----\n')
        return True   


def custom_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining > 0.8: #0.95
            return initial_value
        elif progress_remaining > 0.5: #0.9
            return initial_value * 0.5
        else:
            return initial_value * 0.1

    return func

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func