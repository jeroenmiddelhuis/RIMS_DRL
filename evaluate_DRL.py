import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_env import gym_env
from sb3_contrib import MaskablePPO

model = MaskablePPO.load("/Users/francescameneghello/Documents/GitHub/rl-rims/tmp/20000000_25600_2023-11-21 13:00:43.547943/rl_model_1300000_steps.zip/")

NAME_LOG = 'BPI_Challenge_2012_W_Two_TS'

for i in range(0, 1):
    env_simulator = gym_env(NAME_LOG, i)
    obs = env_simulator.reset()
    isTerminated = False
    ##### simulation #####
    while not isTerminated:
        state = env_simulator.get_state()
        mask = env_simulator.action_masks()
        action, _states = model.predict(state, action_masks=mask)
        state, reward, isTerminated, dones, info = env_simulator.step(action)
    print('END simulation')
