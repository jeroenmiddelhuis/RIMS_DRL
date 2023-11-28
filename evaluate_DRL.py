import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_env import gym_env
from sb3_contrib import MaskablePPO
import random


### FIX right path
#model = MaskablePPO.load("/tmp/20000000_25600_2023-11-21 13:00:43.547943/rl_model_1300000_steps.zip/")

NAME_LOG = 'slow_server'
POlICY = 'SPT'

median_processing_time = {"Activity E": {"R5": 0.97040, "R6": 1.24766}, "Activity F": {"R5": 1.10903, "R6": 2.07944}}

def FIFO(state, tokens):
    if len(tokens) > 0 and len(state['resource_available'])>0:
        token_id = max(tokens.items(), key=lambda x: x[1][1])[0]
        activity = tokens[token_id][0]._next_activity
        res = random.choice(state['resource_available'])
        return (token_id, activity, res)
    else:
        return None

def SPT(state, tokens):
    if len(tokens) > 0 and len(state['resource_available']) > 0:
        possible_ass = []
        for token in tokens:
            act = tokens[token][0]._next_activity
            for res in state['resource_available']:
                possible_ass.append((token, act, res, median_processing_time[act][res]))
        min_ass = min(possible_ass, key = lambda t: t[3])
        return (min_ass[0], min_ass[1], min_ass[2])
    else:
        return None


for i in range(0, 1):
    env_simulator = gym_env(NAME_LOG, i)
    obs = env_simulator.reset()
    isTerminated = False
    ##### simulation #####
    while not isTerminated:
        state = env_simulator.get_state()
        if POlICY == 'FIFO':
            action = FIFO(env_simulator.simulation_process.get_state(), env_simulator.simulation_process.tokens_pending)
            state, reward, isTerminated, dones, info = env_simulator.step_baseline(action)
        elif POlICY == 'SPT':
            action = SPT(env_simulator.simulation_process.get_state(), env_simulator.simulation_process.tokens_pending)
            state, reward, isTerminated, dones, info = env_simulator.step_baseline(action)
        else:
            mask = env_simulator.action_masks()
            action, _states = model.predict(state, action_masks=mask)
            state, reward, isTerminated, dones, info = env_simulator.step(action)
    print('END simulation')


