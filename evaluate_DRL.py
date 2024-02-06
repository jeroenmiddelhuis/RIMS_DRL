import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_env import gym_env
from sb3_contrib import MaskablePPO
import random
import glob
import pandas as pd
import pm4py
from pm4py.objects.log.util import sorting
import numpy as np
import scipy.stats as st
from pm4py.objects.log.obj import EventLog, Trace


def FIFO(state, tokens):
    if len(tokens) > 0 and len(state['resource_available'])>0:
        token_id = max(tokens.items(), key=lambda x: x[1][1])[0]
        activity = tokens[token_id][0]._next_activity
        res = random.choice(state['resource_available'])
        return (token_id, activity, res)
    else:
        return None


def SPT(state, tokens, median_processing_time, possible_action):
    if len(tokens) > 0 and len(state['resource_available']) > 0:
        possible_ass = []
        for token in tokens:
            act = tokens[token][0]._next_activity
            for res in state['resource_available']:
                if (res, act) in possible_action:
                    possible_ass.append((token, act, res, median_processing_time[act][res]))
            if len(possible_ass) == 0:
                return None
            else:
                min_ass = min(possible_ass, key=lambda t: t[3])
                return (min_ass[0], min_ass[1], min_ass[2])
    else:
        return None


def retrieve_median_processing_time(NAME_LOG):
    path = './example/' + NAME_LOG + '/' + NAME_LOG + '.xes'
    log = pm4py.read_xes(path)
    activities = pm4py.get_event_attribute_values(log, "concept:name")
    median_processing_time = {a: {} for a in activities}
    for index, row in log.iterrows():
        res = row['org:resource']
        activity = row['concept:name']
        time = (row['time:timestamp'] - row['start:timestamp']).total_seconds()
        if res in median_processing_time[activity]:
            median_processing_time[activity][res].append(time)
        else:
            median_processing_time[activity][res] = [time]

    for a in median_processing_time:
        for res in median_processing_time[a]:
            median_processing_time[a][res] = np.median(median_processing_time[a][res])
    return median_processing_time


def confidence_interval(data):
    n = len(data)
    C = 0.95
    alpha = 1 - C
    tails = 2
    q = 1 - (alpha / tails)
    dof = n - 1
    t_star = st.t.ppf(q, dof)
    x_bar = np.mean(data)
    s = np.std(data, ddof=1)
    ci_upper = x_bar + t_star * s / np.sqrt(n)
    ci_lower = x_bar - t_star * s / np.sqrt(n)

    print('CI ', ci_lower, ci_upper)


def evaluate(NAME, POLICY):
    path = 'output/output_' + NAME + '/simulated_log_' + NAME + '_' + POLICY + '*.csv'
    all_file = glob.glob(path)
    cycle_time = []
    for file in all_file:
        simulated_log = pd.read_csv(file)
        simulated_log = pm4py.format_dataframe(simulated_log, case_id='id_case', activity_key='activity',
                                               timestamp_key='end_time', start_timestamp_key='start_time')
        simulated_log = pm4py.convert_to_event_log(simulated_log)
        simulated_log = sorting.sort_timestamp_log(simulated_log)
        for trace in simulated_log:
            cycle_time.append((trace[-1]['end_time'] - trace[0]['start_time']).total_seconds())
    print('MEAN CYCLE', np.mean(cycle_time))
    confidence_interval(cycle_time)


def run_simulation(NAME_LOG, POLICY, N_SIMULATION, model=None, median_processing_time=None):
    for i in range(0, N_SIMULATION):
        env_simulator = gym_env(NAME_LOG, POLICY, True, i)
        obs = env_simulator.reset()
        isTerminated = False
        ##### simulation #####
        while not isTerminated:
            state = env_simulator.get_state()
            if POLICY == 'FIFO':
                action = FIFO(env_simulator.simulation_process.get_state(), env_simulator.simulation_process.tokens_pending)
                state, reward, isTerminated, dones, info = env_simulator.step_baseline(action)
            elif POLICY == 'SPT':
                action = SPT(env_simulator.simulation_process.get_state(), env_simulator.simulation_process.tokens_pending, median_processing_time, env_simulator.output)
                state, reward, isTerminated, dones, info = env_simulator.step_baseline(action)
            else:
                mask = env_simulator.action_masks()
                action, _states = model.predict(state, action_masks=mask)
                state, reward, isTerminated, dones, info = env_simulator.step(action)
        print('END simulation')


### Path DRL model
#model = MaskablePPO.load("/tmp/20000000_25600_2023-11-21 13:00:43.547943/rl_model_1300000_steps.zip/")
model = None

NAME_LOG = 'BPI_Challenge_2017_W_Two_TS'
POLICY = 'SPT'
N_SIMULATION = 1
## i number of simulations for log
if POLICY == 'SPT':
    median_processing_time = retrieve_median_processing_time(NAME_LOG)
    run_simulation(NAME_LOG, POLICY, N_SIMULATION, median_processing_time=median_processing_time)
elif POLICY == 'FIFO':
    run_simulation(NAME_LOG, POLICY, N_SIMULATION)
else:
    run_simulation(NAME_LOG, POLICY, N_SIMULATION, model=model)

evaluate(NAME_LOG, POLICY)


