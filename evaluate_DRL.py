
from gym_env import gym_env
from sb3_contrib import MaskablePPO
import random
import glob
import pandas as pd
import pm4py
from pm4py.objects.log.util import sorting
import numpy as np
import scipy.stats as st
import json
from collections import defaultdict
import sys, os


def FIFO(state, tokens_pending, possible_action):
    if len(tokens_pending) > 0 and len(state['resource_available']) > 0:
        possible_ass = []
        ### Finding all possible assignments (resources available and tokens pending)
        for token in tokens_pending:
            act = tokens_pending[token][0]._next_activity
            for res in state['resource_available']:
                if (res, act) in possible_action:
                    possible_ass.append((token, act, res))
        if len(possible_ass) == 0:
            return None
        else:
            #In order of cycle time find the first possible allocation
            list_tokens_pending = list(tokens_pending.keys())
            i = 0
            next_ass = None
            while i < len(list_tokens_pending):
                token_id = list_tokens_pending[i]
                possible_ass_token = [a for a in possible_ass if a[0] == token_id]
                if len(possible_ass_token) > 0:
                    i = len(list_tokens_pending)
                    next_ass = random.choice(possible_ass_token)
                else:
                    i += 1
            return next_ass
    else:
        return None


def FIFO_case(state, tokens_pending, possible_action):
    if len(tokens_pending) > 0 and len(state['resource_available']) > 0:
        possible_ass = []
        ### Finding all possible assignments (resources available and tokens pending)
        for token in tokens_pending:
            act = tokens_pending[token][0]._next_activity
            for res in state['resource_available']:
                if (res, act) in possible_action:
                    possible_ass.append((token, act, res))
        if len(possible_ass) == 0:
            return None
        else:
            #In order of arrival time find the first possible allocation
            list_tokens_pending = list(tokens_pending.keys())
            next_ass = None
            while len(list_tokens_pending) > 0:
                first_case = min(list_tokens_pending)
                possible_ass_token = [a for a in possible_ass if a[0] == first_case]
                if len(possible_ass_token) > 0:
                    next_ass = random.choice(possible_ass_token)
                    list_tokens_pending = []
                else:
                    list_tokens_pending.remove(first_case)
            return next_ass
    else:
        return None


def RANDOM(state, tokens_pending, possible_action):
    next_ass = None
    #print('TOKEN pendings', tokens_pending, "N resource available", len(state['resource_available']))
    if len(tokens_pending) > 0 and len(state['resource_available']) > 0:
        possible_ass = []
        for token in tokens_pending:
            act = tokens_pending[token][0]._next_activity
            for res in state['resource_available']:
                if (res, act) in possible_action:
                    possible_ass.append((token, act, res))
        if len(possible_ass) > 0:
            next_ass = random.choice(possible_ass)
    return next_ass


def SPT(state, tokens_pendings, median_processing_time, possible_action):
    if len(tokens_pendings) > 0 and len(state['resource_available']) > 0:
        possible_ass = []
        for token in tokens_pendings:
            act = tokens_pendings[token][0]._next_activity
            for res in state['resource_available']:
                if (res, act) in possible_action:
                    possible_ass.append((token, act, res, median_processing_time[act][res]))
            if len(possible_ass) == 0:
                return None
            else:
                minimum_value = min(possible_ass, key=lambda t: t[3])
                min_ass_list = [elem for elem in possible_ass if elem[1] == minimum_value[1]]
                min_ass = random.choice(min_ass_list)
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
            #median_processing_time[a][res] = np.median(median_processing_time[a][res])
            median_processing_time[a][res] = np.mean(median_processing_time[a][res])
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
    print(x_bar, ci_lower, ci_upper)
    return ci_lower, ci_upper


def evaluate(NAME, POLICY, data, calendar, threshold):
    #"output/output_{}_{}_{}/simulated_log_{}_{}_{}".format(self.name_log, calendar, self.threshold, self.name_log, self.policy, str(i)) + ".csv", 'w'
    path = f'output/test/simulated_log_{NAME}_{POLICY}'# + '*.csv'
    #path = 'output/output_' + NAME + '/ENTIRE_LOG_CALENDAR/'+POLICY+'/simulated_log_' + NAME + '_' + POLICY + '*.csv'
    #all_file = glob.glob(path)
    all_file = [path + '_' + str(i) + '.csv' for i in range(0, 1)]
    cycle_time = []
    for file in all_file:
        simulated_log = pd.read_csv(file)
        simulated_log['start_time'] = pd.to_datetime(simulated_log['start_time'], format="%Y-%m-%d %H:%M:%S")
        simulated_log['end_time'] = pd.to_datetime(simulated_log['start_time'], format="%Y-%m-%d %H:%M:%S")
        simulated_log = pm4py.format_dataframe(simulated_log, case_id='id_case', activity_key='activity',
                                               timestamp_key='end_time', start_timestamp_key='start_time')
        simulated_log = pm4py.convert_to_event_log(simulated_log)
        simulated_log = sorting.sort_timestamp_log(simulated_log)
        for trace in simulated_log:
            cycle_time.append((trace[-1]['end_time'] - trace[0]['start_time']).total_seconds())
    #print('evaluate',cycle_time)
    ci_lower, ci_upper = confidence_interval(cycle_time)
    data[POLICY] = {'values': cycle_time, 'mean': [np.mean(cycle_time), ci_lower, ci_upper]}


def run_simulation(NAME_LOG, POLICY, N_SIMULATION, threshold=0, postpone=True, reward_function='inverse_CT', model=None, median_processing_time=None):
    env_simulator = gym_env(NAME_LOG, N_TRACES, CALENDAR, POLICY=POLICY, threshold=threshold, postpone=postpone, reward_function=reward_function)
    if POLICY in ['FIFO_activity', 'FIFO_case', 'SPT', 'RANDOM']:
        file = f'./output/result_{NAME_LOG}_C{CALENDAR}_T{THRESHOLD}_{POLICY}.json'
    else:
        file = f'./output/result_{NAME_LOG}_{N_TRACES}_C{CALENDAR}_T{THRESHOLD}_P{postpone}_{reward_function}_{POLICY}.json'
    cycle_times = {}
    for i in range(0, N_SIMULATION):
        obs = env_simulator.reset(i=i)
        isTerminated = False
        ##### simulation #####
        while not isTerminated:
            state = env_simulator.get_state()
            if POLICY == 'FIFO_activity':
                action = FIFO(env_simulator.simulation_process.get_state(), env_simulator.simulation_process.tokens_pending, env_simulator.output)
                state, reward, isTerminated, dones, info = env_simulator.step_baseline(action)
            elif POLICY == 'FIFO_case':
                action = FIFO_case(env_simulator.simulation_process.get_state(), env_simulator.simulation_process.tokens_pending, env_simulator.output)
                state, reward, isTerminated, dones, info = env_simulator.step_baseline(action)
            elif POLICY == 'SPT':
                action = SPT(env_simulator.simulation_process.get_state(), env_simulator.simulation_process.tokens_pending, median_processing_time, env_simulator.output)
                state, reward, isTerminated, dones, info = env_simulator.step_baseline(action)
            elif POLICY == 'RANDOM':
                action = RANDOM(env_simulator.simulation_process.get_state(), env_simulator.simulation_process.tokens_pending, env_simulator.output)
                state, reward, isTerminated, dones, info = env_simulator.step_baseline(action)
            else:
                mask = env_simulator.action_masks()
                action, _states = model.predict(state, action_masks=mask)
                state, reward, isTerminated, dones, info = env_simulator.step(action)
        cycle_times[f'simulation_{i}'] = env_simulator.cycle_times
        if i == 24:
            with open(file, 'w') as f:
                json.dump(cycle_times, f)
    with open(file, 'w') as f:
        json.dump(cycle_times, f)
    print('END simulation')


### Path DRL model
traces = {"BPI_Challenge_2012_W_Two_TS":1200,
        "confidential_1000":159,
        "ConsultaDataMining201618":181,
        "PurchasingExample":101,
        "BPI_Challenge_2017_W_Two_TS":5789,
        "Productions":45}
if len(sys.argv) > 1:
    NAME_LOG = sys.argv[1]#'BPI_Challenge_2017_W_Two_TS'
    if not sys.argv[2] == 'from_input_data':    
        N_TRACES = int(sys.argv[2])#2000
    else:
        N_TRACES = sys.argv[2]
    CALENDAR = True if sys.argv[3] == "True" else False
    THRESHOLD = int(sys.argv[4])
    postpone = True if sys.argv[5] == "True" else False
    reward_function = sys.argv[6]
    POLICY = sys.argv[7]

    if POLICY == 'None':
        model = MaskablePPO.load(os.getcwd() + "/tmp/" + f"{NAME_LOG}_{N_TRACES}_C{CALENDAR}_T{THRESHOLD}_P{postpone}_{reward_function}" + "/best_model.zip")
    N_SIMULATION = 100
else:
    model = None
    NAME_LOG = 'confidential_1000'#'PurchasingExample'#'BPI_Challenge_2017_W_Two_TS', 'confidential_1000', 'ConsultaDataMining201618','PurchasingExample'
    N_TRACES = 'from_input_data'
    CALENDAR = True
    THRESHOLD = 20
    POLICY = 'RANDOM'
    N_SIMULATION = 100

if POLICY == 'SPT':
    median_processing_time = retrieve_median_processing_time(NAME_LOG)
    run_simulation(NAME_LOG, POLICY, N_SIMULATION, THRESHOLD, median_processing_time=median_processing_time)
elif POLICY == 'FIFO_activity' or POLICY == 'FIFO_case' or POLICY == 'RANDOM':
    run_simulation(NAME_LOG, POLICY, N_SIMULATION, THRESHOLD)
else:
    run_simulation(NAME_LOG, POLICY, N_SIMULATION, threshold=THRESHOLD, postpone=postpone, reward_function=reward_function, model=model)


# data = {'FIFO_activity': {}, 'FIFO_case': {}, 'RANDOM': {}, 'SPT': {}, 'None': {}}
# evaluate(NAME_LOG, POLICY, data, calendar=CALENDAR, threshold=THRESHOLD)


# i number of simulations for log
# for NAME_LOG in ['ConsultaDataMining201618']:#'BPI_Challenge_2012_W_Two_TS', 'confidential_1000', 'Productions','PurchasingExample', , 'BPI_Challenge_2017_W_Two_TS'
#     for POLICY in ['FIFO_case', 'RANDOM', 'SPT']:
#         print(POLICY, NAME_LOG)
#         if POLICY == 'SPT':
#             median_processing_time = retrieve_median_processing_time(NAME_LOG)
#             run_simulation(NAME_LOG, POLICY, N_SIMULATION, THRESHOLD, median_processing_time=median_processing_time)
#         elif POLICY == 'FIFO_activity' or POLICY == 'FIFO_case' or POLICY == 'RANDOM':
#             run_simulation(NAME_LOG, POLICY, N_SIMULATION, THRESHOLD)
#         else:
#             pass
#             run_simulation(NAME_LOG, POLICY, N_SIMULATION, model=model)
        
        

# for log in ['ConsultaDataMining201618']:#['BPI_Challenge_2017_W_Two_TS', 'ConsultaDataMining201618', 'BPI_Challenge_2012_W_Two_TS', 'PurchasingExample', 'confidential_1000','Productions']:
#     for calendar in ['CALENDAR', 'NOT_CALENDAR']:
#         for threshold in ['0', '20']:
#             print(log, calendar, threshold)
#             data = {'None': {}}
#             evaluate(log, POLICY, data, log, calendar, threshold)
#             with open(f'output_no_postpone/output_{NAME_LOG}_{calendar}_{threshold}/results_'+NAME_LOG+'.json', 'w') as f:
#                 json.dump(data, f)

#for POLICY in ['FIFO_activity', 'FIFO_case', 'RANDOM', 'SPT']:
#    evaluate('BPI_Challenge_2012_W_Two_TS', POLICY, data)
