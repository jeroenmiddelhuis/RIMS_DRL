import pandas as pd
import pm4py
from pm4py.objects.log.util import sorting
import numpy as np
import glob
from scipy.stats import wasserstein_distance
from sklearn import preprocessing
import scipy.stats as st
import pathlib


def convert_log(path):
    dataframe = pd.read_csv(path, sep=',')
    dataframe = pm4py.format_dataframe(dataframe, case_id='caseid', activity_key='task', timestamp_key='time:timestamp')
    log = pm4py.convert_to_event_log(dataframe)
    return log


def define_cycle(log):
    log = sorting.sort_timestamp_log(log)
    cycle_time_real = []
    for trace in log:
        cycle_time_real.append((trace[-1]['time:timestamp']-trace[0]['start:timestamp']).total_seconds())
    cycle_time_real.sort()
    return cycle_time_real


def compute_MAE(log, log1):
    cycle1 = define_cycle(log)
    cycle2 = define_cycle(log1)
    diff = len(cycle1) - len(cycle2)
    if diff != 0:
        cycle2 = cycle2 + [0]*diff
        cycle2.sort()
    mae = []
    for i in range(0, len(cycle1)):
        mae.append(abs(cycle1[i]-cycle2[i]))
    return np.mean(mae)


def extract_time_activity(log, task):
    time_activity = dict()
    for trace in log:
        for event in trace:
            if event['concept:name'] == task:
                key = event['start:timestamp'].replace(minute=0, second=0)
                if key in time_activity:
                    time_activity[key] += (event['time:timestamp']-event['start:timestamp']).total_seconds()
                else:
                    time_activity[key] = (event['time:timestamp']-event['start:timestamp']).total_seconds()
    return time_activity


def extract_time_log(log, dates):
    time_activity = dict()
    for d in dates:
        time_activity[d] = 0

    for trace in log:
        for event in trace:
            key1 = event['start:timestamp'].replace(minute=0, second=0)
            time_activity[key1] += 1
            key2 = event['time:timestamp'].replace(minute=0, second=0)
            time_activity[key2] += 1
    return time_activity


def extract_set_date(log, log1):
    dates = set()
    for trace in log:
        for event in trace:
            start = event['start:timestamp'].replace(minute=0, second=0)
            end = event['time:timestamp'].replace(minute=0, second=0)
            dates.add(start)
            dates.add(end)
    for trace in log1:
        for event in trace:
            start = event['start:timestamp'].replace(minute=0, second=0)
            end = event['time:timestamp'].replace(minute=0, second=0)
            dates.add(start)
            dates.add(end)

    dates = list(dates)
    dates.sort()
    return dates


def normalize(times):
    values = list(times.values())
    max_v = max(values)
    for i in range(0, len(values)):
        values[i] = values[i]/max_v
    return values


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


def evaluation_sim(NAME_EXP, type):
    file_path = str(pathlib.Path().resolve())
    path = file_path + '/' + NAME_EXP + '/results/' + type
    all_file = glob.glob(path + '/sim*')
    path_test = path + '/tst_' + NAME_EXP + '.csv'
    real_tst = convert_log(path_test)
    MAE = dict()
    EMD_normalize = dict()
    LEN = dict()

    for idx, file in enumerate(all_file):
        sim_tst = convert_log(file)

        dates = extract_set_date(sim_tst, real_tst)

        real = extract_time_log(real_tst, dates)
        sim = extract_time_log(sim_tst, dates)

        real = list(real.values())
        sim = list(sim.values())

        LEN[idx] = len(sim_tst)
        EMD_normalize[idx] = wasserstein_distance(preprocessing.normalize([real])[0], preprocessing.normalize([sim])[0])
        MAE[file] = compute_MAE(real_tst, sim_tst)

    print('MEAN MAE', np.mean(list(MAE.values())))
    if len(all_file) > 4:
        confidence_interval(list(MAE.values()))
    print('NORMALIZE emd', np.mean(list(EMD_normalize.values())))
    if len(all_file) > 4:
        confidence_interval(list(EMD_normalize.values()))
