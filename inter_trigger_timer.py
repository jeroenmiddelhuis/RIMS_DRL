"""
Class to manage the arrivals times of tokes in the process.
"""

import numpy as np
from datetime import datetime, timedelta
from parameters import Parameters
from process import SimulationProcess
import custom_function as custom
import pandas as pd
from prophet.serialize import model_to_json, model_from_json
from numpy.random import triangular as triang
import json

class InterTriggerTimer(object):

    def __init__(self, params: Parameters, process: SimulationProcess, start: datetime, n_traces: int):
        self._process = process
        self._start_time = start
        self._type = params.INTER_TRIGGER['type']
        self._previous_date = start
        self._previous = 0
        if self._type == 'distribution':
            """Define the distribution of token arrivals from specified in the file json"""
            self.name_distribution = params.INTER_TRIGGER['name']
            self.params = params.INTER_TRIGGER['parameters']
        else:
            self.params = params
        self._interval = 0
        self.generate_arrivals(n_traces, start)

    def get_next_arrival(self, env, case, name_log, calendar):
        """Generate a new arrival from the distribution and check if the new token arrival is inside calendar,
        otherwise wait for a suitable time."""
        next = 0
        if self._type == 'distribution':
            resource = self._process._get_resource('TRIGGER_TIMER')
            arrival = getattr(np.random, self.name_distribution)(**self.params, size=1)[0]
            if calendar and resource._get_calendar():
                stop = resource.to_time_schedule(self._start_time + timedelta(seconds=env.now + arrival))
                self._interval = stop + arrival
            else:
                self._interval = arrival
        elif self._type == 'custom':
            #self._interval += self.custom_arrival(case, self._previous_date, name_log)
            self._previous = (self.arrivals.iloc[case][0] - self._start_time).total_seconds()
        else:
            raise ValueError('ERROR: Invalid arrival times generator')
        #self._previous += self._interval
        #self._previous_date = self._start_time + timedelta(seconds=self._interval)
        return self._previous

    def generate_arrivals(self, num_instances, start_time):
        with open(self.params.MODEL_PROPHET, 'r') as fin:
            m = model_from_json(json.load(fin))  # Load model
        with open(self.params.METADATA_PROPHET) as file:
            max_cap = json.load(file)['max_cap']
        #start_time = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
        gen = list()
        n_gen_inst = 0
        while n_gen_inst < num_instances:
            future = pd.date_range(start=start_time, end=(start_time + timedelta(days=30)), freq='H').to_frame(
                name='ds', index=False)
            future['cap'] = max_cap
            future['floor'] = 0
            forecast = m.predict(future)

            def rand_value(x):
                raw_val = triang(x.yhat_lower, x.yhat, x.yhat_upper, size=1)
                raw_val = raw_val[0] if raw_val[0] > 0 else 0
                return raw_val

            forecast['gen'] = forecast.apply(rand_value, axis=1)
            forecast['gen_round'] = np.ceil(forecast['gen'])
            n_gen_inst += np.sum(forecast['gen_round'])
            gen.append(forecast[forecast.gen_round > 0][['ds', 'gen_round']])
            start_time = forecast.ds.max()

        gen = pd.concat(gen, axis=0, ignore_index=True)

        def pp(start, n):
            start_u = int(start.value // 10 ** 9)
            end_u = int((start + timedelta(hours=1)).value // 10 ** 9)
            return pd.to_datetime(np.random.randint(start_u, end_u, int(n)), unit='s').to_frame(name='timestamp')

        gen_cases = list()
        for row in gen.itertuples(index=False):
            gen_cases.append(pp(row.ds, row.gen_round))
        times = pd.concat(gen_cases, axis=0, ignore_index=True)
        self.arrivals = times.iloc[:num_instances]
        self.arrivals = self.arrivals.sort_values(by='timestamp')

    def custom_arrival(self, case, previous, name_log):
        """
        Call to the custom functions in the file custom_function.py.
        """
        return custom.custom_arrivals_time(case, previous, name_log)

